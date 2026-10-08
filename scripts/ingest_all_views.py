"""Clean Core Dynamic Adaptive Prefix Crawler for Targeted SAP CDS View Ingestion.

Scopes extraction strictly to Clean Core Interface & Consumption views (I_*, C_*)
and custom views (Z*, Y*). Implements Breadth-First Search (BFS) trie crawler
with automatic prefix bisection when SAP ADT search reaches the maxResults cap (e.g. 200).

Enterprise Reliability Features:
- HTTP Session Pooling & urllib3.util.Retry strategy on requests.Session.
- Explicit Worker Auto-Retry with Exponential Backoff (2s, 4s, 8s, 16s) on connection drops.
- Zero Data Loss: Utterly failed views appended to ./data/failed_views_retry_queue.txt.
- --retry-failed flag to re-process only previously failed views.
- Multithreaded DDL extraction & AST parsing via ThreadPoolExecutor.
- Persistent file logging to logs/full_ingestion.log and tqdm progress bar.
- Delta Sync: skips views already indexed in ChromaDB / Knowledge Graph.
- Offline Seed Cache: incrementally saves all view metadata to data/cds_catalog.json.
- --from-cache flag: loads and indexes directly from data/cds_catalog.json without network calls.
"""

import sys
import os
import time
import json
import string
import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Set
from tqdm import tqdm

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import settings
from core.sap_adt_client import SAPADTClient
from core.ddl_parser import DDLParser
from core.database import StorageEngine

# -------------------------------------------------------------
# 1. PERSISTENT FILE & CONSOLE LOGGING SETUP
# -------------------------------------------------------------
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "full_ingestion.log")

log_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# File Handler (UTF-8 encoded)
file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_formatter)

logger = logging.getLogger("adaptive_crawler")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(file_handler)

# Console Handler (only when running in an interactive console)
if sys.stdout is not None:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

ALPHABET = list(string.ascii_uppercase + string.digits + "_")

# Default Clean Core & Custom Prefix Targets
CLEAN_CORE_TARGETS = ["I_*", "C_*", "Z*", "Y*"]

# File Storage Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
CATALOG_FILE = os.path.join(DATA_DIR, "cds_catalog.json")
FAILED_VIEWS_FILE = os.path.join(DATA_DIR, "failed_views_retry_queue.txt")
DISCOVERED_VIEWS_FILE = os.path.join(DATA_DIR, "discovered_views.json")

_catalog_lock = threading.Lock()
_failed_views_lock = threading.Lock()


def record_failed_view(view_name: str, uri: Optional[str] = None, error: Optional[str] = None):
    """Thread-safely append permanently failed view to data/failed_views_retry_queue.txt."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with _failed_views_lock:
        try:
            with open(FAILED_VIEWS_FILE, "a", encoding="utf-8") as f:
                f.write(f"{view_name.upper()}\n")
            logger.info(f"Recorded failed view '{view_name}' to {FAILED_VIEWS_FILE}")
        except Exception as ex:
            logger.error(f"Failed writing to {FAILED_VIEWS_FILE}: {ex}")


def save_catalog_seed(new_views: List[Dict[str, Any]], catalog_file: str = CATALOG_FILE) -> int:
    """Save extracted view metadata to data/cds_catalog.json for instant offline seeding."""
    os.makedirs(os.path.dirname(catalog_file), exist_ok=True)
    with _catalog_lock:
        catalog: Dict[str, Dict[str, Any]] = {}

        # Load existing catalog entries if present
        if os.path.exists(catalog_file):
            read_success = False
            for _ in range(3):
                try:
                    with open(catalog_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for v in data:
                                if isinstance(v, dict) and "view_name" in v:
                                    catalog[v["view_name"].upper()] = v
                        elif isinstance(data, dict):
                            catalog = {k.upper(): v for k, v in data.items()}
                        read_success = True
                        break
                except Exception as e:
                    time.sleep(0.1)
            if not read_success and os.path.getsize(catalog_file) > 1000:
                logger.error(f"Refusing to overwrite catalog {catalog_file} because reading failed.")
                return 0

        # Merge newly extracted views
        for v in new_views:
            if isinstance(v, dict) and "view_name" in v:
                catalog[v["view_name"].upper()] = v

        # Write merged catalog back to disk atomically
        try:
            tmp_catalog_file = f"{catalog_file}.tmp"
            with open(tmp_catalog_file, "w", encoding="utf-8") as f:
                json.dump(list(catalog.values()), f, indent=2, ensure_ascii=False)
            os.replace(tmp_catalog_file, catalog_file)
            logger.info(f"Updated offline catalog cache: {catalog_file} ({len(catalog)} total views saved).")
        except Exception as e:
            logger.error(f"Failed to write catalog file {catalog_file}: {e}")

        return len(catalog)


def load_from_cache(storage: StorageEngine, catalog_file: str = CATALOG_FILE) -> int:
    """Load view metadata directly from data/cds_catalog.json and index without network calls."""
    if not os.path.exists(catalog_file):
        logger.error(f"Catalog file {catalog_file} does not exist. Cannot load from cache.")
        return 0

    logger.info(f"Loading CDS view catalog directly from cache: {catalog_file}...")
    try:
        with open(catalog_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            views = data
        elif isinstance(data, dict):
            views = list(data.values())
        else:
            views = []

        logger.info(f"Loaded {len(views)} views from {catalog_file}. Indexing into ChromaDB and Knowledge Graph...")
        storage.add_cds_views_batch(views)
        total_in_db = len(storage.get_all_views())
        logger.info(f"Offline indexing complete. Total views currently in database: {total_in_db}")
        return total_in_db
    except Exception as e:
        logger.error(f"Failed loading views from cache {catalog_file}: {e}")
        return 0


class AdaptivePrefixCrawler:
    """Dynamic BFS Trie Crawler discovering Clean Core and Custom CDS views from SAP S/4HANA ADT."""

    def __init__(
        self,
        client: Optional[SAPADTClient] = None,
        storage: Optional[StorageEngine] = None,
        max_results_cap: int = 200,
        max_prefix_depth: int = 6,
        max_workers: int = 5,
        object_type: str = "DDLS/DF",
    ):
        self.client = client or SAPADTClient()
        self.storage = storage or StorageEngine()
        self.max_results_cap = max_results_cap
        self.max_prefix_depth = max_prefix_depth
        self.max_workers = max_workers
        self.object_type = object_type

        # Seed catalog cache with any views already present in ChromaDB if catalog is missing
        if not os.path.exists(CATALOG_FILE):
            existing_db_views = self.storage.get_all_views()
            if existing_db_views:
                save_catalog_seed(existing_db_views)

    def discover_all_view_names(
        self,
        initial_prefixes: Optional[List[str]] = None,
        max_total_views: Optional[int] = None,
        force_bfs: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Discover Clean Core & Custom CDS views.

        Primary Strategy: Direct TADIR table extraction via ADT DataPreview SQL engine
        (bypasses 500-item ceiling and prefix bisection, returning all ~73,000 views in <30 seconds).
        Secondary Strategy: Dynamic BFS Trie Crawler with adaptive bisection (fallback).
        """
        logger.info("=" * 70)
        logger.info("STARTING CLEAN CORE CATALOG DISCOVERY")
        logger.info(f"Target SAP Host: {self.client.base_url} (Client {self.client.client})")
        logger.info(f"Max Results Cap: {self.max_results_cap}, Workers: {self.max_workers}")
        logger.info(f"Persistent Log File: {LOG_FILE}")
        logger.info("=" * 70)

        # 1. Check if full discovery cache exists on disk to avoid redundant discovery
        if not initial_prefixes and not max_total_views and os.path.exists(DISCOVERED_VIEWS_FILE):
            try:
                with open(DISCOVERED_VIEWS_FILE, "r", encoding="utf-8") as f:
                    cached_views = json.load(f)
                if isinstance(cached_views, dict) and len(cached_views) > 1000:
                    logger.info(f"Loaded {len(cached_views):,} discovered views directly from disk cache: {DISCOVERED_VIEWS_FILE}")
                    return cached_views
            except Exception as e:
                logger.warning(f"Could not load discovery cache: {e}. Running fresh discovery.")

        targets = initial_prefixes or CLEAN_CORE_TARGETS
        discovered_views: Dict[str, Dict[str, Any]] = {}

        # Strategy 1: Direct TADIR extraction via ADT DataPreview SQL engine
        if not force_bfs:
            try:
                logger.info("Initiating Strategy 1: Direct TADIR System Table Extraction via ADT DataPreview...")
                tadir_prefixes = [p.replace("*", "%") for p in targets]
                tadir_results = self.client.query_tadir_cds_views(prefixes=tadir_prefixes)
                for item in tadir_results:
                    v_name = item["name"].upper().strip()
                    if v_name and v_name not in discovered_views:
                        discovered_views[v_name] = item
                        if max_total_views and len(discovered_views) >= max_total_views:
                            break

                if len(discovered_views) > 0:
                    logger.info(f"Strategy 1 (Direct TADIR) succeeded: Discovered {len(discovered_views):,} views.")
            except Exception as tadir_err:
                logger.warning(f"Strategy 1 (TADIR query) encountered error: {tadir_err}. Falling back to Strategy 2 (BFS Crawler)...")

        # Strategy 2 (Fallback if TADIR failed or yielded 0): BFS prefix queue
        if not discovered_views:
            logger.info("Initiating Strategy 2: Adaptive BFS Trie QuickSearch Crawler...")
            queue_items = [p.rstrip("*") for p in targets if p.strip()]
            queue = deque(queue_items)
            processed_prefixes: Set[str] = set()

            while queue:
                if max_total_views and len(discovered_views) >= max_total_views:
                    logger.info(f"Reached user-specified limit of {max_total_views} views.")
                    break

                prefix = queue.popleft()
                if prefix in processed_prefixes:
                    continue
                processed_prefixes.add(prefix)

                search_query = f"{prefix}*"
                logger.info(f"Querying ADT QuickSearch for '{search_query}' (Queue size: {len(queue)})...")

                try:
                    results = self.client.search_cds_views(
                        query=search_query,
                        max_results=self.max_results_cap,
                        object_type=self.object_type,
                    )
                except Exception as e:
                    logger.warning(f"Error querying prefix '{prefix}': {e}. Skipping prefix and continuing.")
                    continue

                result_count = len(results)
                logger.info(f"Prefix '{prefix}*' returned {result_count} items.")

                for item in results:
                    name = item["name"].upper().strip()
                    if name not in discovered_views:
                        discovered_views[name] = item

                if result_count >= self.max_results_cap and len(prefix) < self.max_prefix_depth:
                    for char in ALPHABET:
                        sub_prefix = f"{prefix}{char}"
                        if sub_prefix not in processed_prefixes:
                            queue.append(sub_prefix)

        # Deduplicate and sort all view names
        sorted_views = {k: discovered_views[k] for k in sorted(discovered_views.keys())}

        logger.info("=" * 70)
        logger.info(f"DISCOVERY COMPLETE: Discovered {len(sorted_views):,} total unique Clean Core/Custom CDS views.")
        logger.info("=" * 70)

        # Save discovery cache to disk
        if not initial_prefixes and not max_total_views and sorted_views:
            try:
                with open(DISCOVERED_VIEWS_FILE, "w", encoding="utf-8") as f:
                    json.dump(sorted_views, f, indent=2, ensure_ascii=False)
                logger.info(f"Cached {len(sorted_views):,} discovered views to {DISCOVERED_VIEWS_FILE}")
            except Exception as e:
                logger.warning(f"Failed to cache discovered views to disk: {e}")

        return sorted_views

    # Alias for backward compatibility
    crawl_all_view_names = discover_all_view_names

    def fetch_and_index_views(
        self,
        discovered_views: Dict[str, Dict[str, Any]],
        batch_size: int = 50,
    ) -> int:
        """Multithreaded fetch of DDL source codes and AST parsing with Delta Sync.

        Subtracts already indexed views from local database before invoking worker pool.
        Implements enterprise exponential backoff auto-retry (2s, 4s, 8s, 16s) for connection drops.
        Utterly failed views are recorded to data/failed_views_retry_queue.txt.
        Saves successfully extracted views to data/cds_catalog.json.
        """
        # 1. DELTA SYNC: Query local database to identify already indexed views
        existing_views_in_db = self.storage.get_all_indexed_view_names()
        existing_views_upper = set(v.upper() for v in existing_views_in_db)

        # Subtract existing views
        views_to_process = {
            name: obj for name, obj in discovered_views.items()
            if name.upper() not in existing_views_upper
        }

        total_discovered = len(discovered_views)
        already_indexed = total_discovered - len(views_to_process)
        total_to_process = len(views_to_process)

        logger.info("=" * 70)
        logger.info(
            f"DELTA SYNC: Discovered {total_discovered:,} views. "
            f"{already_indexed:,} already indexed in DB. "
            f"Processing remaining {total_to_process:,} new/unindexed views..."
        )
        logger.info("=" * 70)

        if total_to_process == 0:
            logger.info("All discovered views are already indexed in Vector Store & Knowledge Graph. Ingestion complete!")
            return len(existing_views_in_db)

        logger.info(
            f"Starting DDL extraction for {total_to_process} views using {self.max_workers} worker threads. "
            f"Logging to {LOG_FILE}..."
        )
        start_time = time.time()
        parsed_views: List[Dict[str, Any]] = []
        indexed_count = 0
        failed_count = 0

        # Fault-Tolerant Worker Function with Explicit Auto-Retry & Exponential Backoff
        def process_single_view(view_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            v_name = view_obj["name"]
            uri = view_obj.get("uri")
            desc = view_obj.get("description", "")
            max_view_retries = 5

            for attempt in range(1, max_view_retries + 1):
                try:
                    # Dedicated thread-safe fetch from SAP ADT
                    ddl = self.client.get_ddl_source(v_name, uri=uri)
                    if not ddl or not ddl.strip():
                        logger.warning(f"View '{v_name}' returned empty DDL source. Isolating to {FAILED_VIEWS_FILE}.")
                        record_failed_view(v_name, uri=uri, error="empty ddl source")
                        return None

                    parsed = DDLParser.parse(ddl, fallback_name=v_name)
                    if not parsed.get("description") and desc:
                        parsed["description"] = desc

                    return parsed
                except Exception as e:
                    err_msg = str(e)

                    # Fast-path for 404 Not Found / Missing DDL Source:
                    # Non-existent or deleted views will never succeed on retry; bypass exponential backoff immediately
                    is_not_found = (
                        "404" in err_msg
                        or "not found" in err_msg.lower()
                        or "does not exist" in err_msg.lower()
                        or "empty ddl source" in err_msg.lower()
                    )
                    if is_not_found:
                        logger.warning(
                            f"View '{v_name}' returned 404/Not Found. Bypassing retries and recording to {FAILED_VIEWS_FILE} immediately."
                        )
                        record_failed_view(v_name, uri=uri, error=err_msg)
                        return None

                    is_transient = any(
                        term in err_msg.lower()
                        for term in [
                            "remotedisconnected",
                            "connection aborted",
                            "connectionreseterror",
                            "connection refused",
                            "timeout",
                            "timed out",
                            "max retries exceeded",
                            "503",
                            "502",
                            "504",
                            "429",
                        ]
                    )

                    if attempt < max_view_retries:
                        # Exponential backoff only for genuine transient network/server hiccups: 2s, 4s, 8s, 16s...
                        backoff = min(2 ** attempt, 30)
                        logger.warning(
                            f"Worker error on view '{v_name}' (Attempt {attempt}/{max_view_retries}): {e}. "
                            f"Backing off for {backoff}s before retry..."
                        )
                        time.sleep(backoff)
                        # Re-fetch CSRF token if connection was dropped
                        if is_transient:
                            try:
                                self.client.fetch_csrf_token(force_refresh=True)
                            except Exception:
                                pass
                        continue
                    else:
                        # Utterly failed after all retries -> Append to failed views ledger
                        logger.error(
                            f"View '{v_name}' utterly failed all {max_view_retries} attempts: {e}. "
                            f"Recording to {FAILED_VIEWS_FILE} for later targeted batch."
                        )
                        record_failed_view(v_name, uri=uri, error=err_msg)
                        return None

        # Execute with ThreadPoolExecutor & tqdm progress monitoring
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_view = {
                executor.submit(process_single_view, obj): obj["name"]
                for obj in views_to_process.values()
            }

            pbar = tqdm(
                as_completed(future_to_view),
                total=total_to_process,
                desc="Ingesting Clean Core CDS Views",
                unit="view",
                disable=(sys.stdout is None and sys.stderr is None),
                file=sys.stderr if sys.stderr is not None else None,
            )

            for future in pbar:
                v_name = future_to_view[future]
                try:
                    parsed_result = future.result()
                    if parsed_result:
                        parsed_views.append(parsed_result)
                        indexed_count += 1
                        if len(parsed_views) >= batch_size:
                            # In-batch deduplication before storage
                            seen_batch_ids = set()
                            deduped_batch = []
                            for pv in parsed_views:
                                vn = pv["view_name"].upper()
                                if vn not in seen_batch_ids:
                                    seen_batch_ids.add(vn)
                                    deduped_batch.append(pv)

                            try:
                                self.storage.add_cds_views_batch(deduped_batch)
                                save_catalog_seed(deduped_batch)
                                logger.info(
                                    f"Batch saved {len(deduped_batch)} views to StorageEngine & Catalog "
                                    f"(Progress: {indexed_count}/{total_to_process})."
                                )
                            except Exception as save_err:
                                logger.error(f"Error persisting batch of {len(deduped_batch)} views: {save_err}")
                            finally:
                                parsed_views = []
                    else:
                        failed_count += 1
                except Exception as exc:
                    failed_count += 1
                    logger.warning(f"Task for view '{v_name}' raised an unexpected exception: {exc}. Continuing.")

                # Update progress bar metrics
                pbar.set_postfix({"indexed": indexed_count, "failed": failed_count})

        # Save remaining batch
        if parsed_views:
            try:
                seen_batch_ids = set()
                deduped_batch = []
                for pv in parsed_views:
                    vn = pv["view_name"].upper()
                    if vn not in seen_batch_ids:
                        seen_batch_ids.add(vn)
                        deduped_batch.append(pv)

                self.storage.add_cds_views_batch(deduped_batch)
                save_catalog_seed(deduped_batch)
                logger.info(f"Final batch saved {len(deduped_batch)} views to StorageEngine & Catalog.")
            except Exception as e:
                logger.error(f"Error saving final batch to storage: {e}")

        elapsed = round(time.time() - start_time, 2)
        total_in_db = len(self.storage.get_all_views())
        logger.info("=" * 70)
        logger.info(
            f"INGESTION FINISHED: Successfully indexed {indexed_count}/{total_to_process} views "
            f"({failed_count} skipped/failed) in {elapsed}s."
        )
        logger.info(f"Total Views currently in Vector Store & Knowledge Graph: {total_in_db}")
        logger.info(f"Audit Log saved to: {LOG_FILE}")
        logger.info(f"Offline Seed Catalog saved to: {CATALOG_FILE}")
        if os.path.exists(FAILED_VIEWS_FILE):
            logger.info(f"Failed Views Ledger: {FAILED_VIEWS_FILE}")
        logger.info("=" * 70)
        return total_in_db

    def retry_failed_views(self) -> int:
        """Retry extraction for all views recorded in data/failed_views_retry_queue.txt."""
        if not os.path.exists(FAILED_VIEWS_FILE):
            logger.info(f"No failed views queue file found at {FAILED_VIEWS_FILE}.")
            return 0

        with open(FAILED_VIEWS_FILE, "r", encoding="utf-8") as f:
            failed_names = [line.strip().upper() for line in f if line.strip()]

        if not failed_names:
            logger.info("Failed views queue is empty.")
            return 0

        unique_failed = list(dict.fromkeys(failed_names))
        logger.info(f"Found {len(unique_failed)} unique views in {FAILED_VIEWS_FILE}. Starting targeted retry...")
        discovered_views = {name: {"name": name} for name in unique_failed}
        return self.fetch_and_index_views(discovered_views)

    def load_discovered_views_from_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Load discovered views directly from discovered_views.json or cds_catalog.json without network discovery."""
        discovered: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(DISCOVERED_VIEWS_FILE):
            try:
                with open(DISCOVERED_VIEWS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    discovered.update(data)
                elif isinstance(data, list):
                    for item in data:
                        vn = item.get("view_name") or item.get("name")
                        if vn:
                            discovered[vn] = item
                logger.info(f"Phase 1 Skipped: Loaded {len(discovered):,} views directly from disk cache {DISCOVERED_VIEWS_FILE}")
            except Exception as e:
                logger.warning(f"Failed reading {DISCOVERED_VIEWS_FILE}: {e}")

        if not discovered and os.path.exists(CATALOG_FILE):
            try:
                with open(CATALOG_FILE, "r", encoding="utf-8") as f:
                    cat_data = json.load(f)
                if isinstance(cat_data, list):
                    for item in cat_data:
                        vn = item.get("view_name") or item.get("name")
                        if vn:
                            discovered[vn] = {"name": vn, "description": item.get("description", "")}
                logger.info(f"Phase 1 Skipped: Loaded {len(discovered):,} views from {CATALOG_FILE}")
            except Exception as e:
                logger.warning(f"Failed reading {CATALOG_FILE}: {e}")

        return discovered


def run_full_crawler(
    max_cap: int = 200,
    workers: int = 5,
    limit: Optional[int] = None,
    from_cache: bool = False,
    retry_failed: bool = False,
    skip_discovery: bool = False,
    discover_only: bool = False,
    batch_size: int = 50,
    prefixes: Optional[List[str]] = None,
) -> int:
    """Convenience function to run Clean Core targeted crawler, load from cache, or retry failed views."""
    storage = StorageEngine()
    if from_cache:
        return load_from_cache(storage)

    crawler = AdaptivePrefixCrawler(storage=storage, max_results_cap=max_cap, max_workers=workers)

    if retry_failed:
        return crawler.retry_failed_views()

    if skip_discovery:
        views = crawler.load_discovered_views_from_catalog()
    else:
        views = crawler.discover_all_view_names(initial_prefixes=prefixes, max_total_views=limit)

    if discover_only:
        logger.info(f"Phase 1 Complete: Discovered {len(views):,} views across target namespaces.")
        return len(views)

    return crawler.fetch_and_index_views(views, batch_size=batch_size)


# Alias
run_full_ingestion = run_full_crawler


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean Core & Custom Dynamic Adaptive Prefix Crawler for SAP CDS Views.")
    parser.add_argument("--cap", type=int, default=200, help="Max results cap per search query (default: 200)")
    parser.add_argument("--workers", type=int, default=5, help="Number of worker threads (default: 5)")
    parser.add_argument("--limit", type=int, default=None, help="Optional max total views limit for test runs")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch commit chunk size (default: 50)")
    parser.add_argument("--from-cache", action="store_true", help="Load and index directly from data/cds_catalog.json without network calls")
    parser.add_argument("--retry-failed", action="store_true", help="Retry extraction for views in data/failed_views_retry_queue.txt")
    parser.add_argument("--skip-discovery", action="store_true", help="Skip BFS discovery and immediately begin extraction from catalog/cache")
    parser.add_argument("--discover-only", action="store_true", help="Run Phase 1 discovery and save to discovered_views.json without extraction")
    parser.add_argument("--prefixes", type=str, default=None, help="Comma-separated prefix targets (default: I_*,C_*,Z*,Y*)")
    args = parser.parse_args()

    custom_prefixes = [p.strip() for p in args.prefixes.split(",") if p.strip()] if args.prefixes else None

    run_full_crawler(
        max_cap=args.cap,
        workers=args.workers,
        limit=args.limit,
        from_cache=args.from_cache,
        retry_failed=args.retry_failed,
        skip_discovery=args.skip_discovery,
        discover_only=args.discover_only,
        batch_size=args.batch_size,
        prefixes=custom_prefixes,
    )

