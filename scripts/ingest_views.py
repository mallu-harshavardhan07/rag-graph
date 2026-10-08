"""Compatibility wrapper delegating to dynamic adaptive prefix crawler."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.ingest_all_views import run_full_crawler, AdaptivePrefixCrawler


def run_ingestion(max_sap_views: int = 200, search_queries=None) -> int:
    """Run dynamic adaptive crawler without hardcoded view arrays."""
    return run_full_crawler(max_cap=max_sap_views, workers=5)


if __name__ == "__main__":
    run_ingestion()
