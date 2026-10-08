"""Dual VectorDB (ChromaDB) + Knowledge Graph (Neo4j / NetworkX) Storage Engine.

Maintains CDS View signatures, semantic embeddings, and relational graph topology
including view entities, exposed fields, and foreign associations with join conditions.
"""

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional, Set
import chromadb
from chromadb.config import Settings as ChromaSettings
import networkx as nx

from config import settings

logger = logging.getLogger("database")
logging.basicConfig(level=logging.INFO)

# Optional Neo4j Driver
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


class VectorStore:
    """ChromaDB Vector Store for semantic CDS View signature matching."""

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="cds_view_signatures",
            metadata={"hnsw:space": "cosine"},
        )
        self._cached_names: Set[str] = set()
        self._cache_timestamp: float = 0.0
        logger.info(f"ChromaDB initialized at {self.persist_dir}. Collection count: {self.collection.count()}")

    def upsert_view(self, view_metadata: Dict[str, Any]) -> None:
        """Upsert a single CDS view signature and metadata."""
        view_name = view_metadata["view_name"]
        description = view_metadata.get("description", "")
        domain = view_metadata.get("domain", "General")
        view_type = view_metadata.get("view_type", "STANDARD")
        fields = [f["name"] for f in view_metadata.get("fields", [])]
        associations = [f"{a['alias']}->{a['target_view']}" for a in view_metadata.get("associations", [])]

        document = (
            f"View: {view_name} | Type: {view_type} | Domain: {domain} | Label: {description}\n"
            f"Fields: {', '.join(fields)}"
        )

        metadata = {
            "view_name": view_name,
            "view_type": view_type,
            "domain": domain,
            "field_count": len(fields),
            "description": description[:300],
            "raw_metadata": json.dumps(view_metadata),
        }

        self.collection.upsert(
            ids=[view_name],
            documents=[document],
            metadatas=[metadata],
        )

    def upsert_batch(self, views: List[Dict[str, Any]]) -> None:
        """Batch upsert CDS view signatures."""
        if not views:
            return

        # Deduplicate views by view_name within batch to satisfy ChromaDB unique ID constraint
        seen_ids = set()
        deduped_views = []
        for v in views:
            v_name = v["view_name"]
            if v_name not in seen_ids:
                seen_ids.add(v_name)
                deduped_views.append(v)

        ids: List[str] = []
        docs: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for v in deduped_views:
            v_name = v["view_name"]
            desc = v.get("description", "")
            dom = v.get("domain", "General")
            v_type = v.get("view_type", "STANDARD")
            f_names = [f["name"] for f in v.get("fields", [])]

            doc = (
                f"View: {v_name} | Type: {v_type} | Domain: {dom} | Label: {desc}\n"
                f"Fields: {', '.join(f_names)}"
            )

            meta = {
                "view_name": v_name,
                "view_type": v_type,
                "domain": dom,
                "field_count": len(f_names),
                "description": desc[:300],
                "raw_metadata": json.dumps(v),
            }

            ids.append(v_name)
            docs.append(doc)
            metadatas.append(meta)

        self.collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
        logger.info(f"Upserted {len(ids)} views to ChromaDB.")

    def search(
        self,
        query_text: str,
        limit: int = 5,
        scope: str = "ALL",
        view_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic vector search against CDS signatures with native metadata filtering.

        Supports scope: 'CUSTOM_ONLY', 'STANDARD_ONLY', 'ALL'.
        """
        if self.collection.count() == 0:
            return []

        where_filter = None
        # Support both scope and legacy view_type
        target_scope = (scope or view_type or "ALL").upper().strip()
        if target_scope in ("CUSTOM_ONLY", "CUSTOM"):
            where_filter = {"view_type": "CUSTOM"}
        elif target_scope in ("STANDARD_ONLY", "STANDARD"):
            where_filter = {"view_type": "STANDARD"}
        else:
            where_filter = None

        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(limit, self.collection.count()),
            where=where_filter,
        )

        candidates = []
        if results and results["ids"] and results["ids"][0]:
            for i, v_name in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i]
                dist = results["distances"][0][i] if "distances" in results and results["distances"] else 0.0
                raw_meta = json.loads(meta.get("raw_metadata", "{}"))
                candidates.append({
                    "view_name": v_name,
                    "view_type": meta.get("view_type", "STANDARD"),
                    "domain": meta.get("domain", "General"),
                    "description": meta.get("description", ""),
                    "field_count": meta.get("field_count", 0),
                    "distance": dist,
                    "similarity": round(1.0 - dist, 4) if dist <= 1.0 else round(1.0 / (1.0 + dist), 4),
                    "metadata": raw_meta,
                })

        return candidates

    def get_view_metadata(self, view_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored metadata for a view (case-insensitive lookup)."""
        ids_to_try = [view_name]
        if view_name.upper() != view_name:
            ids_to_try.append(view_name.upper())
        res = self.collection.get(ids=ids_to_try, include=["metadatas"])
        if res and res.get("metadatas") and len(res["metadatas"]) > 0:
            for m in res["metadatas"]:
                if m and m.get("raw_metadata"):
                    try:
                        return json.loads(m["raw_metadata"])
                    except Exception:
                        pass
        return None

    def get_all_views(self) -> List[Dict[str, Any]]:
        """Retrieve all indexed view summaries safely without hitting SQLite variable limit.

        Strategy 1: If data/cds_catalog.json is available on disk, load directly (<0.5s).
        Strategy 2: Safely chunked/paginated ChromaDB collection.get() in batches of 5,000.
        """
        # Strategy 1: Fast direct load from local offline catalog cache if present
        catalog_path = os.path.join(os.path.dirname(self.persist_dir), "cds_catalog.json")
        if os.path.exists(catalog_path):
            for attempt in range(3):
                try:
                    with open(catalog_path, "r", encoding="utf-8") as f:
                        views = json.load(f)
                    if isinstance(views, list) and len(views) > 0:
                        return views
                except Exception as e:
                    if attempt < 2:
                        time.sleep(0.15)
                    else:
                        logger.warning(f"Could not load direct from catalog cache {catalog_path}: {e}")

        # Strategy 2: Paginated collection.get() in chunks of 5000
        views = []
        try:
            total_count = self.collection.count()
            if total_count == 0:
                return []

            chunk_size = 5000
            for offset in range(0, total_count, chunk_size):
                res = self.collection.get(
                    include=["metadatas"],
                    limit=chunk_size,
                    offset=offset,
                )
                if res and res.get("metadatas"):
                    for m in res["metadatas"]:
                        raw = m.get("raw_metadata")
                        if raw:
                            try:
                                views.append(json.loads(raw))
                            except Exception:
                                pass
        except Exception as e:
            logger.error(f"Error fetching paginated views from ChromaDB: {e}")

        return views

    def get_all_indexed_view_names(self, force_refresh: bool = False) -> Set[str]:
        """Fast lookup of all indexed CDS View names in ChromaDB safely paginated with in-memory caching."""
        now = time.time()
        if self._cached_names and not force_refresh and (now - self._cache_timestamp < 120):
            return self._cached_names

        # Paginated ChromaDB get(include=[]) in chunks of 10,000
        indexed_names = set()
        try:
            total_count = self.collection.count()
            if total_count == 0:
                return set()

            chunk_size = 10000
            for offset in range(0, total_count, chunk_size):
                res = self.collection.get(
                    include=[],
                    limit=chunk_size,
                    offset=offset,
                )
                if res and "ids" in res and res["ids"]:
                    indexed_names.update(v.upper() for v in res["ids"])

            self._cached_names = indexed_names
            self._cache_timestamp = now
            return indexed_names
        except Exception as e:
            logger.warning(f"Error reading indexed view IDs: {e}")
            return self._cached_names or indexed_names


class KnowledgeGraphStore:
    """Knowledge Graph store supporting Neo4j with automatic NetworkX memory fallback."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.uri = uri or settings.NEO4J_URI
        self.user = user or settings.NEO4J_USER
        self.password = password or settings.NEO4J_PASSWORD
        self.neo4j_driver = None
        self.use_neo4j = False

        # In-memory graph (NetworkX) always active as fallback / fast cache
        self.local_graph = nx.MultiDiGraph()

        if NEO4J_AVAILABLE:
            try:
                self.neo4j_driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password),
                    connection_timeout=2.0,
                )
                self.neo4j_driver.verify_connectivity()
                self.use_neo4j = True
                self._init_neo4j_schema()
                logger.info(f"Connected to Neo4j at {self.uri}")
            except Exception as e:
                logger.warning(f"Neo4j connection to {self.uri} failed: {e}. Running with local Knowledge Graph.")
                self.use_neo4j = False
        else:
            logger.info("Neo4j driver not installed. Running with local Knowledge Graph.")

    def _init_neo4j_schema(self) -> None:
        """Create indices and constraints on Neo4j."""
        if not self.use_neo4j or not self.neo4j_driver:
            return
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (v:CDSView) REQUIRE v.name IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Field) REQUIRE f.name IS UNIQUE;",
            "CREATE INDEX IF NOT EXISTS FOR (v:CDSView) ON (v.domain);",
        ]
        with self.neo4j_driver.session() as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    logger.debug(f"Schema query info: {e}")

    def upsert_view(self, view: Dict[str, Any]) -> None:
        """Insert CDS view, fields, and associations into Knowledge Graph."""
        v_name = view["view_name"].upper()
        desc = view.get("description", "")
        v_type = view.get("view_type", "STANDARD")
        domain = view.get("domain", "General")
        fields = view.get("fields", [])
        assocs = view.get("associations", [])

        # 1. Update Local NetworkX Graph
        self.local_graph.add_node(
            v_name,
            node_type="CDSView",
            name=v_name,
            description=desc,
            view_type=v_type,
            domain=domain,
            raw_data=view,
        )

        for f in fields:
            f_name = f["name"].upper()
            self.local_graph.add_node(f_name, node_type="Field", name=f_name)
            self.local_graph.add_edge(v_name, f_name, relation="EXPOSES", is_key=f.get("is_key", False))

        for a in assocs:
            target = a["target_view"].upper()
            self.local_graph.add_node(target, node_type="CDSView", name=target)
            self.local_graph.add_edge(
                v_name,
                target,
                relation="ASSOCIATES_TO",
                alias=a.get("alias", ""),
                cardinality=a.get("cardinality", "[0..1]"),
                on_condition=a.get("on_condition", ""),
            )

        # 2. Update Neo4j Graph if connected
        if self.use_neo4j and self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    session.execute_write(self._neo4j_upsert_tx, view)
            except Exception as e:
                logger.warning(f"Neo4j write error: {e}")

    @staticmethod
    def _neo4j_upsert_tx(tx, view: Dict[str, Any]) -> None:
        v_name = view["view_name"].upper()
        desc = view.get("description", "")
        v_type = view.get("view_type", "STANDARD")
        domain = view.get("domain", "General")
        fields = view.get("fields", [])
        assocs = view.get("associations", [])

        # Merge CDS View
        tx.run(
            """
            MERGE (v:CDSView {name: $name})
            SET v.description = $desc, v.view_type = $v_type, v.domain = $domain
            """,
            name=v_name,
            desc=desc,
            v_type=v_type,
            domain=domain,
        )

        # Merge Fields and Relationships
        for f in fields:
            f_name = f["name"].upper()
            tx.run(
                """
                MERGE (f:Field {name: $fname})
                MERGE (v:CDSView {name: $vname})
                MERGE (v)-[r:EXPOSES]->(f)
                SET r.is_key = $is_key
                """,
                fname=f_name,
                vname=v_name,
                is_key=f.get("is_key", False),
            )

        # Merge Associations
        for a in assocs:
            target = a["target_view"].upper()
            tx.run(
                """
                MERGE (v:CDSView {name: $vname})
                MERGE (t:CDSView {name: $target})
                MERGE (v)-[r:ASSOCIATES_TO {alias: $alias}]->(t)
                SET r.cardinality = $cardinality, r.on_condition = $on_condition
                """,
                vname=v_name,
                target=target,
                alias=a.get("alias", ""),
                cardinality=a.get("cardinality", "[0..1]"),
                on_condition=a.get("on_condition", ""),
            )

    def find_associated_views_for_missing_fields(
        self,
        top_candidate: str,
        missing_fields: List[str],
    ) -> List[Dict[str, Any]]:
        """Find associated CDS views within 1-2 hops that cover missing fields.

        Returns list of associated views with their coverage, join paths, and ON conditions.
        """
        top_candidate = top_candidate.upper()
        missing_set = set(f.upper() for f in missing_fields)

        # Strategy 1: Try Neo4j Cypher query if connected
        if self.use_neo4j and self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    result = session.run(
                        """
                        MATCH (v:CDSView {name: $top_candidate})-[:ASSOCIATES_TO*1..2]-(connected:CDSView)-[:EXPOSES]->(f:Field)
                        WHERE f.name IN $missing_fields AND connected.name <> $top_candidate
                        RETURN connected.name AS connected_view,
                               collect(DISTINCT f.name) AS covered_missing,
                               connected.description AS description
                        """,
                        top_candidate=top_candidate,
                        missing_fields=list(missing_set),
                    )
                    neo_records = []
                    for record in result:
                        neo_records.append({
                            "view_name": record["connected_view"],
                            "covered_fields": record["covered_missing"],
                            "description": record.get("description", ""),
                        })
                    if neo_records:
                        return self._enrich_join_conditions(top_candidate, neo_records)
            except Exception as e:
                logger.warning(f"Neo4j Cypher query failed: {e}. Falling back to graph traversal.")

        # Strategy 2: Local NetworkX Graph Traversal
        traversal_results = []
        if top_candidate not in self.local_graph:
            return traversal_results

        # Explore 1 and 2 hops from top_candidate
        candidates_to_check: Set[str] = set()

        # 1-hop forward and backward
        for neighbor in self.local_graph.neighbors(top_candidate):
            if self.local_graph.nodes[neighbor].get("node_type") == "CDSView":
                candidates_to_check.add(neighbor)
        for predecessor in self.local_graph.predecessors(top_candidate):
            if self.local_graph.nodes[predecessor].get("node_type") == "CDSView":
                candidates_to_check.add(predecessor)

        # 2-hops
        one_hop_views = list(candidates_to_check)
        for h1 in one_hop_views:
            for h2 in self.local_graph.neighbors(h1):
                if self.local_graph.nodes[h2].get("node_type") == "CDSView" and h2 != top_candidate:
                    candidates_to_check.add(h2)
            for h2 in self.local_graph.predecessors(h1):
                if self.local_graph.nodes[h2].get("node_type") == "CDSView" and h2 != top_candidate:
                    candidates_to_check.add(h2)

        for cand in candidates_to_check:
            # Check what fields cand exposes
            cand_fields = set()
            for neighbor in self.local_graph.neighbors(cand):
                if self.local_graph.nodes[neighbor].get("node_type") == "Field":
                    cand_fields.add(neighbor.upper())

            covered = list(missing_set.intersection(cand_fields))
            if covered:
                node_data = self.local_graph.nodes[cand]
                traversal_results.append({
                    "view_name": cand,
                    "covered_fields": covered,
                    "description": node_data.get("description", ""),
                })

        return self._enrich_join_conditions(top_candidate, traversal_results)

    def _enrich_join_conditions(
        self,
        primary_view: str,
        supporting_views: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Add exact join conditions and aliases between primary and supporting views."""
        enriched = []
        for item in supporting_views:
            target = item["view_name"]
            join_cond = None
            alias = None
            cardinality = "[0..1]"

            # Check direct edge in graph
            if self.local_graph.has_edge(primary_view, target):
                edge_dict = self.local_graph.get_edge_data(primary_view, target)
                for _, edge_data in edge_dict.items():
                    if edge_data.get("relation") == "ASSOCIATES_TO":
                        join_cond = edge_data.get("on_condition")
                        alias = edge_data.get("alias")
                        cardinality = edge_data.get("cardinality")
                        break

            # Reverse edge check
            if not join_cond and self.local_graph.has_edge(target, primary_view):
                edge_dict = self.local_graph.get_edge_data(target, primary_view)
                for _, edge_data in edge_dict.items():
                    if edge_data.get("relation") == "ASSOCIATES_TO":
                        join_cond = edge_data.get("on_condition")
                        alias = f"_{target}"
                        cardinality = edge_data.get("cardinality")
                        break

            enriched.append({
                "view_name": target,
                "covered_fields": item["covered_fields"],
                "description": item.get("description", ""),
                "alias": alias or f"_{target.replace('I_', '').replace('C_', '')}",
                "cardinality": cardinality,
                "on_condition": join_cond or f"$projection.{item['covered_fields'][0]} = _{target.replace('I_', '').replace('C_', '')}.{item['covered_fields'][0]}",
            })

        return enriched

    def get_view_subgraph(self, view_name: str, depth: int = 2) -> Dict[str, Any]:
        """Extract nodes and edges for visualization in UI."""
        view_name = view_name.upper()
        nodes = []
        edges = []
        visited_nodes = set()

        if view_name not in self.local_graph:
            return {"nodes": nodes, "edges": edges}

        subgraph_nodes = {view_name}
        current_layer = {view_name}

        for _ in range(depth):
            next_layer = set()
            for n in current_layer:
                for nbr in self.local_graph.neighbors(n):
                    next_layer.add(nbr)
                for pred in self.local_graph.predecessors(n):
                    next_layer.add(pred)
            subgraph_nodes.update(next_layer)
            current_layer = next_layer

        for n in subgraph_nodes:
            ndata = self.local_graph.nodes[n]
            ntype = ndata.get("node_type", "Unknown")
            nodes.append({
                "id": n,
                "label": n,
                "type": ntype,
                "domain": ndata.get("domain", ""),
                "view_type": ndata.get("view_type", ""),
            })

        for u, v, key, edata in self.local_graph.edges(keys=True, data=True):
            if u in subgraph_nodes and v in subgraph_nodes:
                edges.append({
                    "from": u,
                    "to": v,
                    "relation": edata.get("relation", ""),
                    "alias": edata.get("alias", ""),
                    "on_condition": edata.get("on_condition", ""),
                })

        return {"nodes": nodes, "edges": edges}


class StorageEngine:
    """Unified dual storage engine for VectorDB and Knowledge Graph."""

    def __init__(self):
        self.vector_store = VectorStore()
        self.graph_store = KnowledgeGraphStore()

    def add_cds_view(self, view_metadata: Dict[str, Any]) -> None:
        """Store CDS view in both VectorDB and Knowledge Graph."""
        self.vector_store.upsert_view(view_metadata)
        self.graph_store.upsert_view(view_metadata)

    def add_cds_views_batch(self, views: List[Dict[str, Any]]) -> None:
        """Batch store CDS views."""
        self.vector_store.upsert_batch(views)
        for v in views:
            self.graph_store.upsert_view(v)

    def search_vector(
        self,
        query: str,
        limit: int = 5,
        scope: str = "ALL",
        view_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self.vector_store.search(query, limit=limit, scope=scope, view_type=view_type)

    def get_view_metadata(self, view_name: str) -> Optional[Dict[str, Any]]:
        return self.vector_store.get_view_metadata(view_name)

    def get_all_views(self) -> List[Dict[str, Any]]:
        return self.vector_store.get_all_views()

    def get_all_indexed_view_names(self, force_refresh: bool = False) -> Set[str]:
        return self.vector_store.get_all_indexed_view_names(force_refresh=force_refresh)

    def get_total_indexed_count(self) -> int:
        """Fast O(1) total view count from ChromaDB collection."""
        try:
            return self.vector_store.collection.count()
        except Exception:
            return len(self.get_all_indexed_view_names())
