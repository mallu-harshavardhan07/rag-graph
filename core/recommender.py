"""Reusability Evaluator and Deterministic Graph Join-Tree Resolver.

Determines whether an existing SAP CDS View completely satisfies requirements (100% Match -> REUSE),
or performs Knowledge Graph traversal to construct a deterministic join-tree plan
(GENERATE_COMPOSITE) with strict Zero-Hallucination guarantees.
Includes Pre-Retrieval LLM Query Expansion for dense semantic retrieval.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field

from core.database import StorageEngine, KnowledgeGraphStore
from core.query_expander import QueryExpander, QueryExpansion
from core.canonical_metadata import get_canonical_registry, CanonicalMetadataRegistry

logger = logging.getLogger("recommender")
logging.basicConfig(level=logging.INFO)


def _clean_field_name(raw_name: str) -> str:
    """Extract clean identifier from annotated CDS field names."""
    m = re.search(r'([A-Za-z0-9_]+)\s*$', str(raw_name).strip())
    return m.group(1) if m else str(raw_name).strip()


def _match_field_in_dict(field_query: str, fields_dict: Dict[str, str]) -> Optional[str]:
    """Match requested field against view fields with case-insensitivity and SAP synonyms."""
    fq_upper = field_query.upper().strip()
    # 1. Exact match
    if fq_upper in fields_dict:
        return fields_dict[fq_upper]

    # 2. Underscore-stripped match (e.g. TAX_AMOUNT -> TAXAMOUNT)
    fq_stripped = fq_upper.replace("_", "")
    for k_upper, orig_name in fields_dict.items():
        if fq_stripped == k_upper.replace("_", ""):
            return orig_name

    # 3. Canonical SAP Synonyms
    SYNONYM_MAP = {
        "CUSTOMER": ["CUSTOMER", "SOLDTOPARTY", "PAYERPARTY", "BILLTOPARTY", "CUSTOMERPRICEGROUP"],
        "TAXAMOUNT": ["TAXAMOUNT", "TOTALTAXAMOUNT", "TAXBASEAMOUNTINTRANSCRCURRENCY", "TAXCODE", "TAXJURISDICTION"],
        "TAX": ["TAXAMOUNT", "TOTALTAXAMOUNT", "TAXCODE", "TAXJURISDICTION"],
        "SUPPLIER": ["SUPPLIER", "VENDOR", "CREDITOR"],
        "VENDOR": ["SUPPLIER", "VENDOR", "CREDITOR"],
        "TOTALNETAMOUNT": ["TOTALNETAMOUNT", "NETAMOUNT", "NETPRICEAMOUNT"],
        "NETAMOUNT": ["TOTALNETAMOUNT", "NETAMOUNT", "NETPRICEAMOUNT"],
        "BILLINGDOCUMENT": ["BILLINGDOCUMENT", "ACCOUNTINGDOCUMENT", "INVOICE"],
        "COMPANYCODE": ["COMPANYCODE"],
        "FISCALYEAR": ["FISCALYEAR"],
        "PRODUCT": ["PRODUCT", "MATERIAL"],
        "MATERIAL": ["PRODUCT", "MATERIAL"],
    }
    candidates = SYNONYM_MAP.get(fq_stripped, [])
    # 3a. Exact match against synonym candidates
    for cand in candidates:
        if cand in fields_dict:
            return fields_dict[cand]
        for k_upper, orig_name in fields_dict.items():
            if cand == k_upper.replace("_", ""):
                return orig_name

    # 3b. Prefix/partial match against synonym candidates (fallback)
    for cand in candidates:
        for k_upper, orig_name in fields_dict.items():
            if k_upper.replace("_", "").startswith(cand):
                return orig_name

    return None


def resolve_field_join_plan(
    primary_view: str,
    requested_fields: List[str],
    graph: Any,
) -> Dict[str, Any]:
    """Deterministically resolve requested fields to primary view or verified graph associations.

    CRITICAL RULE: Stop allowing the generator to fallback to `_Primary` for missing fields.
    If a field cannot be resolved via the Knowledge Graph, it is added to `unresolved_fields`.

    Returns strict schema:
    {
        "primary_view": str,
        "field_mapping": dict,
        "required_associations": list,
        "unresolved_fields": list
    }
    """
    primary_view_upper = primary_view.strip().upper()
    req_fields_clean = [f.strip() for f in requested_fields if f.strip()]

    # Extract local NetworkX graph or KnowledgeGraphStore
    if isinstance(graph, StorageEngine):
        graph_store = graph.graph_store
        storage = graph
    elif isinstance(graph, KnowledgeGraphStore):
        graph_store = graph
        storage = getattr(graph, "storage", None)
    else:
        graph_store = getattr(graph, "graph_store", graph)
        storage = getattr(graph, "storage", None)

    nx_graph = getattr(graph_store, "local_graph", None)

    # Dynamic on-demand load if primary_view is not yet populated in graph
    if storage and nx_graph is not None and primary_view_upper not in nx_graph:
        prim_meta = storage.get_view_metadata(primary_view_upper)
        if prim_meta:
            graph_store.upsert_view(prim_meta)
            # Also populate 1-hop associations so join-tree is available
            for a in prim_meta.get("associations", []):
                t_name = a.get("target_view")
                if t_name:
                    t_meta = storage.get_view_metadata(t_name)
                    if t_meta:
                        graph_store.upsert_view(t_meta)

    # 1. Collect all fields exposed by the primary view
    primary_fields_dict: Dict[str, str] = {}
    if nx_graph and primary_view_upper in nx_graph:
        node_raw = nx_graph.nodes[primary_view_upper].get("raw_data", {})
        for f in node_raw.get("fields", []):
            raw_f = f.get("name", "")
            clean_f = _clean_field_name(raw_f)
            primary_fields_dict[clean_f.upper()] = clean_f
            primary_fields_dict[raw_f.upper()] = clean_f

        # Also check EXPOSES edges
        for nbr in nx_graph.neighbors(primary_view_upper):
            if nx_graph.nodes[nbr].get("node_type") == "Field":
                clean_f = _clean_field_name(nbr)
                primary_fields_dict[clean_f.upper()] = clean_f
                primary_fields_dict[nbr.upper()] = clean_f

    field_mapping: Dict[str, Dict[str, Any]] = {}
    required_associations_map: Dict[str, Dict[str, Any]] = {}
    unresolved_fields: List[str] = []

    # 2. Iterate requested fields
    for req_field in req_fields_clean:
        # Check if field exists in primary view (exact, stripped, or synonym)
        matched_primary = _match_field_in_dict(req_field, primary_fields_dict)
        if matched_primary:
            field_mapping[req_field] = {
                "source_view": primary_view_upper,
                "alias": "_Primary",
                "view_field": matched_primary,
                "field_expr": f"_Primary.{matched_primary} as {req_field}",
            }
            continue

        # 3. Field missing in primary -> Traverse Knowledge Graph for exact associations
        resolved_in_graph = False

        if nx_graph and primary_view_upper in nx_graph:
            # Check 1-hop associated views
            candidates_to_check: List[str] = []
            for nbr in nx_graph.neighbors(primary_view_upper):
                if nx_graph.nodes[nbr].get("node_type") == "CDSView":
                    candidates_to_check.append(nbr)
            for pred in nx_graph.predecessors(primary_view_upper):
                if nx_graph.nodes[pred].get("node_type") == "CDSView" and pred not in candidates_to_check:
                    candidates_to_check.append(pred)

            # Check each candidate view
            for cand_view in candidates_to_check:
                # Ensure candidate view metadata is loaded into graph
                if storage and nx_graph.nodes[cand_view].get("node_type") == "CDSView" and not nx_graph.nodes[cand_view].get("raw_data"):
                    cand_meta = storage.get_view_metadata(cand_view)
                    if cand_meta:
                        graph_store.upsert_view(cand_meta)

                cand_fields_dict: Dict[str, str] = {}
                cand_raw = nx_graph.nodes[cand_view].get("raw_data", {})
                for f in cand_raw.get("fields", []):
                    raw_f = f.get("name", "")
                    clean_f = _clean_field_name(raw_f)
                    cand_fields_dict[clean_f.upper()] = clean_f
                    cand_fields_dict[raw_f.upper()] = clean_f

                for nbr in nx_graph.neighbors(cand_view):
                    if nx_graph.nodes[nbr].get("node_type") == "Field":
                        clean_f = _clean_field_name(nbr)
                        cand_fields_dict[clean_f.upper()] = clean_f
                        cand_fields_dict[nbr.upper()] = clean_f

                # Match req_field against candidate view
                matched_target_field = _match_field_in_dict(req_field, cand_fields_dict)

                if matched_target_field:
                    alias = None
                    cardinality = "[0..1]"
                    on_condition = None

                    if nx_graph.has_edge(primary_view_upper, cand_view):
                        edge_dict = nx_graph.get_edge_data(primary_view_upper, cand_view)
                        for _, edata in edge_dict.items():
                            if edata.get("relation") == "ASSOCIATES_TO":
                                alias = edata.get("alias")
                                cardinality = edata.get("cardinality", "[0..1]")
                                on_condition = edata.get("on_condition")
                                break

                    if not alias and nx_graph.has_edge(cand_view, primary_view_upper):
                        edge_dict = nx_graph.get_edge_data(cand_view, primary_view_upper)
                        for _, edata in edge_dict.items():
                            if edata.get("relation") == "ASSOCIATES_TO":
                                alias = f"_{cand_view.replace('I_', '').replace('C_', '')}"
                                cardinality = edata.get("cardinality", "[0..1]")
                                on_condition = edata.get("on_condition")
                                break

                    clean_alias = alias or f"_{cand_view.replace('I_', '').replace('C_', '')}"
                    cond = on_condition or f"$projection.{matched_target_field} = {clean_alias}.{matched_target_field}"

                    # Record association
                    if cand_view not in required_associations_map:
                        required_associations_map[cand_view] = {
                            "target_view": cand_view,
                            "alias": clean_alias,
                            "cardinality": cardinality,
                            "on_condition": cond,
                            "covered_fields": [matched_target_field],
                        }
                    else:
                        if matched_target_field not in required_associations_map[cand_view]["covered_fields"]:
                            required_associations_map[cand_view]["covered_fields"].append(matched_target_field)

                    field_mapping[req_field] = {
                        "source_view": cand_view,
                        "alias": clean_alias,
                        "view_field": matched_target_field,
                        "field_expr": f"{clean_alias}.{matched_target_field} as {req_field}",
                    }
                    resolved_in_graph = True
                    break

        # 4. CRITICAL: If field could not be resolved, add to unresolved_fields (DO NOT MAP TO _Primary)
        if not resolved_in_graph:
            unresolved_fields.append(req_field)

    return {
        "primary_view": primary_view_upper,
        "field_mapping": field_mapping,
        "required_associations": list(required_associations_map.values()),
        "unresolved_fields": unresolved_fields,
    }


from core.composition_solver import (
    RequirementAnalyzer,
    RequirementSolver,
    CandidateView,
    DecisionResult,
    RequirementAnalysis,
    detect_analytical_intent,
    classify_cardinality,
    evaluate_view_pair,
)

# Backwards-compatible alias for DecisionResult
RecommendationResult = DecisionResult


def classify_vdm_type(cand: CandidateView) -> str:
    """Classify SAP Virtual Data Model (VDM) hierarchy type.
    
    Checks CDS metadata first (@VDM.viewType, @Consumption.valueHelpDefinition);
    falls back to SAP naming conventions:
    - VALUE_HELP: Metadata #VALUE_HELP, or name ends with 'VH' / contains 'VALUEHELP'
    - CONSUMPTION: Metadata #CONSUMPTION, or name starts with 'C_'
    - INTERFACE: Metadata #BASIC, #COMPOSITE, #TRANSACTIONAL, or name starts with 'I_'
    - CUSTOM: Starts with 'Z' or 'Y'
    - UNKNOWN: Other views
    """
    name = cand.name.upper().strip()
    raw_meta = getattr(cand, "raw_metadata", {}) or {}
    raw_ddl = raw_meta.get("raw_ddl", "")

    # 1. Metadata First: check explicit metadata dict
    vdm_meta = str(raw_meta.get("vdm_type") or raw_meta.get("vdm_view_type") or "").upper().strip()
    if vdm_meta:
        if any(vh in vdm_meta for vh in ["VALUE_HELP", "VALUEHELP", "VH"]):
            return "VALUE_HELP"
        elif "CONSUMPTION" in vdm_meta:
            return "CONSUMPTION"
        elif any(core in vdm_meta for core in ["BASIC", "COMPOSITE", "INTERFACE", "CORE", "TRANSACTIONAL"]):
            return "INTERFACE"

    # Check annotations in raw_ddl if present
    if raw_ddl:
        vdm_match = re.search(r"@VDM\.viewType\s*:\s*#?([A-Za-z0-9_]+)", raw_ddl, re.IGNORECASE)
        if vdm_match:
            vtype = vdm_match.group(1).upper()
            if vtype in ("VALUE_HELP", "VALUEHELP", "VH", "HELP"):
                return "VALUE_HELP"
            elif vtype in ("CONSUMPTION",):
                return "CONSUMPTION"
            elif vtype in ("BASIC", "COMPOSITE", "INTERFACE", "CORE", "TRANSACTIONAL"):
                return "INTERFACE"
        if re.search(r"@Consumption\.valueHelpDefinition", raw_ddl, re.IGNORECASE):
            return "VALUE_HELP"

    # 2. Naming Conventions Fallback
    # Value Help checks: ends with VH (e.g. _VH, VH) or contains VALUEHELP
    if name.endswith("_VH") or name.endswith("VH") or "VALUEHELP" in name or "_VH_" in name:
        return "VALUE_HELP"
    # Consumption check: starts with C_
    if name.startswith("C_") or name.startswith("C/"):
        return "CONSUMPTION"
    # Interface / Core check: starts with I_
    if name.startswith("I_") or name.startswith("I/"):
        return "INTERFACE"
    # Custom views
    if name.startswith(("Z", "Y")):
        return "CUSTOM"

    return "UNKNOWN"


def find_unique_source_views(
    candidates: List[CandidateView],
    required_fields: Optional[List[str]],
) -> Set[str]:
    """Identify candidate views that are the SOLE provider of at least one required field in the pool."""
    if not candidates or not required_fields:
        return set()

    req_set = {f.upper().strip() for f in required_fields if f and f.strip()}
    if not req_set:
        return set()

    field_to_cands: Dict[str, List[str]] = {rf: [] for rf in req_set}
    for cand in candidates:
        for rf in req_set:
            if rf in cand.fields:
                field_to_cands[rf].append(cand.name.upper())

    unique_views: Set[str] = set()
    for rf, providers in field_to_cands.items():
        if len(providers) == 1:
            unique_views.add(providers[0])

    return unique_views


def rerank_candidates(
    candidates: List[CandidateView],
    intent: Optional[str] = None,
    required_fields: Optional[List[str]] = None,
    apply_vdm: bool = True,
) -> List[CandidateView]:
    """Apply SAP VDM hierarchy reranking, penalization, and intent soft boosting.

    1. Reranking Rules (Metadata First, Naming Fallback):
       - Value Help Views (#VALUE_HELP, *_VH, VALUEHELP): -0.5 strong penalty
       - Consumption Views (#CONSUMPTION, C_*): -0.2 moderate penalty
       - Interface/Core Views (#BASIC, #COMPOSITE, I_*): +0.1 positive boost
    2. Intent Soft Boosting:
       - ANALYTICAL: +0.3 to #CUBE / #QUERY
       - TRANSACTIONAL: +0.2 to #FACT / line items
       - MASTER_DATA: +0.2 to #DIMENSION / #TEXT
    3. Unique Field Protection:
       - If a candidate is the ONLY retrieved provider of a requested required field,
         its VDM penalty is waived and a protection boost is applied to ensure it is retained at the top of the pool.
    4. Auditability:
       - Preserves base_score and records adjusted_score, vdm_type, and vdm_adjustment in candidate metadata.
    """
    if not candidates:
        return []

    # 1. Identify Unique Source candidates for required fields
    unique_source_views = find_unique_source_views(candidates, required_fields)

    # 2. Compute base and adjusted scores for each candidate
    for cand in candidates:
        # Preserve original base score
        if not hasattr(cand, "base_score") or cand.base_score == 0.0:
            cand.base_score = round(float(cand.retrieval_score), 4)

        base = cand.base_score

        # A. VDM Hierarchy Adjustment
        vdm_type = classify_vdm_type(cand)
        vdm_adjustment = 0.0

        if apply_vdm:
            if vdm_type == "VALUE_HELP":
                vdm_adjustment = -0.50
            elif vdm_type == "CONSUMPTION":
                vdm_adjustment = -0.20
            elif vdm_type == "INTERFACE":
                vdm_adjustment = 0.10

        # B. Intent Soft Boosting
        intent_boost = 0.0
        cat = getattr(cand, "data_category", "#UNKNOWN").upper()
        if intent == "ANALYTICAL":
            if cat in ("#CUBE", "#QUERY", "#AGGREGATION") or cand.is_analytical_cube:
                intent_boost = 0.30
        elif intent == "TRANSACTIONAL":
            if cat in ("#FACT", "#TRANSACTIONAL") or any(k in cand.name.upper() for k in ["ITEM", "ORDER", "DOCUMENT", "INVOICE", "DELIVERY", "POSTING"]):
                intent_boost = 0.20
        elif intent == "MASTER_DATA":
            if cat in ("#DIMENSION", "#TEXT") or any(k in cand.name.upper() for k in ["CUSTOMER", "SUPPLIER", "VENDOR", "MATERIAL", "PRODUCT", "PLANT", "BANK"]):
                intent_boost = 0.20

        # C. Unique Field Protection
        is_unique_source = cand.name.upper() in unique_source_views
        unique_boost = 0.0
        if is_unique_source:
            # Waive VDM penalty if negative and elevate to retain at top of pool
            if vdm_adjustment < 0:
                vdm_adjustment = 0.0
            unique_boost = 1.0  # Protection boost guarantees retention at top of pool

        # Final Adjusted Score
        adjusted = round(base + intent_boost + vdm_adjustment + unique_boost, 4)

        cand.adjusted_score = adjusted
        cand.retrieval_score = adjusted
        cand.vdm_type = vdm_type

        # Ensure candidate raw_metadata is updated for full auditability
        if not isinstance(cand.raw_metadata, dict):
            cand.raw_metadata = {}
        cand.raw_metadata["base_score"] = base
        cand.raw_metadata["adjusted_score"] = adjusted
        cand.raw_metadata["vdm_type"] = vdm_type
        cand.raw_metadata["vdm_adjustment"] = vdm_adjustment
        cand.raw_metadata["is_unique_source"] = is_unique_source

    return sorted(candidates, key=lambda c: c.retrieval_score, reverse=True)


class ReusabilityEvaluator:
    """Enterprise Tri-State Decision Engine (REUSE, EXTEND, CREATE) with GraphRAG Safety Gates.
    
    Implements strictly:
    Layer 1: Requirement Analyzer & Semantic Router (Intent, Dimensions, Measures, Aggregations)
    Layer 2: Retrieval Layer (ChromaDB Top 30 Vector Search + 1-Hop Association Expansion)
    Layer 3: Metadata-Aware Reranker (Soft Boosting based on SAP Intent and @Analytics.dataCategory)
    Layer 4: Coverage Layer (Pure Python Set Math on Dimensions & Measures)
    Layer 5: Composition Solver & Graph Safety (Hard Cardinality Ban: 1..1, 0..1 only)
    Layer 6: Decision (Standardized JSON API Contract & Evidence Confidence)
    """

    def __init__(
        self,
        storage_engine: Optional[StorageEngine] = None,
        registry: Optional[CanonicalMetadataRegistry] = None,
    ):
        self.storage = storage_engine or StorageEngine()
        self.registry = registry or get_canonical_registry()
        self.query_expander = QueryExpander()
        self.analyzer = RequirementAnalyzer(query_expander=self.query_expander, registry=self.registry)
        self.solver = RequirementSolver()

    def retrieve_secondary_candidates(
        self,
        primary: CandidateView,
        missing_fields: List[str],
        scope: str = "ALL",
        limit: int = 20,
    ) -> List[CandidateView]:
        """Targeted secondary retrieval for EXTEND state composition.
        
        1. Queries deterministic Field -> View Reverse Index (Sections 4, 12, 13)
        2. Queries direct 1-hop associations of primary view
        3. Queries ChromaDB vector store for semantic similarity
        4. Ranks secondary candidates by missing-field coverage and retrieval score.
        """
        missing_set = {f.upper() for f in missing_fields if f and f.strip()}
        if not missing_set:
            return []

        clean_missing = [f.strip() for f in missing_fields if f and f.strip()]
        missing_query = " ".join(clean_missing)

        secondary_views: List[CandidateView] = []
        seen_names: Set[str] = {primary.name.upper()}

        def _is_scope_match(view_name: str, view_type: str) -> bool:
            v_type = (view_type or "").upper()
            v_name = view_name.upper()
            is_custom = v_type == "CUSTOM" or v_name.startswith(("Z", "Y"))
            scope_norm = (scope or "ALL").upper().strip()
            if scope_norm in ("CUSTOM_ONLY", "CUSTOM"):
                return is_custom
            elif scope_norm in ("STANDARD_ONLY", "STANDARD"):
                return not is_custom
            return True

        # 0. Field -> View Reverse Index (Primary Deterministic Discovery - Sections 4, 12, 13)
        for mf in clean_missing:
            providers = self.registry.get_views_for_field(mf)
            for p_name in providers:
                p_upper = p_name.upper()
                if p_upper not in seen_names:
                    p_meta = self.storage.get_view_metadata(p_upper)
                    if not p_meta:
                        reg_v = self.registry.get_view(p_upper)
                        p_meta = reg_v.to_dict() if reg_v else None
                    if p_meta and _is_scope_match(p_upper, p_meta.get("view_type", "STANDARD")):
                        seen_names.add(p_upper)
                        cand = CandidateView.from_metadata(p_meta, retrieval_score=0.94)
                        secondary_views.append(cand)

        # 1. Targeted 1-Hop Graph Associations of Primary View
        # Check direct associations declared on primary
        for a in primary.associations:
            t_name = a.get("target_view", "").upper()
            if t_name and t_name not in seen_names:
                t_meta = self.storage.get_view_metadata(t_name)
                if t_meta and _is_scope_match(t_name, t_meta.get("view_type", "STANDARD")):
                    seen_names.add(t_name)
                    cand = CandidateView.from_metadata(t_meta, retrieval_score=0.95)
                    secondary_views.append(cand)

        # Check graph_store for associations covering missing fields
        if hasattr(self.storage, "graph_store"):
            try:
                assoc_results = self.storage.graph_store.find_associated_views_for_missing_fields(
                    top_candidate=primary.name,
                    missing_fields=clean_missing,
                )
                for item in assoc_results:
                    t_name = item.get("view_name", "").upper()
                    if t_name and t_name not in seen_names:
                        t_meta = self.storage.get_view_metadata(t_name)
                        if t_meta and _is_scope_match(t_name, t_meta.get("view_type", "STANDARD")):
                            seen_names.add(t_name)
                            cand = CandidateView.from_metadata(t_meta, retrieval_score=0.92)
                            secondary_views.append(cand)
            except Exception as e:
                logger.debug(f"Error querying graph associations for missing fields: {e}")

            # Check reverse 1-hop edges (predecessors) in local_graph
            nx_graph = getattr(self.storage.graph_store, "local_graph", None)
            if nx_graph and primary.name in nx_graph:
                for pred in nx_graph.predecessors(primary.name):
                    pred_upper = pred.upper()
                    if pred_upper not in seen_names and nx_graph.nodes[pred].get("node_type") == "CDSView":
                        pred_meta = self.storage.get_view_metadata(pred_upper)
                        if pred_meta and _is_scope_match(pred_upper, pred_meta.get("view_type", "STANDARD")):
                            seen_names.add(pred_upper)
                            cand = CandidateView.from_metadata(pred_meta, retrieval_score=0.88)
                            secondary_views.append(cand)

        # 2. Targeted Vector Retrieval specifically for missing fields
        try:
            candidates_raw = self.storage.search_vector(
                query=missing_query,
                limit=limit,
                scope=scope,
            )
            for c in candidates_raw:
                v_name = c["view_name"].upper()
                if v_name not in seen_names:
                    seen_names.add(v_name)
                    meta = c.get("metadata") or self.storage.get_view_metadata(v_name) or {}
                    if meta:
                        cand = CandidateView.from_metadata(meta, retrieval_score=c.get("similarity", 0.85))
                        secondary_views.append(cand)
        except Exception as e:
            logger.debug(f"Error during secondary vector search: {e}")

        # 3. Rank secondary candidates strictly by missing-field coverage and retrieval score
        def _rank_sec(s: CandidateView) -> Tuple[float, float]:
            cov = len(s.fields & missing_set) / len(missing_set) if missing_set else 0.0
            return (cov, s.retrieval_score)

        secondary_views.sort(key=_rank_sec, reverse=True)
        return secondary_views

    def evaluate(
        self,
        business_requirement: str,
        domain: str = "General",
        required_fields: Optional[List[str]] = None,
        requested_fields: Optional[List[str]] = None,
        scope: str = "ALL",  # CUSTOM_ONLY, STANDARD_ONLY, ALL
        target_view_name: str = "Z_CUSTOM_VIEW",
        **kwargs,
    ) -> DecisionResult:
        """Run the multi-layer Tri-State Decision Pipeline."""
        # -------------------------------------------------------------
        # Layer 1: Requirement Analyzer (Semantic Router & Zero-Field Guard)
        # -------------------------------------------------------------
        req_fields = required_fields if required_fields is not None else (requested_fields if requested_fields is not None else [])
        analysis = self.analyzer.analyze(
            business_requirement=business_requirement,
            requested_fields=req_fields,
            domain=domain,
        )

        logger.info(
            f"Layer 1: Semantic Routing Complete:\n"
            f"  Intent: {analysis.intent} | Domain: {analysis.predicted_domain}\n"
            f"  Dimensions: {analysis.dimensions} | Measures: {analysis.measures}\n"
            f"  Aggregations: {analysis.aggregations}\n"
            f"  Canonical Fields: {analysis.requested_canonical_fields}\n"
            f"  Search Keywords: {analysis.dense_search_query}"
        )

        # -------------------------------------------------------------
        # Layer 2: Retrieval Layer (Top 30 Candidates Search + Association Expansion)
        # -------------------------------------------------------------
        scope_normalized = (scope or "ALL").upper().strip()
        candidates_raw = self.storage.search_vector(
            query=analysis.dense_search_query,
            limit=30,
            scope=scope_normalized,
        )

        seen_views: Set[str] = set()
        candidate_views: List[CandidateView] = []

        def _is_scope_match(view_name: str, view_type: str) -> bool:
            v_type = (view_type or "").upper()
            v_name = view_name.upper()
            is_custom = v_type == "CUSTOM" or v_name.startswith(("Z", "Y"))
            if scope_normalized in ("CUSTOM_ONLY", "CUSTOM"):
                return is_custom
            elif scope_normalized in ("STANDARD_ONLY", "STANDARD"):
                return not is_custom
            return True

        # Prioritize canonical seed views if they match requested scope
        for seed in analysis.canonical_seeds:
            seed_upper = seed.upper()
            seed_meta = self.storage.get_view_metadata(seed_upper)
            if seed_meta:
                v_type = seed_meta.get("view_type", "STANDARD")
                if _is_scope_match(seed_upper, v_type):
                    seen_views.add(seed_upper)
                    cand_view = CandidateView.from_metadata(seed_meta, retrieval_score=0.98)
                    candidate_views.append(cand_view)

        # Add ChromaDB vector search candidates
        for c in candidates_raw:
            v_name = c["view_name"].upper()
            if v_name not in seen_views:
                seen_views.add(v_name)
                meta = c.get("metadata") or self.storage.get_view_metadata(v_name) or {}
                if meta:
                    cand_view = CandidateView.from_metadata(meta, retrieval_score=c.get("similarity", 0.90))
                    candidate_views.append(cand_view)

        # Enrich candidate pool with 1-hop associations of top candidates for 2-view set-cover
        associated_targets: Set[str] = set()
        for cv in candidate_views[:5]:
            for a in cv.associations:
                t_name = a.get("target_view", "").upper()
                if t_name and t_name not in seen_views and t_name not in associated_targets:
                    associated_targets.add(t_name)

        for target in associated_targets:
            target_meta = self.storage.get_view_metadata(target)
            if target_meta:
                v_type = target_meta.get("view_type", "STANDARD")
                if _is_scope_match(target, v_type):
                    seen_views.add(target)
                    cand_view = CandidateView.from_metadata(target_meta, retrieval_score=0.80)
                    candidate_views.append(cand_view)

        # -------------------------------------------------------------
        # Layer 3: Metadata-Aware Reranking (Soft Boosting)
        # -------------------------------------------------------------
        candidate_views = rerank_candidates(
            candidate_views,
            intent=analysis.intent,
            required_fields=analysis.requested_canonical_fields,
        )

        # Collect all known catalog fields for accurate missing vs unresolved classification
        all_known_fields: Set[str] = set()
        for cv in candidate_views:
            all_known_fields.update(cv.fields)

        # Register views dynamically in canonical registry for reverse index lookup
        for cv in candidate_views:
            if cv.raw_metadata:
                try:
                    self.registry.register_view(cv.raw_metadata)
                except Exception:
                    pass

        # Targeted secondary retriever callback for EXTEND state
        def _secondary_retriever(prim: CandidateView, missing_f: List[str]) -> List[CandidateView]:
            logger.info(
                f"Targeted Secondary Retrieval triggered for primary '{prim.name}' "
                f"with missing fields: {missing_f}"
            )
            return self.retrieve_secondary_candidates(
                primary=prim,
                missing_fields=missing_f,
                scope=scope_normalized,
            )

        # Check infrastructure health
        infra_error = None
        if hasattr(self.storage, "graph_store"):
            if getattr(self.storage.graph_store, "is_broken", False):
                infra_error = "GRAPH_VALIDATION_UNAVAILABLE"

        # -------------------------------------------------------------
        # Layers 4, 5: Coverage Layer, Composition Solver & Graph Safety
        # -------------------------------------------------------------
        decision_result = self.solver.solve(
            analysis=analysis,
            candidates=candidate_views,
            all_known_fields=all_known_fields,
            target_view_name=target_view_name,
            secondary_retriever=_secondary_retriever,
            infrastructure_error=infra_error,
        )

        decision_result.candidate_views = [cv.raw_metadata for cv in candidate_views]

        logger.info(
            f"Layer 6: Decision Reached -> {decision_result.decision} "
            f"(Reason: {decision_result.reason_code}, Confidence: {decision_result.confidence}, Deployable: {decision_result.deployable})"
        )

        return decision_result
