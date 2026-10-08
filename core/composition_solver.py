"""Deterministic Tri-State Decision Engine (REUSE, EXTEND, CREATE) for Enterprise SAP CDS Views.

Implements pure Python set math and GraphRAG safety gates:
- Zero-Field Guard
- Analytical Intent & Grain Validation
- 2-View Set-Cover with Hard Cardinality Ban (SAFE: 1..1, 0..1; UNSAFE: 1..*, 0..*)
- Deterministic Tie-Breaking Hierarchy
- Strict JSON API Contract & Evidence-Based Confidence
- Zero Technical Debt & DCL Security (#CHECK)
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field

from core.query_expander import QueryExpander, QueryExpansion
from core.canonical_metadata import get_canonical_registry, CanonicalMetadataRegistry, ResolvedField

logger = logging.getLogger("composition_solver")
logging.basicConfig(level=logging.INFO)


SYNONYM_MAP: Dict[str, List[str]] = {
    "CUSTOMER": ["CUSTOMER", "SOLDTOPARTY", "PAYERPARTY", "BILLTOPARTY"],
    "TOTALNETAMOUNT": ["TOTALNETAMOUNT", "NETAMOUNT", "NETPRICEAMOUNT", "NETPRICE", "AMOUNT"],
    "NETAMOUNT": ["TOTALNETAMOUNT", "NETAMOUNT", "NETPRICEAMOUNT", "NETPRICE", "AMOUNT"],
    "NETPRICE": ["NETPRICEAMOUNT", "NETPRICE", "TOTALNETAMOUNT", "NETAMOUNT", "PRICE"],
    "NETPRICEAMOUNT": ["NETPRICEAMOUNT", "NETPRICE", "TOTALNETAMOUNT", "NETAMOUNT", "PRICE"],
    "SUPPLIER": ["SUPPLIER", "VENDOR", "CREDITOR"],
    "VENDOR": ["SUPPLIER", "VENDOR", "CREDITOR"],
    "BILLINGDOCUMENT": ["BILLINGDOCUMENT", "ACCOUNTINGDOCUMENT", "INVOICE"],
    "PRODUCT": ["PRODUCT", "MATERIAL"],
    "MATERIAL": ["PRODUCT", "MATERIAL"],
    "PLANT": ["PLANT", "FACTORY", "SITE"],
    "TARGETQUANTITY": ["TARGETQUANTITY", "PLANNEDQUANTITY", "QUANTITY", "TOTALQUANTITY", "MFGORDERPLANNEDTOTALQTY"],
    "BILLINGQUANTITY": ["BILLINGQUANTITY", "ORDERQUANTITY", "QUANTITY", "ACTUALDELIVERYQUANTITY"],
}


def _clean_field_name(raw_name: str) -> str:
    """Extract clean identifier from annotated CDS field names."""
    m = re.search(r'([A-Za-z0-9_]+)\s*$', str(raw_name).strip())
    return m.group(1) if m else str(raw_name).strip()


# Common analytical intent keywords
ANALYTICAL_PATTERNS = [
    r"\btotal\b",
    r"\bsum\b",
    r"\bgroup\s+by\b",
    r"\baggregat",
    r"\baverage\b",
    r"\bavg\b",
    r"\bcount\b",
    r"\bkpi\b",
    r"\bmeasure\b",
    r"\bcube\b",
    r"\banalytics?\b",
    r"\bmetric\b",
    r"\bdimension\b",
]


def detect_analytical_intent(text: str) -> bool:
    """Determine if requirement specifies aggregations, grouping, or analytical metrics."""
    if not text:
        return False
    text_lower = text.lower()
    return any(re.search(pat, text_lower) for pat in ANALYTICAL_PATTERNS)


def classify_cardinality(raw_card: Optional[str]) -> Tuple[bool, str]:
    """Map graph cardinalities to SAFE (1..1, 0..1) or UNSAFE (1..*, 0..*, or unknown/malformed).
    
    Hard Cardinality Ban: No exceptions for Header->Item relationships.
    Returns (is_safe: bool, normalized_cardinality: str)
    """
    if not raw_card:
        return False, "UNKNOWN"

    clean = re.sub(r'[\[\]\s\'"]', '', str(raw_card)).strip().lower()

    # Exact safe cardinalities
    if clean in ("1..1", "exact_one", "one_to_one"):
        return True, "1..1"
    if clean in ("0..1", "1", "to_one", "zero_or_one"):
        return True, "0..1"

    # Known unsafe cardinalities (to-many)
    if clean in ("1..*", "0..*", "*", "..*", "0..n", "1..n", "to_many", "many"):
        return False, clean.replace("n", "*")

    # Malformed or unrecognized
    return False, clean or "MALFORMED"


@dataclass
class CandidateView:
    """Structured candidate view contract with deterministic metadata."""
    name: str
    fields: Set[str]  # Normalized uppercase field names
    retrieval_score: float = 0.0
    base_score: float = 0.0
    adjusted_score: float = 0.0
    keys: List[str] = field(default_factory=list)
    is_analytical_cube: bool = False
    data_category: str = "#UNKNOWN"  # #CUBE, #QUERY, #FACT, #DIMENSION, #TEXT
    vdm_type: str = "UNKNOWN"
    associations: List[Dict[str, Any]] = field(default_factory=list)
    field_dict: Dict[str, str] = field(default_factory=dict)  # UPPER -> original casing
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.base_score == 0.0 and self.retrieval_score != 0.0:
            self.base_score = round(float(self.retrieval_score), 4)
        if self.adjusted_score == 0.0 and self.retrieval_score != 0.0:
            self.adjusted_score = round(float(self.retrieval_score), 4)
        if isinstance(self.raw_metadata, dict):
            if "base_score" not in self.raw_metadata:
                self.raw_metadata["base_score"] = self.base_score
            if "adjusted_score" not in self.raw_metadata:
                self.raw_metadata["adjusted_score"] = self.adjusted_score

    @classmethod
    def from_metadata(cls, meta: Dict[str, Any], retrieval_score: float = 0.0) -> "CandidateView":
        """Instantiate CandidateView from parsed view metadata dictionary."""
        name = meta.get("view_name", "UNKNOWN").strip().upper()
        fields_list = meta.get("fields", [])
        fields_set: Set[str] = set()
        field_dict: Dict[str, str] = {}
        keys_list: List[str] = []

        for f in fields_list:
            raw_name = (f.get("canonical_name") or f.get("name") or f.get("sap_name", "")) if isinstance(f, dict) else str(f)
            clean = _clean_field_name(raw_name)
            if clean:
                upper = clean.upper()
                fields_set.add(upper)
                field_dict[upper] = clean
                if isinstance(f, dict) and f.get("is_key"):
                    keys_list.append(clean)

        if not keys_list and meta.get("keys"):
            keys_list = [str(k) for k in meta.get("keys", [])]

        # Expand standard SAP canonical synonyms
        for syn_primary, syn_variants in SYNONYM_MAP.items():
            for v in syn_variants:
                if v in fields_set:
                    matched_clean = field_dict[v]
                    for s in syn_variants:
                        fields_set.add(s)
                        if s not in field_dict:
                            field_dict[s] = matched_clean
                    break

        raw_ddl = meta.get("raw_ddl", "")
        desc = meta.get("description", "")

        # 1. Detect or extract data category
        data_cat = meta.get("data_category") or ""
        if data_cat:
            data_cat = f"#{data_cat.upper().replace('#', '').strip()}"
        else:
            match = re.search(r"@Analytics\.dataCategory\s*:\s*#([A-Za-z0-9_]+)", raw_ddl, re.IGNORECASE)
            if match:
                data_cat = f"#{match.group(1).upper()}"
            elif re.search(r"@Analytics\.query\s*:\s*true", raw_ddl, re.IGNORECASE):
                data_cat = "#QUERY"
            else:
                combined = f"{raw_ddl} {desc} {name}".upper()
                if "CUBE" in combined:
                    data_cat = "#CUBE"
                elif "QUERY" in combined:
                    data_cat = "#QUERY"
                elif any(k in combined for k in ["TEXT", "DESCRIPTION"]):
                    data_cat = "#TEXT"
                elif any(k in combined for k in ["DIMENSION", "MASTER", "CUSTOMER", "SUPPLIER", "VENDOR", "PLANT", "COMPANY"]):
                    if not any(k in combined for k in ["ITEM", "ORDER", "DOCUMENT", "INVOICE", "DELIVERY", "JOURNAL"]):
                        data_cat = "#DIMENSION"
                    else:
                        data_cat = "#FACT"
                elif any(k in combined for k in ["ITEM", "FACT", "ORDER", "DOCUMENT", "INVOICE", "DELIVERY", "LINE"]):
                    data_cat = "#FACT"
                else:
                    data_cat = "#UNKNOWN"

        # 2. Detect analytical cube / query
        is_cube = False
        if "is_analytical_cube" in meta:
            is_cube = bool(meta["is_analytical_cube"])
        else:
            is_cube = (data_cat in ("#CUBE", "#QUERY", "#AGGREGATION"))

        assocs = meta.get("associations", [])
        score_val = round(float(retrieval_score), 4)
        base_val = meta.get("base_score", score_val)
        adj_val = meta.get("adjusted_score", score_val)
        meta["base_score"] = base_val
        meta["adjusted_score"] = adj_val

        return cls(
            name=name,
            fields=fields_set,
            retrieval_score=adj_val,
            base_score=base_val,
            adjusted_score=adj_val,
            keys=keys_list,
            is_analytical_cube=is_cube,
            data_category=data_cat,
            associations=assocs,
            field_dict=field_dict,
            raw_metadata=meta,
        )


@dataclass
class RequirementAnalysis:
    """Normalized requirement contract with analytical intent, dimensions, measures, and canonical field resolution."""
    business_requirement: str
    requested_fields: List[str]
    requested_canonical_fields: List[str]
    is_analytical: bool
    predicted_domain: str = "General"
    dense_search_query: str = ""
    canonical_seeds: List[str] = field(default_factory=list)
    expansion: Optional[QueryExpansion] = None
    intent: str = "TRANSACTIONAL"
    dimensions: List[str] = field(default_factory=list)
    measures: List[str] = field(default_factory=list)
    aggregations: List[str] = field(default_factory=list)
    likely_domains: List[str] = field(default_factory=list)
    search_keywords: str = ""
    business_objects: List[str] = field(default_factory=list)
    grain: List[str] = field(default_factory=list)
    canonical_field_mapping: List[Dict[str, Any]] = field(default_factory=list)
    unvalidated_fields: List[str] = field(default_factory=list)


class RequirementAnalyzer:
    """Layer 1: Normalizes requirements, routes intent, and performs metadata-backed canonical field resolution."""

    def __init__(
        self,
        query_expander: Optional[QueryExpander] = None,
        registry: Optional[CanonicalMetadataRegistry] = None,
    ):
        self.query_expander = query_expander or QueryExpander()
        self.registry = registry or get_canonical_registry()

    def analyze(
        self,
        business_requirement: str,
        requested_fields: List[str],
        domain: str = "General",
    ) -> RequirementAnalysis:
        # 1. Zero-Field Guard
        clean_fields = [f.strip() for f in requested_fields if f and f.strip()]
        if not clean_fields:
            raise ValueError("At least one requested field is required")

        # 2. Run Semantic Router (Query Expander)
        router_output = self.query_expander.expand(
            business_requirement=business_requirement,
            requested_fields=clean_fields,
        )

        # 3. Canonical Field Resolution Layer (Metadata-backed validation)
        # All user-requested fields are preserved in canonical_fields
        canonical_fields: List[str] = []
        field_mapping: List[Dict[str, Any]] = []
        validated_fields: List[str] = []
        unvalidated_fields: List[str] = []

        for term in clean_fields:
            res = self.registry.field_resolver.resolve(term, domain_hint=router_output.predicted_domain)
            canon = res.canonical_name if res.validated else term
            if canon not in canonical_fields:
                canonical_fields.append(canon)
            field_mapping.append({
                "requested_term": term,
                "canonical_name": canon,
                "confidence": res.confidence,
                "source": res.source,
                "validated": res.validated,
            })
            if res.validated:
                validated_fields.append(canon)
            else:
                unvalidated_fields.append(term)

        # Supplement with router dimensions & measures only if validated
        for term in router_output.dimensions + router_output.measures:
            res = self.registry.field_resolver.resolve(term, domain_hint=router_output.predicted_domain)
            if res.validated and res.canonical_name not in canonical_fields:
                canonical_fields.append(res.canonical_name)
                validated_fields.append(res.canonical_name)

        if not canonical_fields:
            raise ValueError("At least one requested field is required")

        is_analytical = (router_output.intent == "ANALYTICAL")

        return RequirementAnalysis(
            business_requirement=business_requirement,
            requested_fields=clean_fields,
            requested_canonical_fields=canonical_fields,
            is_analytical=is_analytical,
            predicted_domain=router_output.predicted_domain or domain,
            dense_search_query=router_output.search_keywords,
            canonical_seeds=router_output.canonical_seed_views,
            expansion=router_output,
            intent=router_output.intent,
            dimensions=router_output.dimensions,
            measures=router_output.measures,
            aggregations=router_output.aggregations,
            likely_domains=router_output.likely_domains,
            search_keywords=router_output.search_keywords,
            business_objects=router_output.business_objects,
            grain=router_output.grain,
            canonical_field_mapping=field_mapping,
            unvalidated_fields=unvalidated_fields,
        )


@dataclass
class DecisionResult:
    """Standardized Decision Result adhering strictly to Enterprise Tri-State API Contract."""
    decision: str  # "REUSE", "EXTEND", "CREATE"
    reason_code: str
    confidence: float
    primary_view: Any  # dict for REUSE ({"name": str, "coverage": float}) or str for EXTEND/CREATE
    secondary_view: Optional[str] = None
    association: Optional[Dict[str, Any]] = None
    coverage: float = 0.0
    requested_fields: List[str] = field(default_factory=list)
    resolved_fields: List[str] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    unresolved_fields: List[str] = field(default_factory=list)
    deployable: bool = False
    warnings: List[str] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    ddl: Optional[str] = None
    decision_trace: Dict[str, Any] = field(default_factory=dict)
    # UI / Graph helper attributes
    supporting_views: List[Dict[str, Any]] = field(default_factory=list)
    join_plan: Optional[Dict[str, Any]] = None
    query_expansion: Optional[Dict[str, Any]] = None
    rationale: str = ""
    candidate_views: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.reason_codes and self.reason_code:
            self.reason_codes = [self.reason_code]

    @property
    def confidence_score(self) -> float:
        """Alias for confidence."""
        return self.confidence

    @property
    def primary_view_name(self) -> str:
        """Helper to get string view name regardless of primary_view schema format."""
        if isinstance(self.primary_view, dict):
            return self.primary_view.get("name", "")
        return str(self.primary_view or "")

    @property
    def match_percentage(self) -> float:
        """Percentage match for backwards compatibility with UI metrics."""
        return round(self.coverage * 100.0, 1)

    @property
    def covered_fields(self) -> List[str]:
        return self.resolved_fields

    @property
    def uncovered_fields(self) -> List[str]:
        return self.unresolved_fields

    def to_dict(self) -> Dict[str, Any]:
        """Produce the standardized enterprise contract matching Product-Level V1 Schema."""
        primary_out = self.primary_view
        secondary_out = self.secondary_view
        intent_dict = self.decision_trace.get("intent") or (self.query_expansion or {})

        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "intent": intent_dict,
            "primary_view": primary_out,
            "secondary_view": secondary_out,
            "coverage": self.coverage,
            "requested_fields": self.requested_fields,
            "resolved_fields": self.resolved_fields,
            "missing_fields": self.missing_fields,
            "unresolved_fields": self.unresolved_fields,
            "association": self.association or {
                "target": None,
                "cardinality": None,
                "safe": False,
                "join_keys": [],
            },
            "deployable": self.deployable,
            "warnings": self.warnings,
            "reason_code": self.reason_code,
            "reason_codes": self.reason_codes or [self.reason_code],
            "ddl": self.ddl,
            "decision_trace": self.decision_trace,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class RequirementSolver:
    """Pure Python set-math Tri-State Decision Engine with Graph Safety Gates."""

    def __init__(self):
        pass

    @classmethod
    def evaluate_pair(
        cls,
        primary: CandidateView,
        secondary: CandidateView,
        requested_set: Set[str],
        is_analytical: bool = False,
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """Evaluate a 2-view combination (Primary + Secondary) against strict GraphRAG safety gates.
        
        A valid pair must have:
        1. 100% combined field coverage: requested_set <= (primary.fields | secondary.fields)
        2. A direct 1-hop Knowledge Graph association
        3. Safe cardinality (strictly 1..1 or 0..1)
        4. Valid, accessible join keys in primary view
        5. Analytical grain compatibility if is_analytical is True
        
        Returns: (is_valid: bool, evaluation_dict: Optional[dict], rejection_reason: str)
        """
        # Condition 1: Combined 100% field coverage
        combined_fields = primary.fields | secondary.fields
        if not (requested_set <= combined_fields):
            return False, None, "INSUFFICIENT_COMBINED_COVERAGE"

        # Condition 2: Direct Association
        assoc_info = cls._find_direct_association(primary, secondary)
        if not assoc_info:
            return False, None, "NO_DIRECT_ASSOCIATION"

        # Condition 3: Hard Cardinality Ban (SAFE: 1..1, 0..1; UNSAFE: 1..*, 0..*)
        card_str = assoc_info.get("cardinality", "")
        is_safe, clean_card = classify_cardinality(card_str)
        if not is_safe:
            return False, None, f"UNSAFE_CARDINALITY_{clean_card}"

        # Condition 4: Valid join keys exist and are accessible in primary view
        join_keys, on_condition = cls._resolve_join_keys(primary, secondary, assoc_info)
        if not join_keys:
            return False, None, "JOIN_KEYS_NOT_ACCESSIBLE: MISSING_JOIN_KEYS"

        accessible_in_primary = all(k.upper() in primary.fields for k in join_keys)
        if not accessible_in_primary:
            missing_keys = [k for k in join_keys if k.upper() not in primary.fields]
            return False, None, f"JOIN_KEYS_NOT_ACCESSIBLE: MISSING_JOIN_KEYS {missing_keys}"

        # Grain validation for EXTEND if analytical
        if is_analytical and not primary.is_analytical_cube:
            return False, None, "ANALYTICAL_GRAIN_MISMATCH"

        # Compute ranking metrics for tie-breaking:
        cov_prim = len(primary.fields & requested_set) / len(requested_set)
        card_rank = 2 if clean_card == "1..1" else 1
        assoc_dir_rank = 1 if assoc_info.get("from_primary") else 0
        hybrid_score = round(primary.retrieval_score + secondary.retrieval_score, 4)

        return True, {
            "primary": primary,
            "secondary": secondary,
            "primary_cov": cov_prim,
            "card_rank": card_rank,
            "clean_card": clean_card,
            "assoc_dir_rank": assoc_dir_rank,
            "hybrid_score": hybrid_score,
            "assoc_info": assoc_info,
            "join_keys": join_keys,
            "on_condition": on_condition,
        }, "VALID"

    def solve(
        self,
        analysis: RequirementAnalysis,
        candidates: List[CandidateView],
        all_known_fields: Optional[Set[str]] = None,
        target_view_name: str = "Z_CUSTOM_VIEW",
        secondary_retriever: Optional[Callable[[CandidateView, List[str]], List[CandidateView]]] = None,
        secondary_candidates: Optional[List[CandidateView]] = None,
        infrastructure_error: Optional[str] = None,
    ) -> DecisionResult:
        """Solve requirement using pure Python set math and GraphRAG safety gates."""
        # 0. Infrastructure Failure Protection (Section 27)
        if infrastructure_error:
            logger.error(f"Infrastructure failure detected during solve: {infrastructure_error}")
            return DecisionResult(
                decision="CREATE",
                reason_code=infrastructure_error,
                confidence=0.0,
                primary_view="UNKNOWN",
                coverage=0.0,
                requested_fields=analysis.requested_fields,
                resolved_fields=[],
                missing_fields=list(analysis.requested_canonical_fields),
                unresolved_fields=list(analysis.requested_canonical_fields),
                deployable=False,
                warnings=[
                    f"INFRASTRUCTURE ERROR: {infrastructure_error}. Safe composition could not be verified.",
                    "SECURITY: Generated DDL uses #CHECK. A DCL role is required.",
                ],
                reason_codes=[infrastructure_error],
                ddl=None,
                decision_trace={
                    "error": infrastructure_error,
                    "intent": analysis.expansion.to_dict() if analysis.expansion else {},
                    "final_decision": "CREATE",
                    "final_reason_codes": [infrastructure_error],
                },
                rationale=f"Infrastructure failure ({infrastructure_error}) prevented safe composition verification.",
            )

        # Zero-Field Guard
        if not analysis.requested_canonical_fields:
            raise ValueError("At least one requested field is required")

        requested_set = {f.upper() for f in analysis.requested_canonical_fields}
        all_known = {f.upper() for f in (all_known_fields or set())}
        pair_evaluations_trace: List[Dict[str, Any]] = []

        # Validate unvalidated fields against repository candidate fields
        pool_fields = set(all_known)
        for c in candidates:
            pool_fields.update(c.fields)
        if secondary_candidates:
            for s in secondary_candidates:
                pool_fields.update(s.fields)

        remaining_unvalidated = [
            uf for uf in analysis.unvalidated_fields
            if uf.upper() not in pool_fields
        ]

        # -------------------------------------------------------------
        # STATE 1: REUSE (100% Single View Match)
        # -------------------------------------------------------------
        for cand in candidates:
            if requested_set <= cand.fields and not remaining_unvalidated:
                # Grain validation
                if analysis.is_analytical and not cand.is_analytical_cube:
                    logger.info(
                        f"View {cand.name} covers 100% fields but failed analytical grain check "
                        f"(query requires aggregation/cube, but view is raw transaction). Rejecting REUSE."
                    )
                    break  # Evaluates for EXTEND / CREATE
                
                # REUSE Confirmed!
                resolved = [cand.field_dict.get(f.upper(), f) for f in analysis.requested_canonical_fields]
                join_plan = {
                    "primary_view": cand.name,
                    "field_mapping": {
                        f: {
                            "source_view": cand.name,
                            "alias": "_Primary",
                            "view_field": cand.field_dict.get(f.upper(), f),
                            "field_expr": f"_Primary.{cand.field_dict.get(f.upper(), f)} as {f}",
                        }
                        for f in analysis.requested_canonical_fields
                    },
                    "required_associations": [],
                    "unresolved_fields": [],
                }

                decision_trace = {
                    "intent": analysis.expansion.to_dict() if analysis.expansion else {},
                    "canonical_field_mapping": analysis.canonical_field_mapping,
                    "primary_candidates": [
                        {
                            "view": c.name,
                            "base_score": getattr(c, "base_score", 0.0),
                            "adjusted_score": getattr(c, "adjusted_score", 0.0),
                            "coverage": round(len(c.fields & requested_set) / len(requested_set), 2),
                        }
                        for c in candidates[:10]
                    ],
                    "missing_fields": [],
                    "secondary_candidates": [],
                    "pair_evaluations": [],
                    "final_decision": "REUSE",
                    "final_reason_codes": ["FULL_SINGLE_VIEW_COVERAGE"],
                }

                return DecisionResult(
                    decision="REUSE",
                    reason_code="FULL_SINGLE_VIEW_MATCH",
                    confidence=1.0,
                    primary_view={"name": cand.name, "coverage": 1.0},
                    coverage=1.0,
                    requested_fields=analysis.requested_fields,
                    resolved_fields=resolved,
                    missing_fields=[],
                    unresolved_fields=[],
                    deployable=True,
                    warnings=[],
                    reason_codes=["FULL_SINGLE_VIEW_COVERAGE", "FULL_SINGLE_VIEW_MATCH"],
                    ddl=None,  # REUSE must NEVER generate DDL
                    decision_trace=decision_trace,
                    supporting_views=[],
                    join_plan=join_plan,
                    query_expansion=analysis.expansion.to_dict() if analysis.expansion else None,
                    rationale=f"Candidate view '{cand.name}' achieves 100% coverage with compatible grain. Fully reusable.",
                )

        # -------------------------------------------------------------
        # STATE 2: EXTEND (Targeted Secondary Retrieval + 2-View Set-Cover)
        # -------------------------------------------------------------
        valid_compositions: List[Dict[str, Any]] = []
        rejected_for_cardinality = False

        # Rank candidates by individual coverage on requested_set, tie-break with retrieval score
        def _score_prim(c: CandidateView) -> Tuple[float, float]:
            cov = len(c.fields & requested_set) / len(requested_set)
            return (cov, c.retrieval_score)

        sorted_primaries = sorted(candidates, key=_score_prim, reverse=True)

        if sorted_primaries:
            best_primary = sorted_primaries[0]
            unresolved_fields = [
                f for f in analysis.requested_canonical_fields
                if f.upper() not in best_primary.fields
            ]

            # 1. Gather all potential secondary candidates
            all_secondaries: List[CandidateView] = []
            seen_sec_names: Set[str] = {best_primary.name}

            # 1a. Explicitly passed secondary candidates
            if secondary_candidates:
                for sec in secondary_candidates:
                    if sec.name not in seen_sec_names:
                        seen_sec_names.add(sec.name)
                        all_secondaries.append(sec)

            # 1b. Targeted secondary retrieval for unresolved fields
            if secondary_retriever and unresolved_fields:
                retrieved = secondary_retriever(best_primary, unresolved_fields)
                for sec in retrieved:
                    if sec.name not in seen_sec_names:
                        seen_sec_names.add(sec.name)
                        all_secondaries.append(sec)

            # 1c. Include remaining candidates from initial candidate pool
            for c in candidates:
                if c.name not in seen_sec_names:
                    seen_sec_names.add(c.name)
                    all_secondaries.append(c)

            # 2. Rank secondary candidates by missing-field coverage and retrieval score
            missing_set = {f.upper() for f in unresolved_fields}

            def _rank_secondary(sec: CandidateView) -> Tuple[float, float]:
                cov_missing = len(sec.fields & missing_set) / len(missing_set) if missing_set else 0.0
                return (cov_missing, sec.retrieval_score)

            ranked_secondaries = sorted(all_secondaries, key=_rank_secondary, reverse=True)

            # 3. Evaluate 2-view combinations
            tested_pairs: Set[Tuple[str, str]] = set()

            # Test top primary candidates against ranked secondaries
            test_primaries = sorted_primaries[:5]
            for prim in test_primaries:
                for sec in ranked_secondaries:
                    if prim.name == sec.name:
                        continue

                    pair_key = tuple(sorted([prim.name, sec.name]))
                    if pair_key in tested_pairs:
                        continue
                    tested_pairs.add(pair_key)

                    # Evaluate both possible orientations: (prim, sec) and (sec, prim)
                    for eval_prim, eval_sec in [(prim, sec), (sec, prim)]:
                        is_valid, comp_dict, rejection_reason = self.evaluate_pair(
                            primary=eval_prim,
                            secondary=eval_sec,
                            requested_set=requested_set,
                            is_analytical=analysis.is_analytical,
                        )

                        assoc_info = self._find_direct_association(eval_prim, eval_sec)
                        card_clean = classify_cardinality(assoc_info.get("cardinality") if assoc_info else "")[1] if assoc_info else "NONE"
                        join_keys_found, _ = self._resolve_join_keys(eval_prim, eval_sec, assoc_info) if assoc_info else ([], "")

                        pair_evaluations_trace.append({
                            "primary": eval_prim.name,
                            "secondary": eval_sec.name,
                            "combined_coverage": round(len((eval_prim.fields | eval_sec.fields) & requested_set) / len(requested_set), 2),
                            "association": assoc_info is not None,
                            "cardinality": card_clean,
                            "join_keys": join_keys_found,
                            "result": "ACCEPTED" if is_valid else "REJECTED",
                            "reason_code": "VALID" if is_valid else rejection_reason,
                        })

                        if is_valid and comp_dict:
                            valid_compositions.append(comp_dict)
                        elif "UNSAFE_CARDINALITY" in rejection_reason:
                            logger.info(
                                f"Rejecting composition {eval_prim.name} + {eval_sec.name}: "
                                f"{rejection_reason}"
                            )
                            rejected_for_cardinality = True

        # Tie-Breaking Selection:
        if valid_compositions:
            # Sort according to strict hierarchy:
            # 1. primary_cov (descending)
            # 2. card_rank (descending: 1..1 > 0..1)
            # 3. Direct association ownership (from_primary: Primary -> Secondary)
            # 4. hybrid_score (descending)
            valid_compositions.sort(
                key=lambda x: (
                    x["primary_cov"],
                    x["card_rank"],
                    x["assoc_dir_rank"],
                    x["hybrid_score"],
                ),
                reverse=True,
            )
            winner = valid_compositions[0]
            primary = winner["primary"]
            secondary = winner["secondary"]
            clean_card = winner["clean_card"]
            join_keys = winner["join_keys"]
            on_cond = winner["on_condition"]
            assoc_info = winner.get("assoc_info", {})
            sec_alias = assoc_info.get("alias") or f"_{secondary.name.replace('I_', '').replace('C_', '')}"

            # Build field mappings and join plan
            field_mapping: Dict[str, Dict[str, Any]] = {}
            for req_f in analysis.requested_canonical_fields:
                rf_upper = req_f.upper()
                if rf_upper in primary.fields:
                    prim_f = primary.field_dict.get(rf_upper, req_f)
                    field_mapping[req_f] = {
                        "source_view": primary.name,
                        "alias": "_Primary",
                        "view_field": prim_f,
                        "field_expr": f"_Primary.{prim_f} as {req_f}",
                    }
                elif rf_upper in secondary.fields:
                    sec_f = secondary.field_dict.get(rf_upper, req_f)
                    field_mapping[req_f] = {
                        "source_view": secondary.name,
                        "alias": sec_alias,
                        "view_field": sec_f,
                        "field_expr": f"{sec_alias}.{sec_f} as {req_f}",
                    }

            assoc_obj = {
                "target": secondary.name,
                "alias": sec_alias,
                "cardinality": clean_card,
                "safe": True,
                "join_keys": join_keys,
            }

            supporting_views = [
                {
                    "target_view": secondary.name,
                    "alias": sec_alias,
                    "cardinality": f"[{clean_card}]",
                    "on_condition": on_cond,
                    "covered_fields": [
                        secondary.field_dict.get(rf.upper(), rf)
                        for rf in analysis.requested_canonical_fields
                        if rf.upper() in secondary.fields and rf.upper() not in primary.fields
                    ],
                }
            ]

            join_plan = {
                "primary_view": primary.name,
                "field_mapping": field_mapping,
                "required_associations": supporting_views,
                "unresolved_fields": [],
            }

            # Generate wrapper CDS DDL with #CHECK authorization
            extend_ddl = self._generate_extend_ddl(
                new_view_name=target_view_name,
                primary=primary,
                secondary=secondary,
                sec_alias=sec_alias,
                clean_card=clean_card,
                on_condition=on_cond,
                field_mapping=field_mapping,
                label=analysis.business_requirement[:60],
            )

            decision_trace = {
                "intent": analysis.expansion.to_dict() if analysis.expansion else {},
                "canonical_field_mapping": analysis.canonical_field_mapping,
                "primary_candidates": [
                    {
                        "view": c.name,
                        "base_score": getattr(c, "base_score", 0.0),
                        "adjusted_score": getattr(c, "adjusted_score", 0.0),
                        "coverage": round(len(c.fields & requested_set) / len(requested_set), 2),
                    }
                    for c in sorted_primaries[:10]
                ],
                "missing_fields": unresolved_fields,
                "secondary_candidates": [
                    {
                        "view": s.name,
                        "missing_field_coverage": round(len(s.fields & missing_set) / len(missing_set), 2) if missing_set else 0.0,
                        "score": s.retrieval_score,
                    }
                    for s in ranked_secondaries[:10]
                ],
                "pair_evaluations": pair_evaluations_trace,
                "final_decision": "EXTEND",
                "final_reason_codes": ["SAFE_TWO_VIEW_COMPOSITION"],
            }

            return DecisionResult(
                decision="EXTEND",
                reason_code="SAFE_TWO_VIEW_COMPOSITION",
                confidence=0.95,
                primary_view=primary.name,
                secondary_view=secondary.name,
                association=assoc_obj,
                coverage=1.0,
                requested_fields=analysis.requested_fields,
                resolved_fields=list(analysis.requested_canonical_fields),
                missing_fields=[],
                unresolved_fields=[],
                deployable=True,
                warnings=[
                    "SECURITY: Authorization check enabled (#CHECK). A corresponding DCL role may be required.",
                    "SECURITY: A DCL authorization role must be assigned manually.",
                ],
                reason_codes=["SAFE_TWO_VIEW_COMPOSITION"],
                ddl=extend_ddl,
                decision_trace=decision_trace,
                supporting_views=supporting_views,
                join_plan=join_plan,
                query_expansion=analysis.expansion.to_dict() if analysis.expansion else None,
                rationale=(
                    f"Composed primary view '{primary.name}' with secondary view '{secondary.name}' "
                    f"via safe direct association [{clean_card}] achieving 100% field coverage."
                ),
            )

        # -------------------------------------------------------------
        # STATE 3: CREATE (Fallback Partial Match)
        # -------------------------------------------------------------
        # Select best partial-match primary view
        if not candidates:
            # No candidates available
            create_ddl = (
                f"@EndUserText.label: '{analysis.business_requirement[:60]}'\n"
                f"@AccessControl.authorizationCheck: #CHECK\n"
                f"define view entity {target_view_name}\n"
                f"  as select from UNKNOWN as _Primary\n"
                f"{{\n"
            )
            for f in analysis.requested_canonical_fields:
                create_ddl += f"  // TODO: Unresolved user field: {f}\n"
            create_ddl += "}"

            decision_trace = {
                "intent": analysis.expansion.to_dict() if analysis.expansion else {},
                "canonical_field_mapping": analysis.canonical_field_mapping,
                "primary_candidates": [],
                "missing_fields": list(analysis.requested_canonical_fields),
                "secondary_candidates": [],
                "pair_evaluations": [],
                "final_decision": "CREATE",
                "final_reason_codes": ["NO_CANDIDATE_VIEWS"],
            }

            return DecisionResult(
                decision="CREATE",
                reason_code="NO_CANDIDATE_VIEWS",
                confidence=0.0,
                primary_view="UNKNOWN",
                coverage=0.0,
                requested_fields=analysis.requested_fields,
                resolved_fields=[],
                missing_fields=list(analysis.requested_canonical_fields),
                unresolved_fields=list(analysis.requested_canonical_fields),
                deployable=False,
                warnings=[
                    "No candidate views found in repository.",
                    "SECURITY: Generated DDL uses #CHECK. A DCL role is required.",
                ],
                reason_codes=["NO_CANDIDATE_VIEWS"],
                ddl=create_ddl,
                decision_trace=decision_trace,
                supporting_views=[],
                join_plan={
                    "primary_view": "UNKNOWN",
                    "field_mapping": {},
                    "required_associations": [],
                    "unresolved_fields": list(analysis.requested_canonical_fields),
                },
                query_expansion=analysis.expansion.to_dict() if analysis.expansion else None,
                rationale="No candidate CDS views found in repository. Full view synthesis required.",
            )

        # Pick candidate with highest coverage, tie-break with retrieval score
        def _score_cand(c: CandidateView) -> Tuple[float, float]:
            cov = len(c.fields & requested_set) / len(requested_set)
            return (cov, c.retrieval_score)

        sorted_candidates = sorted(candidates, key=_score_cand, reverse=True)
        best_cand = sorted_candidates[0]
        resolved = [best_cand.field_dict.get(f.upper(), f) for f in analysis.requested_canonical_fields if f.upper() in best_cand.fields]
        unmapped = [f for f in analysis.requested_canonical_fields if f.upper() not in best_cand.fields]

        # Distinguish missing_fields (known in catalog/candidates) vs unresolved_fields (completely unknown)
        pool_fields = set()
        for c in candidates:
            pool_fields.update(c.fields)
        pool_fields.update(all_known)

        missing_fields = [f for f in unmapped if f.upper() in pool_fields]
        unresolved_fields = [f for f in unmapped if f.upper() not in pool_fields]
        # Include any terms that failed canonical validation
        for uf in analysis.unvalidated_fields:
            if uf not in unresolved_fields and uf not in missing_fields:
                unresolved_fields.append(uf)

        coverage_ratio = round(len(resolved) / len(analysis.requested_canonical_fields), 2)

        # Determine precise reason codes (Section 23)
        reason_codes: List[str] = ["NO_SINGLE_VIEW"]
        has_card_failure = rejected_for_cardinality or any("UNSAFE_CARDINALITY" in p.get("reason_code", "") for p in pair_evaluations_trace)
        has_join_key_failure = any("MISSING_JOIN_KEYS" in p.get("reason_code", "") or "JOIN_KEYS" in p.get("reason_code", "") for p in pair_evaluations_trace)
        has_no_assoc = any(p.get("reason_code") == "NO_DIRECT_ASSOCIATION" for p in pair_evaluations_trace)

        if has_card_failure or has_join_key_failure or has_no_assoc or missing_fields:
            reason_codes.append("NO_SAFE_ASSOCIATION")
        if has_join_key_failure:
            reason_codes.append("MISSING_JOIN_KEYS")
        if has_card_failure:
            reason_codes.append("UNSAFE_CARDINALITY")
        if has_no_assoc:
            reason_codes.append("NO_DIRECT_ASSOCIATION")
        if 'all_secondaries' in locals() and not all_secondaries:
            reason_codes.append("NO_SECONDARY_CANDIDATE")
        if len(reason_codes) == 1:
            reason_codes.append("PARTIAL_FIELD_MATCH")

        if analysis.unvalidated_fields:
            reason_codes.append("UNRESOLVED_CANONICAL_FIELD")

        primary_reason_code = reason_codes[1] if len(reason_codes) > 1 else reason_codes[0]

        warnings: List[str] = []
        for mf in missing_fields:
            warnings.append(f"{mf} could not be safely mapped to the graph.")
        for uf in unresolved_fields:
            warnings.append(f"{uf} could not be safely mapped to the graph.")
        warnings.append("SECURITY: Generated DDL uses #CHECK. A corresponding DCL role is required.")
        warnings.append("SECURITY: Generated DDL uses #CHECK. A DCL role is required.")

        # Best-effort DDL with commented-out unresolved/missing fields and #CHECK
        create_ddl = self._generate_create_ddl(
            new_view_name=target_view_name,
            primary=best_cand,
            requested_fields=analysis.requested_canonical_fields,
            label=analysis.business_requirement[:60],
        )

        join_plan = {
            "primary_view": best_cand.name,
            "field_mapping": {
                f: {
                    "source_view": best_cand.name,
                    "alias": "_Primary",
                    "view_field": best_cand.field_dict.get(f.upper(), f),
                    "field_expr": f"_Primary.{best_cand.field_dict.get(f.upper(), f)} as {f}",
                }
                for f in analysis.requested_canonical_fields
                if f.upper() in best_cand.fields
            },
            "required_associations": [],
            "unresolved_fields": unmapped,
        }

        decision_trace = {
            "intent": analysis.expansion.to_dict() if analysis.expansion else {},
            "canonical_field_mapping": analysis.canonical_field_mapping,
            "primary_candidates": [
                {
                    "view": c.name,
                    "base_score": getattr(c, "base_score", 0.0),
                    "adjusted_score": getattr(c, "adjusted_score", 0.0),
                    "coverage": round(len(c.fields & requested_set) / len(requested_set), 2),
                }
                for c in sorted_candidates[:10]
            ],
            "missing_fields": missing_fields,
            "secondary_candidates": [
                {
                    "view": s.name,
                    "missing_field_coverage": round(len(s.fields & missing_set) / len(missing_set), 2) if 'missing_set' in locals() and missing_set else 0.0,
                    "score": s.retrieval_score,
                }
                for s in (ranked_secondaries[:10] if 'ranked_secondaries' in locals() else [])
            ],
            "pair_evaluations": pair_evaluations_trace,
            "final_decision": "CREATE",
            "final_reason_codes": reason_codes,
        }

        return DecisionResult(
            decision="CREATE",
            reason_code=primary_reason_code,
            confidence=round(coverage_ratio * 0.8, 2),  # Partial match evidence-based confidence
            primary_view=best_cand.name,
            coverage=coverage_ratio,
            requested_fields=analysis.requested_fields,
            resolved_fields=resolved,
            missing_fields=missing_fields,
            unresolved_fields=unresolved_fields,
            deployable=False,  # CREATE is NEVER deployable
            warnings=warnings,
            reason_codes=reason_codes,
            ddl=create_ddl,
            decision_trace=decision_trace,
            supporting_views=[],
            join_plan=join_plan,
            query_expansion=analysis.expansion.to_dict() if analysis.expansion else None,
            rationale=(
                f"Candidate '{best_cand.name}' provides partial coverage ({int(coverage_ratio * 100)}%). "
                f"Missing fields {unmapped} could not be safely joined through verified 1..1 or 0..1 associations. "
                f"Synthesis required (informational only)."
            ),
        )

    # -------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------
    @staticmethod
    def _find_direct_association(view_a: CandidateView, view_b: CandidateView) -> Optional[Dict[str, Any]]:
        """Look for a direct association between View A and View B in either direction."""
        # 1. A -> B
        for a in view_a.associations:
            target = (a.get("target_view") or a.get("target", "")).upper()
            if target == view_b.name:
                return {
                    "from_primary": True,
                    "source": view_a.name,
                    "target": view_b.name,
                    "alias": a.get("alias", f"_{view_b.name.replace('I_', '').replace('C_', '')}"),
                    "cardinality": a.get("cardinality", "[0..1]"),
                    "on_condition": a.get("on_condition", ""),
                    "raw": a,
                }

        # 2. B -> A (Reverse association: fan-out danger unless strictly 1..1)
        for a in view_b.associations:
            target = (a.get("target_view") or a.get("target", "")).upper()
            if target == view_a.name:
                raw_card = a.get("cardinality", "[0..1]")
                rev_card = "[1..1]" if "1..1" in raw_card else "[0..*]"
                return {
                    "from_primary": False,
                    "source": view_b.name,
                    "target": view_a.name,
                    "alias": a.get("alias", f"_{view_a.name.replace('I_', '').replace('C_', '')}"),
                    "cardinality": rev_card,
                    "on_condition": a.get("on_condition", ""),
                    "raw": a,
                }

        return None

    @staticmethod
    def _resolve_join_keys(
        primary: CandidateView,
        secondary: CandidateView,
        assoc_info: Dict[str, Any],
    ) -> Tuple[List[str], str]:
        """Extract join keys and normalize the ON condition for the association."""
        on_cond = assoc_info.get("on_condition", "")
        join_keys: List[str] = []

        # 1. Try parsing join keys from on_condition
        if on_cond:
            # Look for expressions like $projection.Field = _Alias.Field or _Primary.Field = _Alias.Field
            matches = re.findall(r'(?:\$projection|_Primary)\.([A-Za-z0-9_]+)', on_cond, re.IGNORECASE)
            for m in matches:
                if m not in join_keys:
                    join_keys.append(m)

        # 2. Check raw association metadata for explicit join_keys
        raw_keys = assoc_info.get("raw", {}).get("join_keys", [])
        if not join_keys and raw_keys:
            join_keys = list(raw_keys)

        # 3. If not found in ON condition or raw_keys, check secondary view keys
        if not join_keys and secondary.keys:
            common_keys = [k for k in secondary.keys if k.upper() in primary.fields]
            if common_keys:
                join_keys = common_keys
            else:
                join_keys = list(secondary.keys)

        sec_alias = assoc_info.get("alias") or f"_{secondary.name.replace('I_', '').replace('C_', '')}"
        if not on_cond and join_keys:
            conds = [f"_Primary.{k} = {sec_alias}.{k}" for k in join_keys]
            on_cond = " and ".join(conds)

        return join_keys, on_cond

    @staticmethod
    def _generate_extend_ddl(
        new_view_name: str,
        primary: CandidateView,
        secondary: CandidateView,
        sec_alias: str,
        clean_card: str,
        on_condition: str,
        field_mapping: Dict[str, Dict[str, Any]],
        label: str,
    ) -> str:
        """Deterministic wrapper CDS DDL generator strictly enforcing #CHECK."""
        lines = [
            f"@EndUserText.label: '{label}'",
            "@AccessControl.authorizationCheck: #CHECK",
            f"define view entity {new_view_name}",
            f"  as select from {primary.name} as _Primary",
            f"  association [{clean_card}] to {secondary.name} as {sec_alias}",
            f"    on {on_condition}",
            "{",
        ]

        # Key fields first
        projected_keys = set()
        for k in primary.keys:
            if k.upper() in [rf.upper() for rf in field_mapping.keys()]:
                lines.append(f"  key _Primary.{k} as {k},")
                projected_keys.add(k.upper())
            elif len(projected_keys) == 0:
                lines.append(f"  key _Primary.{k} as {k},")
                projected_keys.add(k.upper())

        # Projections
        for rf, m in field_mapping.items():
            if rf.upper() in projected_keys:
                continue
            lines.append(f"  {m['field_expr']},")

        # Association exposure
        lines.append(f"  /* Exposed association */")
        lines.append(f"  {sec_alias}")

        # Clean trailing commas
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].endswith(","):
                lines[i] = lines[i][:-1]
                break

        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def _generate_create_ddl(
        new_view_name: str,
        primary: CandidateView,
        requested_fields: List[str],
        label: str,
    ) -> str:
        """Informational best-effort DDL with commented-out missing fields and #CHECK."""
        lines = [
            f"@EndUserText.label: '{label}'",
            "@AccessControl.authorizationCheck: #CHECK",
            f"define view entity {new_view_name}",
            f"  as select from {primary.name} as _Primary",
            "{",
        ]

        # Key fields
        projected = set()
        for k in primary.keys:
            lines.append(f"  key _Primary.{k} as {k},")
            projected.add(k.upper())

        for rf in requested_fields:
            rf_upper = rf.upper()
            if rf_upper in projected:
                continue
            if rf_upper in primary.fields:
                pf = primary.field_dict.get(rf_upper, rf)
                lines.append(f"  _Primary.{pf} as {rf},")
                projected.add(rf_upper)
            else:
                lines.append(f"  // TODO: Unresolved user field: {rf}")
                lines.append(f"  /* Field '{rf}' omitted: could not be safely mapped to the graph */")

        # Clean trailing commas
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].endswith(","):
                lines[i] = lines[i][:-1]
                break

        lines.append("}")
        return "\n".join(lines)


# Standalone alias for Pair Evaluation
evaluate_view_pair = RequirementSolver.evaluate_pair

