"""Property Invariant Tests for Enterprise SAP CDS Recommendation & Composition Agent.

Verifies the 10 core architectural invariants specified in Section 36 of the Directive:
Invariant 1: If a single candidate contains all requested fields and is compatible, result cannot be CREATE.
Invariant 2: If a safe pair exists, CREATE must not be returned.
Invariant 3: If every possible pair violates cardinality, EXTEND must not be returned.
Invariant 4: If a field exists in the catalog, reverse index discovers at least one provider.
Invariant 5: No generated DDL contains an unvalidated SAP field in projection.
Invariant 6: No automatic EXTEND uses 1..* or 0..*.
Invariant 7: No CREATE silently removes a requested field (missing fields are reported).
Invariant 8: No infrastructure failure is represented as a normal CREATE decision.
Invariant 9: LLM proposal cannot override deterministic metadata / graph safety.
Invariant 10: Adding a new CDS view to metadata does not require code changes.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.canonical_metadata import (
    CanonicalMetadataRegistry,
    CanonicalViewMetadata,
    CanonicalField,
    CanonicalAssociation,
)
from core.composition_solver import (
    CandidateView,
    RequirementAnalyzer,
    RequirementSolver,
    classify_cardinality,
)
from core.query_expander import QueryExpander


class TestArchitecturalInvariants(unittest.TestCase):
    """Rigorous property-style invariant verification."""

    def setUp(self):
        self.registry = CanonicalMetadataRegistry()
        self.expander = QueryExpander()
        self.analyzer = RequirementAnalyzer(query_expander=self.expander, registry=self.registry)
        self.solver = RequirementSolver()

    def test_invariant_1_single_candidate_complete_coverage_never_creates(self):
        """Invariant 1: If single candidate contains 100% fields and compatible, result CANNOT be CREATE."""
        cand = CandidateView(
            name="I_TESTVIEW",
            fields={"FIELD1", "FIELD2", "FIELD3"},
            keys=["FIELD1"],
            retrieval_score=0.90,
        )
        analysis = self.analyzer.analyze(
            business_requirement="Test requirement",
            requested_fields=["FIELD1", "FIELD2"],
        )
        res = self.solver.solve(analysis=analysis, candidates=[cand])
        self.assertNotEqual(res.decision, "CREATE")
        self.assertEqual(res.decision, "REUSE")

    def test_invariant_2_safe_pair_exists_never_creates(self):
        """Invariant 2: If a safe pair exists, CREATE must NOT be returned."""
        cand_prim = CandidateView(
            name="I_VIEW_A",
            fields={"KEY_A", "FIELD_A"},
            keys=["KEY_A"],
            retrieval_score=0.90,
            associations=[
                {
                    "target_view": "I_VIEW_B",
                    "alias": "_B",
                    "cardinality": "0..1",
                    "join_keys": ["KEY_A"],
                }
            ],
        )
        cand_sec = CandidateView(
            name="I_VIEW_B",
            fields={"KEY_A", "FIELD_B"},
            keys=["KEY_A"],
            retrieval_score=0.85,
        )
        analysis = self.analyzer.analyze(
            business_requirement="Test pair requirement",
            requested_fields=["FIELD_A", "FIELD_B"],
        )
        res = self.solver.solve(
            analysis=analysis,
            candidates=[cand_prim],
            secondary_candidates=[cand_sec],
        )
        self.assertNotEqual(res.decision, "CREATE")
        self.assertEqual(res.decision, "EXTEND")

    def test_invariant_3_all_pairs_violate_cardinality_never_extends(self):
        """Invariant 3: If every possible pair violates cardinality, EXTEND must NOT be returned."""
        cand_prim = CandidateView(
            name="I_HEADER",
            fields={"HEADER_ID", "DATE"},
            keys=["HEADER_ID"],
            retrieval_score=0.90,
            associations=[
                {
                    "target_view": "I_ITEM",
                    "alias": "_Item",
                    "cardinality": "1..*",  # To-many!
                    "join_keys": ["HEADER_ID"],
                }
            ],
        )
        cand_sec = CandidateView(
            name="I_ITEM",
            fields={"HEADER_ID", "ITEM_ID", "AMOUNT"},
            keys=["HEADER_ID", "ITEM_ID"],
            retrieval_score=0.85,
        )
        analysis = self.analyzer.analyze(
            business_requirement="Header with amount",
            requested_fields=["DATE", "AMOUNT"],
        )
        res = self.solver.solve(
            analysis=analysis,
            candidates=[cand_prim],
            secondary_candidates=[cand_sec],
        )
        self.assertNotEqual(res.decision, "EXTEND")
        self.assertEqual(res.decision, "CREATE")

    def test_invariant_4_catalog_field_always_discoverable_via_reverse_index(self):
        """Invariant 4: If a field exists in metadata catalog, reverse index discovers it."""
        view = CanonicalViewMetadata(
            view_name="I_ARBITRARY_NEW_VIEW",
            fields=[
                CanonicalField("SpecialFieldAlpha", "SpecialFieldAlpha"),
            ],
        )
        self.registry.register_view(view)
        providers = self.registry.get_views_for_field("SpecialFieldAlpha")
        self.assertGreater(len(providers), 0)
        self.assertIn("I_ARBITRARY_NEW_VIEW", providers)

    def test_invariant_5_no_unvalidated_sap_field_in_projected_ddl(self):
        """Invariant 5: No generated DDL contains an unvalidated SAP field in projection."""
        cand = CandidateView(
            name="I_VIEW",
            fields={"VALID_FIELD"},
            keys=["VALID_FIELD"],
            retrieval_score=0.80,
        )
        analysis = self.analyzer.analyze(
            business_requirement="Request with bogus field",
            requested_fields=["VALID_FIELD", "BogusFieldNotInRepo"],
        )
        res = self.solver.solve(analysis=analysis, candidates=[cand])
        # Unvalidated field must not be in active projection
        self.assertIn("// TODO: Unresolved user field: BogusFieldNotInRepo", res.ddl)
        self.assertNotIn("as BogusFieldNotInRepo,", res.ddl)

    def test_invariant_6_no_automatic_extend_uses_to_many(self):
        """Invariant 6: classify_cardinality strictly rejects 1..* and 0..*."""
        self.assertFalse(classify_cardinality("1..*")[0])
        self.assertFalse(classify_cardinality("0..*")[0])
        self.assertFalse(classify_cardinality("[1..*]")[0])
        self.assertFalse(classify_cardinality("[0..*]")[0])
        self.assertFalse(classify_cardinality("to_many")[0])
        self.assertTrue(classify_cardinality("0..1")[0])
        self.assertTrue(classify_cardinality("1..1")[0])

    def test_invariant_7_no_create_silently_removes_requested_field(self):
        """Invariant 7: No CREATE silently removes a requested field."""
        cand = CandidateView(
            name="I_VIEW",
            fields={"FIELD_A"},
            keys=["FIELD_A"],
            retrieval_score=0.75,
        )
        analysis = self.analyzer.analyze(
            business_requirement="Query with missing field",
            requested_fields=["FIELD_A", "FIELD_B"],
        )
        res = self.solver.solve(analysis=analysis, candidates=[cand])
        self.assertEqual(res.decision, "CREATE")
        self.assertIn("FIELD_B", res.unresolved_fields + res.missing_fields)

    def test_invariant_8_infrastructure_failure_never_normal_create(self):
        """Invariant 8: Infrastructure failures are marked with error reason codes, not normal partial match."""
        analysis = self.analyzer.analyze(
            business_requirement="Test query",
            requested_fields=["FieldA"],
        )
        res = self.solver.solve(
            analysis=analysis,
            candidates=[],
            infrastructure_error="GRAPH_VALIDATION_UNAVAILABLE",
        )
        self.assertEqual(res.reason_code, "GRAPH_VALIDATION_UNAVAILABLE")
        self.assertEqual(res.confidence, 0.0)
        self.assertFalse(res.deployable)

    def test_invariant_9_llm_cannot_override_metadata_safety(self):
        """Invariant 9: LLM proposal cannot claim REUSE or safe EXTEND without metadata proof."""
        # Query expander proposes SalesOrder + Product
        analysis = self.analyzer.analyze(
            business_requirement="Sales order items",
            requested_fields=["SalesOrder", "Product"],
        )
        # Even if user/LLM says view is I_Customer, if I_Customer doesn't have SalesOrder/Product, REUSE is impossible
        cand_cust = CandidateView(name="I_Customer", fields={"CUSTOMER"}, retrieval_score=0.99)
        res = self.solver.solve(analysis=analysis, candidates=[cand_cust])
        self.assertNotEqual(res.decision, "REUSE")
        self.assertEqual(res.decision, "CREATE")

    def test_invariant_10_dynamic_view_addition_requires_zero_code_changes(self):
        """Invariant 10: Adding arbitrary new CDS view to metadata works without any Python code modifications."""
        # Create a brand new custom view for QM (Quality Management)
        qm_view_dict = {
            "view_name": "I_QUALITYINSPECTIONLOT",
            "description": "Quality Inspection Lot View",
            "domain": "QM",
            "fields": [
                {"name": "InspectionLot", "is_key": True},
                {"name": "InspectionLotStatus", "is_key": False},
                {"name": "InspectionQuantity", "is_measure": True},
            ],
            "associations": [],
        }

        # Register dynamically in canonical registry
        v_meta = self.registry.register_view(qm_view_dict)
        self.assertEqual(v_meta.view_name, "I_QUALITYINSPECTIONLOT")

        # Discoverable immediately via reverse index
        providers = self.registry.get_views_for_field("InspectionLotStatus")
        self.assertIn("I_QUALITYINSPECTIONLOT", providers)

        # Evaluates immediately to REUSE with 100% field coverage
        cand_qm = CandidateView.from_metadata(v_meta.to_dict(), retrieval_score=0.95)
        analysis = self.analyzer.analyze(
            business_requirement="Inspection lots with status",
            requested_fields=["InspectionLot", "InspectionLotStatus"],
            domain="QM",
        )
        res = self.solver.solve(analysis=analysis, candidates=[cand_qm])
        self.assertEqual(res.decision, "REUSE")
        self.assertEqual(res.primary_view_name, "I_QUALITYINSPECTIONLOT")


if __name__ == "__main__":
    unittest.main()
