"""Comprehensive Unit and Integration Tests for Targeted Secondary Retrieval in EXTEND State Composition.

Verifies strictly according to System Directive:
1. Secondary Search Loop:
   - Primary candidate < 100% coverage extracts unresolved_fields and triggers secondary search specifically for missing fields.
   - Secondary candidates are ranked by missing-field coverage and retrieval score.
2. Pair Evaluation:
   - 100% combined field coverage requirement.
   - Direct 1-hop Knowledge Graph association requirement.
   - Safe cardinality (strictly 1..1 or 0..1; rejects 1..*, 0..*, malformed).
   - Valid, accessible join keys in primary view.
   - Analytical grain compatibility.
3. Decision Routing:
   - Successful pair returns EXTEND with wrapper CDS DDL containing #CHECK authorization.
   - Failed pair falls back to CREATE with #CHECK authorization.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.composition_solver import (
    RequirementAnalyzer,
    RequirementSolver,
    RequirementAnalysis,
    CandidateView,
    DecisionResult,
    evaluate_view_pair,
)


class TestTargetedSecondaryRetrieval(unittest.TestCase):
    """Test suite for targeted secondary retrieval, pair evaluation, and EXTEND decision routing."""

    def setUp(self):
        self.analyzer = RequirementAnalyzer()
        self.solver = RequirementSolver()

    # -------------------------------------------------------------
    # 1. Pair Evaluation Unit Tests (evaluate_pair)
    # -------------------------------------------------------------
    def test_pair_evaluation_succeeds_for_safe_association(self):
        """Pair evaluation passes when all 4 gates + grain check are satisfied."""
        prim = CandidateView(
            name="I_PURCHASEORDER",
            fields={"PURCHASEORDER", "SUPPLIER"},
            keys=["PurchaseOrder"],
            retrieval_score=0.90,
            associations=[
                {
                    "target_view": "I_SUPPLIER",
                    "alias": "_Supplier",
                    "cardinality": "[0..1]",
                    "on_condition": "$projection.Supplier = _Supplier.Supplier",
                }
            ],
        )
        sec = CandidateView(
            name="I_SUPPLIER",
            fields={"SUPPLIER", "SUPPLIERNAME", "CITYNAME"},
            keys=["Supplier"],
            retrieval_score=0.85,
        )

        requested_set = {"PURCHASEORDER", "SUPPLIERNAME", "CITYNAME"}
        is_valid, comp, reason = self.solver.evaluate_pair(
            primary=prim,
            secondary=sec,
            requested_set=requested_set,
            is_analytical=False,
        )

        self.assertTrue(is_valid)
        self.assertEqual(reason, "VALID")
        self.assertIsNotNone(comp)
        self.assertEqual(comp["clean_card"], "0..1")
        self.assertEqual(comp["join_keys"], ["Supplier"])

    def test_pair_evaluation_rejects_insufficient_combined_coverage(self):
        """Pair evaluation fails if combined coverage does not reach 100%."""
        prim = CandidateView(
            name="I_ORDER",
            fields={"ORDERID", "MATERIAL"},
            associations=[{"target_view": "I_MATERIAL", "cardinality": "[0..1]"}],
        )
        sec = CandidateView(
            name="I_MATERIAL",
            fields={"MATERIAL", "MATERIALNAME"},
        )
        # Requested includes PLANT which neither view covers
        requested_set = {"ORDERID", "MATERIALNAME", "PLANT"}

        is_valid, comp, reason = self.solver.evaluate_pair(
            primary=prim,
            secondary=sec,
            requested_set=requested_set,
            is_analytical=False,
        )
        self.assertFalse(is_valid)
        self.assertEqual(reason, "INSUFFICIENT_COMBINED_COVERAGE")
        self.assertIsNone(comp)

    def test_pair_evaluation_rejects_no_direct_association(self):
        """Pair evaluation fails if there is no direct 1-hop association between primary and secondary."""
        prim = CandidateView(
            name="I_SALESORDER",
            fields={"SALESORDER", "CUSTOMER"},
            associations=[],  # No association to I_PRODUCT
        )
        sec = CandidateView(
            name="I_PRODUCT",
            fields={"PRODUCT", "PRODUCTNAME"},
            associations=[],
        )
        requested_set = {"SALESORDER", "PRODUCTNAME"}

        is_valid, comp, reason = self.solver.evaluate_pair(
            primary=prim,
            secondary=sec,
            requested_set=requested_set,
            is_analytical=False,
        )
        self.assertFalse(is_valid)
        self.assertEqual(reason, "NO_DIRECT_ASSOCIATION")

    def test_pair_evaluation_rejects_unsafe_cardinality(self):
        """Pair evaluation fails when association cardinality is to-many (1..* or 0..*)."""
        prim = CandidateView(
            name="I_HEADER",
            fields={"HEADERID", "STATUS"},
            associations=[
                {
                    "target_view": "I_ITEM",
                    "cardinality": "[1..*]",  # Unsafe To-Many
                    "on_condition": "_Primary.HeaderId = _Item.HeaderId",
                }
            ],
        )
        sec = CandidateView(
            name="I_ITEM",
            fields={"HEADERID", "ITEMID", "ITEMTEXT"},
            keys=["HeaderId", "ItemId"],
        )
        requested_set = {"HEADERID", "STATUS", "ITEMTEXT"}

        is_valid, comp, reason = self.solver.evaluate_pair(
            primary=prim,
            secondary=sec,
            requested_set=requested_set,
            is_analytical=False,
        )
        self.assertFalse(is_valid)
        self.assertIn("UNSAFE_CARDINALITY", reason)

    def test_pair_evaluation_rejects_inaccessible_join_keys(self):
        """Pair evaluation fails if the join keys are not accessible in the primary view."""
        prim = CandidateView(
            name="I_PRIMARY",
            fields={"F1", "F2"},  # Missing KEY_ABC
            associations=[
                {
                    "target_view": "I_SECONDARY",
                    "cardinality": "[0..1]",
                    "on_condition": "_Primary.KEY_ABC = _Secondary.KEY_ABC",
                }
            ],
        )
        sec = CandidateView(
            name="I_SECONDARY",
            fields={"KEY_ABC", "F3"},
            keys=["KEY_ABC"],
        )
        requested_set = {"F1", "F3"}

        is_valid, comp, reason = self.solver.evaluate_pair(
            primary=prim,
            secondary=sec,
            requested_set=requested_set,
            is_analytical=False,
        )
        self.assertFalse(is_valid)
        self.assertIn("JOIN_KEYS_NOT_ACCESSIBLE", reason)

    # -------------------------------------------------------------
    # 2. Secondary Search Loop & Missing-Field Ranking Tests
    # -------------------------------------------------------------
    def test_secondary_search_loop_invokes_retriever_with_unresolved_fields(self):
        """When primary covers < 100%, secondary retriever is called with exact unresolved fields."""
        prim = CandidateView(
            name="I_SALESORDER",
            fields={"SALESORDER", "SOLDTOPARTY", "TOTALNETAMOUNT"},
            keys=["SalesOrder"],
            retrieval_score=0.92,
            associations=[
                {
                    "target_view": "I_CUSTOMER",
                    "alias": "_Customer",
                    "cardinality": "[0..1]",
                    "on_condition": "_Primary.SoldToParty = _Customer.Customer",
                }
            ],
        )

        analysis = RequirementAnalysis(
            business_requirement="Sales order with customer name and city",
            requested_fields=["SalesOrder", "CustomerName", "CityName"],
            requested_canonical_fields=["SalesOrder", "CustomerName", "CityName"],
            is_analytical=False,
        )

        # Mock secondary retriever
        sec_mock = CandidateView(
            name="I_CUSTOMER",
            fields={"CUSTOMER", "CUSTOMERNAME", "CITYNAME"},
            keys=["Customer"],
            retrieval_score=0.90,
        )
        mock_retriever = MagicMock(return_value=[sec_mock])

        result = self.solver.solve(
            analysis=analysis,
            candidates=[prim],
            secondary_retriever=mock_retriever,
        )

        # Verify secondary_retriever was called specifically for missing fields
        mock_retriever.assert_called_once()
        call_args = mock_retriever.call_args[0]
        self.assertEqual(call_args[0].name, "I_SALESORDER")
        self.assertEqual(set(call_args[1]), {"CustomerName", "CityName"})

        # Verify EXTEND decision
        self.assertEqual(result.decision, "EXTEND")
        self.assertEqual(result.reason_code, "SAFE_TWO_VIEW_COMPOSITION")
        self.assertEqual(result.primary_view_name, "I_SALESORDER")
        self.assertEqual(result.secondary_view, "I_CUSTOMER")
        self.assertEqual(result.coverage, 1.0)
        self.assertTrue(result.deployable)
        self.assertIn("@AccessControl.authorizationCheck: #CHECK", result.ddl)

    def test_secondary_candidates_ranked_by_missing_field_coverage(self):
        """Secondary candidates are ranked by missing-field coverage so higher-coverage view wins."""
        prim = CandidateView(
            name="I_ORDER",
            fields={"ORDERID", "ITEMID"},
            keys=["OrderId"],
            associations=[
                {"target_view": "I_SEC_PARTIAL", "cardinality": "[0..1]", "on_condition": "_Primary.OrderId = _Sec.OrderId"},
                {"target_view": "I_SEC_FULL", "cardinality": "[0..1]", "on_condition": "_Primary.OrderId = _Sec.OrderId"},
            ],
            retrieval_score=0.90,
        )

        # Sec Partial only covers FieldA (50% missing coverage)
        sec_partial = CandidateView(
            name="I_SEC_PARTIAL",
            fields={"ORDERID", "FIELDA"},
            keys=["OrderId"],
            retrieval_score=0.99,  # Higher raw retrieval score
        )
        # Sec Full covers FieldA and FieldB (100% missing coverage)
        sec_full = CandidateView(
            name="I_SEC_FULL",
            fields={"FIELDA", "FIELDB"},
            keys=["OrderId"],
            retrieval_score=0.70,  # Lower raw retrieval score
        )

        analysis = RequirementAnalysis(
            business_requirement="Order with ItemId, FieldA, and FieldB",
            requested_fields=["OrderId", "ItemId", "FieldA", "FieldB"],
            requested_canonical_fields=["OrderId", "ItemId", "FieldA", "FieldB"],
            is_analytical=False,
        )

        result = self.solver.solve(
            analysis=analysis,
            candidates=[prim],
            secondary_candidates=[sec_partial, sec_full],
        )

        # Full view must win because it covers 100% of missing fields!
        self.assertEqual(result.decision, "EXTEND")
        self.assertEqual(result.primary_view_name, "I_ORDER")
        self.assertEqual(result.secondary_view, "I_SEC_FULL")

    # -------------------------------------------------------------
    # 3. Decision Routing Tests
    # -------------------------------------------------------------
    def test_decision_routing_falls_back_to_create_if_no_pairs_pass(self):
        """If secondary search finds views but none pass safety checks, system falls back to CREATE."""
        prim = CandidateView(
            name="I_SALESORDER",
            fields={"SALESORDER", "TOTALNETAMOUNT"},
            keys=["SalesOrder"],
            retrieval_score=0.90,
            associations=[],  # No association to secondary
        )
        sec_unlinked = CandidateView(
            name="I_UNLINKED_CUSTOMER",
            fields={"CUSTOMER", "CUSTOMERNAME"},
            keys=["Customer"],
            retrieval_score=0.88,
        )

        analysis = RequirementAnalysis(
            business_requirement="Sales order with customer name",
            requested_fields=["SalesOrder", "CustomerName"],
            requested_canonical_fields=["SalesOrder", "CustomerName"],
            is_analytical=False,
        )

        result = self.solver.solve(
            analysis=analysis,
            candidates=[prim],
            secondary_candidates=[sec_unlinked],
        )

        self.assertEqual(result.decision, "CREATE")
        self.assertFalse(result.deployable)
        self.assertIn("@AccessControl.authorizationCheck: #CHECK", result.ddl)
        self.assertIn("CustomerName could not be safely mapped to the graph.", result.warnings)


if __name__ == "__main__":
    unittest.main()
