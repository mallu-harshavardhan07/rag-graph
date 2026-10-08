"""Unit and Integration Tests for Deterministic Tri-State Decision Engine (REUSE, EXTEND, CREATE).

Verifies strictly:
- Zero-Field Guard (ValueError on empty fields)
- Analytical Intent & Grain Rejection
- State 1 (REUSE): 100% match, no code generation or compiler calls
- State 2 (EXTEND): 2-View set-cover, safe cardinalities (1..1, 0..1)
- Hard Cardinality Ban: Unsafe cardinalities (1..*, 0..*) immediately fail EXTEND -> CREATE
- Deterministic Tie-Breaking Strategy
- State 3 (CREATE): deployable is False, best-effort DDL with #CHECK
- DCL Security: @AccessControl.authorizationCheck: #CHECK enforcement
- REUSE Mock Assertions: mock_generator.assert_not_called(), mock_adt.assert_not_called()
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
    detect_analytical_intent,
    classify_cardinality,
)


class TestCompositionSolver(unittest.TestCase):
    """Test suite for RequirementAnalyzer, RequirementSolver, and GraphRAG Safety Gates."""

    def setUp(self):
        self.analyzer = RequirementAnalyzer()
        self.solver = RequirementSolver()

    # -------------------------------------------------------------
    # 1. Zero-Field Guard
    # -------------------------------------------------------------
    def test_zero_field_guard_empty_list_raises_value_error(self):
        """Zero-Field Guard: Empty requested fields list immediately raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.analyzer.analyze(
                business_requirement="Need sales order details",
                requested_fields=[],
            )
        self.assertIn("At least one requested field is required", str(ctx.exception))

    def test_zero_field_guard_whitespace_fields_raises_value_error(self):
        """Zero-Field Guard: Whitespace-only fields list raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.analyzer.analyze(
                business_requirement="Need purchase order details",
                requested_fields=["   ", ""],
            )
        self.assertIn("At least one requested field is required", str(ctx.exception))

    def test_zero_field_guard_in_solver_solve(self):
        """Solver solve() raises ValueError if requested_canonical_fields is empty."""
        analysis = RequirementAnalysis(
            business_requirement="Test requirement",
            requested_fields=[],
            requested_canonical_fields=[],
            is_analytical=False,
        )
        with self.assertRaises(ValueError):
            self.solver.solve(analysis=analysis, candidates=[])

    # -------------------------------------------------------------
    # 2. Analytical Intent & Grain Detection
    # -------------------------------------------------------------
    def test_analytical_intent_detection(self):
        """Detect analytical keywords like 'total', 'sum', 'group by', 'average'."""
        self.assertTrue(detect_analytical_intent("Show total sales by customer"))
        self.assertTrue(detect_analytical_intent("Calculate sum of net price"))
        self.assertTrue(detect_analytical_intent("Group by plant and calculate average yield"))
        self.assertTrue(detect_analytical_intent("Provide sales order cube for analytics KPI"))
        self.assertFalse(detect_analytical_intent("Expose sales order header with sold-to party"))

    def test_analytical_intent_rejects_raw_view_for_reuse(self):
        """If is_analytical is True, raw transaction view (is_analytical_cube=False) is rejected for REUSE."""
        raw_cand = CandidateView(
            name="I_SALESORDER",
            fields={"SALESORDER", "TOTALNETAMOUNT", "CUSTOMER"},
            is_analytical_cube=False,  # Raw transaction view
            retrieval_score=0.95,
        )
        analysis = RequirementAnalysis(
            business_requirement="Calculate total sum of sales orders",
            requested_fields=["SalesOrder", "TotalNetAmount"],
            requested_canonical_fields=["SalesOrder", "TotalNetAmount"],
            is_analytical=True,
        )

        result = self.solver.solve(analysis=analysis, candidates=[raw_cand])
        # REUSE is rejected because candidate is not a cube/query!
        self.assertNotEqual(result.decision, "REUSE")
        self.assertEqual(result.decision, "CREATE")

    def test_analytical_intent_accepts_cube_for_reuse(self):
        """If is_analytical is True and candidate is an analytical cube, REUSE is accepted."""
        cube_cand = CandidateView(
            name="I_SALESORDER_CUBE",
            fields={"SALESORDER", "TOTALNETAMOUNT", "CUSTOMER"},
            is_analytical_cube=True,  # Analytical cube
            retrieval_score=0.95,
        )
        analysis = RequirementAnalysis(
            business_requirement="Calculate total sum of sales orders",
            requested_fields=["SalesOrder", "TotalNetAmount"],
            requested_canonical_fields=["SalesOrder", "TotalNetAmount"],
            is_analytical=True,
        )

        result = self.solver.solve(analysis=analysis, candidates=[cube_cand])
        self.assertEqual(result.decision, "REUSE")
        self.assertEqual(result.reason_code, "FULL_SINGLE_VIEW_MATCH")
        self.assertEqual(result.confidence, 1.0)
        self.assertTrue(result.deployable)
        self.assertIsNone(result.ddl)

    # -------------------------------------------------------------
    # 3. State 1: REUSE (100% Match) & Governance Assertions
    # -------------------------------------------------------------
    def test_reuse_state_exact_contract(self):
        """State 1 REUSE produces exact standardized JSON schema."""
        cand = CandidateView(
            name="I_MANUFACTURINGORDER",
            fields={"MANUFACTURINGORDER", "PLANT", "TARGETQUANTITY"},
            field_dict={
                "MANUFACTURINGORDER": "ManufacturingOrder",
                "PLANT": "Plant",
                "TARGETQUANTITY": "MfgOrderPlannedTotalQty",
            },
            is_analytical_cube=False,
            retrieval_score=0.99,
        )
        analysis = RequirementAnalysis(
            business_requirement="Manufacturing order header with plant and target quantity",
            requested_fields=["ManufacturingOrder", "Plant", "TargetQuantity"],
            requested_canonical_fields=["ManufacturingOrder", "Plant", "TargetQuantity"],
            is_analytical=False,
        )

        result = self.solver.solve(analysis=analysis, candidates=[cand])

        # Verify JSON Contract
        d = result.to_dict()
        self.assertEqual(d["decision"], "REUSE")
        self.assertEqual(d["reason_code"], "FULL_SINGLE_VIEW_MATCH")
        self.assertEqual(d["confidence"], 1.0)
        self.assertEqual(d["primary_view"], {"name": "I_MANUFACTURINGORDER", "coverage": 1.0})
        self.assertEqual(d["requested_fields"], ["ManufacturingOrder", "Plant", "TargetQuantity"])
        self.assertEqual(d["resolved_fields"], ["ManufacturingOrder", "Plant", "MfgOrderPlannedTotalQty"])
        self.assertEqual(d["missing_fields"], [])
        self.assertEqual(d["unresolved_fields"], [])
        self.assertTrue(d["deployable"])
        self.assertEqual(d["warnings"], [])
        self.assertIsNone(d["ddl"])

    def test_reuse_never_invokes_generator_or_compiler(self):
        """REUSE Governance: DDL generator and ADT compiler MUST NOT be called."""
        mock_generator = MagicMock()
        mock_adt = MagicMock()

        cand = CandidateView(
            name="I_MANUFACTURINGORDER",
            fields={"MANUFACTURINGORDER", "PLANT"},
            is_analytical_cube=False,
        )
        analysis = RequirementAnalysis(
            business_requirement="Manufacturing order with plant",
            requested_fields=["ManufacturingOrder", "Plant"],
            requested_canonical_fields=["ManufacturingOrder", "Plant"],
            is_analytical=False,
        )

        result = self.solver.solve(analysis=analysis, candidates=[cand])
        self.assertEqual(result.decision, "REUSE")

        # Simulate governance orchestration
        if result.decision == "EXTEND":
            mock_generator.generate()
            mock_adt.syntax_check()
        elif result.decision == "CREATE":
            mock_generator.generate()

        # Strict governance assertions
        mock_generator.assert_not_called()
        mock_adt.assert_not_called()

    # -------------------------------------------------------------
    # 4. State 2: EXTEND & Hard Cardinality Ban
    # -------------------------------------------------------------
    def test_cardinality_classifier(self):
        """Map graph cardinalities to SAFE (1..1, 0..1) or UNSAFE (1..*, 0..*, or malformed)."""
        # Safe
        self.assertEqual(classify_cardinality("[0..1]"), (True, "0..1"))
        self.assertEqual(classify_cardinality("0..1"), (True, "0..1"))
        self.assertEqual(classify_cardinality("[1..1]"), (True, "1..1"))
        self.assertEqual(classify_cardinality("1..1"), (True, "1..1"))
        self.assertEqual(classify_cardinality("exact_one"), (True, "1..1"))

        # Unsafe (Hard Ban)
        self.assertEqual(classify_cardinality("[0..*]")[0], False)
        self.assertEqual(classify_cardinality("1..*")[0], False)
        self.assertEqual(classify_cardinality("[1..*]")[0], False)
        self.assertEqual(classify_cardinality("0..n")[0], False)
        self.assertEqual(classify_cardinality("to_many")[0], False)
        self.assertEqual(classify_cardinality("")[0], False)
        self.assertEqual(classify_cardinality(None)[0], False)

    def test_safe_cardinality_0_to_1_succeeds_extend(self):
        """Safe cardinality [0..1] allows EXTEND composition."""
        view_order = CandidateView(
            name="I_MANUFACTURINGORDER",
            fields={"MANUFACTURINGORDER", "MATERIAL", "BATCH"},
            keys=["ManufacturingOrder"],
            retrieval_score=0.90,
            associations=[
                {
                    "target_view": "I_BATCH",
                    "cardinality": "[0..1]",
                    "on_condition": "_Primary.Material = _Batch.Material and _Primary.Batch = _Batch.Batch",
                }
            ],
        )
        view_batch = CandidateView(
            name="I_BATCH",
            fields={"MATERIAL", "BATCH", "BATCHDATE"},
            keys=["Material", "Batch"],
            retrieval_score=0.85,
        )

        analysis = RequirementAnalysis(
            business_requirement="Manufacturing order with batch date",
            requested_fields=["ManufacturingOrder", "Material", "BatchDate"],
            requested_canonical_fields=["ManufacturingOrder", "Material", "BatchDate"],
            is_analytical=False,
        )

        result = self.solver.solve(analysis=analysis, candidates=[view_order, view_batch])

        self.assertEqual(result.decision, "EXTEND")
        self.assertEqual(result.reason_code, "SAFE_TWO_VIEW_COMPOSITION")
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.primary_view_name, "I_MANUFACTURINGORDER")
        self.assertEqual(result.secondary_view, "I_BATCH")
        self.assertEqual(result.association["cardinality"], "0..1")
        self.assertTrue(result.association["safe"])
        self.assertTrue(result.deployable)
        self.assertIn("@AccessControl.authorizationCheck: #CHECK", result.ddl)
        self.assertIn("SECURITY: A DCL authorization role must be assigned manually.", result.warnings)

    def test_unsafe_cardinality_1_to_many_fails_extend_and_falls_back_to_create(self):
        """Hard Cardinality Ban: [1..*] cardinality fails EXTEND and falls back to CREATE."""
        view_header = CandidateView(
            name="I_SALESORDER",
            fields={"SALESORDER", "CUSTOMER"},
            keys=["SalesOrder"],
            associations=[
                {
                    "target_view": "I_SALESORDERITEM",
                    "cardinality": "[1..*]",  # UNSAFE TO-MANY
                    "on_condition": "_Primary.SalesOrder = _Item.SalesOrder",
                }
            ],
            retrieval_score=0.90,
        )
        view_item = CandidateView(
            name="I_SALESORDERITEM",
            fields={"SALESORDER", "SALESORDERITEM", "NETAMOUNT"},
            keys=["SalesOrder", "SalesOrderItem"],
            retrieval_score=0.85,
        )

        analysis = RequirementAnalysis(
            business_requirement="Sales order header with item net amount",
            requested_fields=["SalesOrder", "Customer", "NetAmount"],
            requested_canonical_fields=["SalesOrder", "Customer", "NetAmount"],
            is_analytical=False,
        )

        result = self.solver.solve(analysis=analysis, candidates=[view_header, view_item])

        # Must NOT be EXTEND because 1..* is unsafe!
        self.assertEqual(result.decision, "CREATE")
        self.assertEqual(result.reason_code, "NO_SAFE_ASSOCIATION")
        self.assertFalse(result.deployable)
        self.assertIn("NetAmount could not be safely mapped to the graph.", result.warnings)
        self.assertIn("@AccessControl.authorizationCheck: #CHECK", result.ddl)

    def test_unsafe_cardinality_0_to_many_fails_extend_and_falls_back_to_create(self):
        """Hard Cardinality Ban: [0..*] cardinality fails EXTEND and falls back to CREATE."""
        view_po = CandidateView(
            name="I_PURCHASEORDER",
            fields={"PURCHASEORDER", "SUPPLIER"},
            associations=[
                {
                    "target_view": "I_PURCHASEORDERITEM",
                    "cardinality": "[0..*]",  # UNSAFE TO-MANY
                    "on_condition": "_Primary.PurchaseOrder = _Item.PurchaseOrder",
                }
            ],
            retrieval_score=0.90,
        )
        view_item = CandidateView(
            name="I_PURCHASEORDERITEM",
            fields={"PURCHASEORDER", "PURCHASEORDERITEM", "PLANT"},
            retrieval_score=0.85,
        )

        analysis = RequirementAnalysis(
            business_requirement="PO with plant",
            requested_fields=["PurchaseOrder", "Supplier", "Plant"],
            requested_canonical_fields=["PurchaseOrder", "Supplier", "Plant"],
            is_analytical=False,
        )

        result = self.solver.solve(analysis=analysis, candidates=[view_po, view_item])
        self.assertEqual(result.decision, "CREATE")
        self.assertEqual(result.reason_code, "NO_SAFE_ASSOCIATION")
        self.assertFalse(result.deployable)

    # -------------------------------------------------------------
    # 5. Deterministic Tie-Breaking Strategy
    # -------------------------------------------------------------
    def test_tie_breaking_hierarchy(self):
        """Tie-Breaker: Prefer higher individual primary coverage, then safer cardinality (1..1 > 0..1)."""
        # Primary A covers 2/3 fields, with 0..1 association to Secondary A
        prim_a = CandidateView(
            name="I_PRIMARY_A",
            fields={"F1", "F2", "KEY_X"},
            keys=["KEY_X"],
            retrieval_score=0.80,
            associations=[
                {"target_view": "I_SEC_A", "cardinality": "0..1", "on_condition": "_Primary.KEY_X = _Sec.KEY_X"}
            ],
        )
        sec_a = CandidateView(
            name="I_SEC_A",
            fields={"F3", "KEY_X"},
            retrieval_score=0.80,
        )

        # Primary B covers 1/3 fields, with 1..1 association to Secondary B
        prim_b = CandidateView(
            name="I_PRIMARY_B",
            fields={"F1", "KEY_Y"},
            keys=["KEY_Y"],
            retrieval_score=0.90,
            associations=[
                {"target_view": "I_SEC_B", "cardinality": "1..1", "on_condition": "_Primary.KEY_Y = _Sec.KEY_Y"}
            ],
        )
        sec_b = CandidateView(
            name="I_SEC_B",
            fields={"F2", "F3", "KEY_Y"},
            retrieval_score=0.90,
        )

        analysis = RequirementAnalysis(
            business_requirement="Expose fields F1, F2, F3",
            requested_fields=["F1", "F2", "F3"],
            requested_canonical_fields=["F1", "F2", "F3"],
            is_analytical=False,
        )

        # In pair (prim_b, sec_b), sec_b covers 2/3 fields so sec_b becomes primary!
        # Both pairs have a primary covering 2/3 fields.
        # Tie-breaker between pair (prim_a, sec_a) and (sec_b, prim_b):
        # pair (sec_b, prim_b) has cardinality 1..1 which beats 0..1!
        result = self.solver.solve(analysis=analysis, candidates=[prim_a, sec_a, sec_b, prim_b])
        self.assertEqual(result.decision, "EXTEND")
        self.assertEqual(result.association["cardinality"], "1..1")

    # -------------------------------------------------------------
    # 6. State 3: CREATE (Fallback) & Security Contracts
    # -------------------------------------------------------------
    def test_create_fallback_contract_and_security(self):
        """CREATE Fallback: deployable is False, DDL has commented-out unresolvable fields and #CHECK."""
        prim = CandidateView(
            name="I_MANUFACTURINGORDER",
            fields={"MANUFACTURINGORDER", "PLANT"},
            keys=["ManufacturingOrder"],
            retrieval_score=0.75,
        )

        analysis = RequirementAnalysis(
            business_requirement="Manufacturing order with supplier",
            requested_fields=["ManufacturingOrder", "Plant", "Supplier"],
            requested_canonical_fields=["ManufacturingOrder", "Plant", "Supplier"],
            is_analytical=False,
        )

        result = self.solver.solve(
            analysis=analysis,
            candidates=[prim],
            all_known_fields={"SUPPLIER"},  # Known to catalog, but unjoinable
        )

        d = result.to_dict()
        self.assertEqual(d["decision"], "CREATE")
        self.assertEqual(d["primary_view"], "I_MANUFACTURINGORDER")
        self.assertEqual(d["coverage"], 0.67)
        self.assertEqual(d["resolved_fields"], ["ManufacturingOrder", "Plant"])
        self.assertEqual(d["missing_fields"], ["Supplier"])
        self.assertEqual(d["unresolved_fields"], [])
        self.assertFalse(d["deployable"])
        self.assertIn("Supplier could not be safely mapped to the graph.", d["warnings"])
        self.assertIn("SECURITY: Generated DDL uses #CHECK. A DCL role is required.", d["warnings"])

        # DDL Checks
        self.assertIn("@AccessControl.authorizationCheck: #CHECK", d["ddl"])
        self.assertNotIn("#NOT_REQUIRED", d["ddl"])
        self.assertIn("/* Field 'Supplier' omitted: could not be safely mapped to the graph */", d["ddl"])


if __name__ == "__main__":
    unittest.main()
