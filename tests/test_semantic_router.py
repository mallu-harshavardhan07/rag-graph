"""Unit Tests for Semantic Router & Metadata-Aware Reranker.

Verifies:
1. Extraction of SAP Intent (ANALYTICAL, TRANSACTIONAL, MASTER_DATA).
2. Separation of Dimensions from Quantitative Measures.
3. Detection of Aggregations (SUM, COUNT, AVG, etc.).
4. Strict JSON API Output Contract conforming to SemanticRouterOutput schema.
5. Soft Boosting Reranker:
   - Proving an ANALYTICAL request for 'Total Amount by Customer' ranks I_SalesOrderItemCube
     higher than I_Customer, even if I_Customer has higher raw vector similarity.
   - Transactional views boost (#FACT).
   - Master data views boost (#DIMENSION, #TEXT).
6. Integration with Tri-State Solver: router dimensions and measures used as canonical fields,
   and intent == 'ANALYTICAL' enforces the grain check.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.query_expander import QueryExpander, QueryExpansion, SemanticRouterOutput
from core.composition_solver import (
    CandidateView,
    RequirementAnalyzer,
    RequirementSolver,
    RequirementAnalysis,
)
from core.recommender import rerank_candidates


class TestSemanticRouterAndReranking(unittest.TestCase):
    """Test suite for Semantic Router intent extraction and metadata-aware soft boosting."""

    def setUp(self):
        self.expander = QueryExpander()
        self.analyzer = RequirementAnalyzer(query_expander=self.expander)
        self.solver = RequirementSolver()

    # -------------------------------------------------------------
    # 1. Output Schema Validation
    # -------------------------------------------------------------
    def test_semantic_router_output_schema(self):
        """Semantic Router outputs strictly valid schema matching contract."""
        res = self.expander.expand(
            business_requirement="Total Net Amount by Customer in SD",
            requested_fields=["Customer", "TotalAmount"],
        )

        d = res.to_dict()
        self.assertIn("intent", d)
        self.assertIn("dimensions", d)
        self.assertIn("measures", d)
        self.assertIn("aggregations", d)
        self.assertIn("likely_domains", d)
        self.assertIn("search_keywords", d)

        self.assertIn(d["intent"], ["ANALYTICAL", "TRANSACTIONAL", "MASTER_DATA"])
        self.assertIsInstance(d["dimensions"], list)
        self.assertIsInstance(d["measures"], list)
        self.assertIsInstance(d["aggregations"], list)
        self.assertIsInstance(d["likely_domains"], list)
        self.assertIsInstance(d["search_keywords"], str)

    # -------------------------------------------------------------
    # 2. Intent Classification: ANALYTICAL, TRANSACTIONAL, MASTER_DATA
    # -------------------------------------------------------------
    def test_analytical_intent_classification(self):
        """Analytical queries with totals/aggregations classify as ANALYTICAL."""
        res = self.expander.expand(
            business_requirement="Total Amount by Customer",
            requested_fields=["Customer", "TotalAmount"],
        )

        self.assertEqual(res.intent, "ANALYTICAL")
        self.assertIn("Customer", res.dimensions)
        self.assertIn("TotalNetAmount", res.measures)
        self.assertIn("SUM", res.aggregations)
        self.assertIn("SD", res.likely_domains)

    def test_transactional_intent_classification(self):
        """Transactional queries with line items classify as TRANSACTIONAL."""
        res = self.expander.expand(
            business_requirement="Sales order line items with net price and product",
            requested_fields=["SalesOrder", "SalesOrderItem", "Product", "NetPrice"],
        )

        self.assertEqual(res.intent, "TRANSACTIONAL")
        self.assertIn("SalesOrder", res.dimensions)
        self.assertIn("Product", res.dimensions)
        self.assertIn("NetPriceAmount", res.measures)
        self.assertIn("SD", res.likely_domains)

    def test_master_data_intent_classification(self):
        """Pure master data queries classify as MASTER_DATA."""
        res = self.expander.expand(
            business_requirement="Customer master data with customer name and city",
            requested_fields=["Customer", "CustomerName", "CityName"],
        )

        self.assertEqual(res.intent, "MASTER_DATA")
        self.assertIn("Customer", res.dimensions)
        self.assertIn("CustomerName", res.dimensions)
        self.assertIn("CityName", res.dimensions)
        self.assertEqual(len(res.measures), 0)
        self.assertIn("SD", res.likely_domains)

    # -------------------------------------------------------------
    # 3. Soft Boosting & Metadata-Aware Reranker (Testing Criteria)
    # -------------------------------------------------------------
    def test_metadata_aware_reranking_analytical_boost(self):
        """CRITICAL REQUIREMENT: An ANALYTICAL request for 'Total Amount by Customer'

        retrieves and ranks I_SalesOrderItemCube higher than I_Customer,
        even though I_Customer starts with higher raw vector similarity.
        """
        # I_Customer has higher initial raw similarity (0.95 vs 0.75) because query has 'Customer'
        cand_customer = CandidateView(
            name="I_CUSTOMER",
            fields={"CUSTOMER", "CUSTOMERNAME", "CITYNAME"},
            retrieval_score=0.95,
            data_category="#DIMENSION",
            is_analytical_cube=False,
        )

        # I_SalesOrderItemCube has lower initial raw similarity (0.75)
        cand_cube = CandidateView(
            name="I_SALESORDERITEMCUBE",
            fields={"CUSTOMER", "TOTALNETAMOUNT", "SALESORDER"},
            retrieval_score=0.75,
            data_category="#CUBE",
            is_analytical_cube=True,
        )

        # Initial order before reranking: I_CUSTOMER (0.95) > I_SALESORDERITEMCUBE (0.75)
        initial_candidates = [cand_customer, cand_cube]
        self.assertEqual(initial_candidates[0].name, "I_CUSTOMER")

        # Apply soft boosting with ANALYTICAL intent
        reranked = rerank_candidates(initial_candidates, intent="ANALYTICAL")

        # Reranked: I_SALESORDERITEMCUBE gets +0.30 boost + 0.10 Core boost (0.75 + 0.30 + 0.10 = 1.15),
        # beating I_CUSTOMER (0.95 + 0.10 = 1.05)!
        self.assertEqual(reranked[0].name, "I_SALESORDERITEMCUBE")
        self.assertGreater(reranked[0].retrieval_score, reranked[1].retrieval_score)
        self.assertEqual(reranked[0].retrieval_score, 1.15)
        self.assertEqual(reranked[1].retrieval_score, 1.05)

    def test_metadata_aware_reranking_master_data_boost(self):
        """MASTER_DATA intent boosts #DIMENSION / #TEXT views."""
        cand_customer = CandidateView(
            name="I_CUSTOMER",
            fields={"CUSTOMER", "CUSTOMERNAME"},
            retrieval_score=0.80,
            data_category="#DIMENSION",
        )
        cand_cube = CandidateView(
            name="I_SALESORDERITEMCUBE",
            fields={"CUSTOMER", "TOTALNETAMOUNT"},
            retrieval_score=0.85,
            data_category="#CUBE",
        )

        # Initial order: Cube (0.85) > Customer (0.80)
        candidates = [cand_cube, cand_customer]
        reranked = rerank_candidates(candidates, intent="MASTER_DATA")

        # Under MASTER_DATA intent, I_Customer (#DIMENSION) gets +0.20 boost + 0.10 Core boost -> 1.10
        # beating Cube (0.85 + 0.10 = 0.95)!
        self.assertEqual(reranked[0].name, "I_CUSTOMER")
        self.assertEqual(reranked[0].retrieval_score, 1.10)

    def test_metadata_aware_reranking_transactional_boost(self):
        """TRANSACTIONAL intent boosts #FACT views."""
        cand_fact = CandidateView(
            name="I_SALESORDERITEM",
            fields={"SALESORDER", "NETAMOUNT"},
            retrieval_score=0.80,
            data_category="#FACT",
        )
        cand_dim = CandidateView(
            name="I_CUSTOMER",
            fields={"CUSTOMER"},
            retrieval_score=0.85,
            data_category="#DIMENSION",
        )

        candidates = [cand_dim, cand_fact]
        reranked = rerank_candidates(candidates, intent="TRANSACTIONAL")

        # Under TRANSACTIONAL intent, #FACT gets +0.20 boost + 0.10 Core boost -> 1.10
        self.assertEqual(reranked[0].name, "I_SALESORDERITEM")
        self.assertEqual(reranked[0].retrieval_score, 1.10)

    # -------------------------------------------------------------
    # 4. End-to-End Integration with RequirementSolver
    # -------------------------------------------------------------
    def test_tri_state_solver_integration_with_semantic_router(self):
        """Semantic Router output passed directly into RequirementAnalyzer and RequirementSolver.

        Proves that an ANALYTICAL query uses dimensions + measures as requested_canonical_fields
        and grants REUSE only when the chosen view is an analytical cube.
        """
        analysis = self.analyzer.analyze(
            business_requirement="Total Amount by Customer",
            requested_fields=["Customer", "TotalAmount"],
        )

        self.assertTrue(analysis.is_analytical)
        self.assertEqual(analysis.intent, "ANALYTICAL")
        self.assertIn("Customer", analysis.requested_canonical_fields)
        self.assertIn("TotalNetAmount", analysis.requested_canonical_fields)

        # Candidate 1: I_CUSTOMER (100% field coverage if only Customer, but not a cube)
        cand_customer = CandidateView(
            name="I_CUSTOMER",
            fields={"CUSTOMER", "TOTALNETAMOUNT"},
            retrieval_score=0.95,
            is_analytical_cube=False,  # Not a cube
            data_category="#DIMENSION",
        )

        # Candidate 2: I_SALESORDERITEMCUBE (is a cube)
        cand_cube = CandidateView(
            name="I_SALESORDERITEMCUBE",
            fields={"CUSTOMER", "TOTALNETAMOUNT"},
            retrieval_score=0.75,
            is_analytical_cube=True,  # Is a cube
            data_category="#CUBE",
        )

        # Rerank candidates with soft boosting
        reranked = rerank_candidates([cand_customer, cand_cube], intent=analysis.intent)
        self.assertEqual(reranked[0].name, "I_SALESORDERITEMCUBE")

        # Solve requirement: I_SalesOrderItemCube is selected and accepted for REUSE!
        result = self.solver.solve(analysis=analysis, candidates=reranked)
        self.assertEqual(result.decision, "REUSE")
        self.assertEqual(result.primary_view_name, "I_SALESORDERITEMCUBE")
        self.assertEqual(result.confidence, 1.0)
        self.assertTrue(result.deployable)


if __name__ == "__main__":
    unittest.main()
