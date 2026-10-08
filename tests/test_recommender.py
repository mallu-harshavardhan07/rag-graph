"""Unit and Integration Tests for Reusability Evaluator & Deterministic Graph Join-Tree Resolver."""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.database import StorageEngine
from core.recommender import ReusabilityEvaluator, resolve_field_join_plan

SAMPLE_TEST_VIEWS = [
    {
        "view_name": "I_SALESORDER",
        "sql_view_name": "ISALESORDER",
        "view_type": "STANDARD",
        "description": "Sales Order Header",
        "domain": "SD (Sales & Distribution)",
        "base_data_sources": ["VBAK"],
        "fields": [
            {"name": "SalesOrder", "alias": "SalesOrder", "is_key": True},
            {"name": "SoldToParty", "alias": "SoldToParty", "is_key": False},
            {"name": "TotalNetAmount", "alias": "TotalNetAmount", "is_key": False},
        ],
        "associations": [
            {
                "target_view": "I_CUSTOMER",
                "alias": "_SoldToParty",
                "cardinality": "[0..1]",
                "on_condition": "$projection.SoldToParty = _SoldToParty.Customer",
            }
        ],
    },
    {
        "view_name": "I_CUSTOMER",
        "sql_view_name": "ICUSTOMER",
        "view_type": "STANDARD",
        "description": "Customer Master",
        "domain": "SD (Sales & Distribution)",
        "base_data_sources": ["KNA1"],
        "fields": [
            {"name": "Customer", "alias": "Customer", "is_key": True},
            {"name": "CustomerName", "alias": "CustomerName", "is_key": False},
            {"name": "CityName", "alias": "CityName", "is_key": False},
        ],
        "associations": [],
    },
    {
        "view_name": "Z_SALES_ITEMS",
        "sql_view_name": "ZSALESITEMS",
        "view_type": "CUSTOM",
        "description": "Custom Sales Order Item Analysis View",
        "domain": "SD (Sales & Distribution)",
        "base_data_sources": ["VBAP"],
        "fields": [
            {"name": "SalesOrder", "alias": "SalesOrder", "is_key": True},
            {"name": "SalesOrderItem", "alias": "SalesOrderItem", "is_key": True},
            {"name": "Product", "alias": "Product", "is_key": False},
            {"name": "NetPrice", "alias": "NetPrice", "is_key": False},
            {"name": "NetAmount", "alias": "NetAmount", "is_key": False},
        ],
        "associations": [],
    },
]


class TestReusabilityEvaluator(unittest.TestCase):
    """Test CDS View reusability evaluation and Knowledge Graph traversal."""

    @classmethod
    def setUpClass(cls):
        cls.storage = StorageEngine()
        for v in SAMPLE_TEST_VIEWS:
            cls.storage.add_cds_view(v)
        cls.evaluator = ReusabilityEvaluator(storage_engine=cls.storage)

    def test_path_a_100_percent_reuse(self):
        """Test Path A: All requested fields exist in Z_SALES_ITEMS -> REUSE."""
        req_fields = ["SalesOrder", "SalesOrderItem", "Product", "NetPrice", "NetAmount"]
        result = self.evaluator.evaluate(
            business_requirement="Sales order item pricing analysis with net amount",
            domain="SD (Sales & Distribution)",
            required_fields=req_fields,
            scope="CUSTOM_ONLY",
        )

        self.assertEqual(result.decision, "REUSE")
        self.assertEqual(result.match_percentage, 100.0)
        self.assertIsNotNone(result.primary_view)
        self.assertEqual(len(result.missing_fields), 0)
        self.assertEqual(len(result.uncovered_fields), 0)

    def test_path_b_graph_guided_composite(self):
        """Test Path B: Missing fields resolved via Knowledge Graph traversal."""
        req_fields = ["SalesOrder", "TotalNetAmount", "CustomerName", "CityName"]
        result = self.evaluator.evaluate(
            business_requirement="Sales order header with customer name and city",
            domain="SD (Sales & Distribution)",
            required_fields=req_fields,
            scope="STANDARD_ONLY",
        )

        self.assertIn(result.decision, ("CREATE", "EXTEND", "GENERATE_COMPOSITE"))
        self.assertIn("join_plan", dir(result))
        plan = result.join_plan
        self.assertIn("field_mapping", plan)
        self.assertIn("required_associations", plan)

    def test_resolve_field_join_plan_strict_unresolved(self):
        """Test that resolve_field_join_plan strictly flags unknown fields as unresolved and never maps to _Primary."""
        req_fields = ["SalesOrder", "TotalNetAmount", "CustomerName", "CompletelyFakeFieldXYZ"]
        plan = resolve_field_join_plan(
            primary_view="I_SALESORDER",
            requested_fields=req_fields,
            graph=self.storage.graph_store,
        )

        # 1. Primary fields mapped to _Primary
        self.assertEqual(plan["field_mapping"]["SalesOrder"]["alias"], "_Primary")
        self.assertEqual(plan["field_mapping"]["TotalNetAmount"]["alias"], "_Primary")

        # 2. Associated fields mapped to _Customer or _SoldToParty
        self.assertIn("CustomerName", plan["field_mapping"])
        self.assertNotEqual(plan["field_mapping"]["CustomerName"]["alias"], "_Primary")

        # 3. CRITICAL: Unresolved field strictly placed in unresolved_fields and NOT in field_mapping
        self.assertIn("CompletelyFakeFieldXYZ", plan["unresolved_fields"])
        self.assertNotIn("CompletelyFakeFieldXYZ", plan["field_mapping"])


if __name__ == "__main__":
    unittest.main()
