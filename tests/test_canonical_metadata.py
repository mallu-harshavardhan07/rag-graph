"""Unit Tests for Canonical Metadata Registry, Field Reverse Index, and Canonical Field Resolver.

Verifies:
1. Canonical view metadata model integrity.
2. Deterministic Field -> View reverse index lookups.
3. Provenance and validation tracking in CanonicalFieldResolver.
4. SAP business vocabulary resolution (e.g. client -> Customer, vendor -> Supplier, planned quantity -> TargetQuantity).
5. Unvalidated terms prevention from entering deterministic coverage.
6. Index versioning and consistency.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.canonical_metadata import (
    CanonicalField,
    CanonicalAssociation,
    AnalyticalCharacteristics,
    CanonicalViewMetadata,
    FieldToViewIndex,
    CanonicalFieldResolver,
    CanonicalMetadataRegistry,
    METADATA_VERSION,
)


class TestCanonicalMetadata(unittest.TestCase):
    """Test suite for Single Source of Truth Canonical Metadata and Reverse Index."""

    def setUp(self):
        self.registry = CanonicalMetadataRegistry()

    def test_canonical_field_matching(self):
        """CanonicalField matches exact, stripped, label, and aliases."""
        field = CanonicalField(
            canonical_name="SalesOrder",
            sap_name="SalesOrder",
            label="Sales Order",
            aliases=["order number", "SO_NUMBER"],
            is_key=True,
        )

        self.assertTrue(field.matches_term("SalesOrder"))
        self.assertTrue(field.matches_term("salesorder"))
        self.assertTrue(field.matches_term("sales_order"))
        self.assertTrue(field.matches_term("Sales Order"))
        self.assertTrue(field.matches_term("order number"))
        self.assertTrue(field.matches_term("so_number"))
        self.assertFalse(field.matches_term("PurchaseOrder"))

    def test_canonical_association_safety(self):
        """CanonicalAssociation strictly classifies 1..1 and 0..1 as safe, others as unsafe."""
        safe_one = CanonicalAssociation(source="A", target="B", alias="_B", cardinality="1..1")
        safe_zero_one = CanonicalAssociation(source="A", target="B", alias="_B", cardinality="0..1")
        unsafe_many = CanonicalAssociation(source="A", target="B", alias="_B", cardinality="0..*")
        unsafe_one_many = CanonicalAssociation(source="A", target="B", alias="_B", cardinality="1..*")

        self.assertTrue(safe_one.is_safe)
        self.assertTrue(safe_zero_one.is_safe)
        self.assertFalse(unsafe_many.is_safe)
        self.assertFalse(unsafe_one_many.is_safe)

    def test_field_to_view_reverse_index(self):
        """FieldToViewIndex provides deterministic sub-millisecond reverse lookups."""
        index = FieldToViewIndex()

        view_order = CanonicalViewMetadata(
            view_name="I_SalesOrder",
            fields=[
                CanonicalField("SalesOrder", "SalesOrder", is_key=True),
                CanonicalField("Customer", "Customer"),
                CanonicalField("TotalNetAmount", "TotalNetAmount", is_measure=True),
            ],
            keys=["SalesOrder"],
        )

        view_item = CanonicalViewMetadata(
            view_name="I_SalesOrderItem",
            fields=[
                CanonicalField("SalesOrder", "SalesOrder", is_key=True),
                CanonicalField("SalesOrderItem", "SalesOrderItem", is_key=True),
                CanonicalField("Product", "Product"),
                CanonicalField("NetPriceAmount", "NetPriceAmount", is_measure=True),
            ],
            keys=["SalesOrder", "SalesOrderItem"],
        )

        index.add_view(view_order)
        index.add_view(view_item)

        # Lookup views for SalesOrder -> both views
        so_views = index.lookup_views("SalesOrder")
        self.assertIn("I_SALESORDER", so_views)
        self.assertIn("I_SALESORDERITEM", so_views)

        # Lookup views for Customer -> only I_SalesOrder
        cust_views = index.lookup_views("Customer")
        self.assertIn("I_SALESORDER", cust_views)
        self.assertNotIn("I_SALESORDERITEM", cust_views)

        # Lookup views for Product -> only I_SalesOrderItem
        prod_views = index.lookup_views("Product")
        self.assertIn("I_SALESORDERITEM", prod_views)
        self.assertNotIn("I_SALESORDER", prod_views)

    def test_canonical_field_resolver_vocabulary_mapping(self):
        """CanonicalFieldResolver maps natural business terminology to canonical concepts."""
        resolver = self.registry.field_resolver

        # 1. 'client' -> Customer
        res_client = resolver.resolve("client")
        self.assertEqual(res_client.canonical_name, "Customer")
        self.assertEqual(res_client.validated, True)
        self.assertIn("VOCABULARY", res_client.source)

        # 2. 'vendor' -> Supplier
        res_vendor = resolver.resolve("vendor")
        self.assertEqual(res_vendor.canonical_name, "Supplier")
        self.assertEqual(res_vendor.validated, True)

        # 3. 'order number' -> SalesOrder
        res_ord = resolver.resolve("order number")
        self.assertEqual(res_ord.canonical_name, "SalesOrder")
        self.assertEqual(res_ord.validated, True)

        # 4. 'net sales' -> TotalNetAmount
        res_sales = resolver.resolve("net sales")
        self.assertEqual(res_sales.canonical_name, "TotalNetAmount")
        self.assertEqual(res_sales.validated, True)

        # 5. 'planned quantity' -> TargetQuantity
        res_qty = resolver.resolve("planned quantity")
        self.assertEqual(res_qty.canonical_name, "TargetQuantity")
        self.assertEqual(res_qty.validated, True)

    def test_canonical_field_resolver_unvalidated_term(self):
        """Unknown or hallucinated business terms are flagged as unvalidated."""
        resolver = self.registry.field_resolver
        res = resolver.resolve("NonExistentCustomFantasyFieldXYZ")
        self.assertFalse(res.validated)
        self.assertEqual(res.source, "UNVALIDATED_PROPOSAL")
        self.assertLess(res.confidence, 0.5)

    def test_registry_view_normalization_and_version(self):
        """CanonicalMetadataRegistry correctly normalizes raw dictionary and tracks version."""
        raw_dict = {
            "view_name": "I_SalesOrder",
            "description": "Sales Order Header View",
            "domain": "SD",
            "data_category": "#TRANSACTIONAL",
            "fields": [
                {"name": "SalesOrder", "is_key": True},
                {"name": "Customer", "is_key": False},
                {"name": "TotalNetAmount", "is_measure": True},
            ],
            "associations": [
                {
                    "target_view": "I_Customer",
                    "alias": "_Customer",
                    "cardinality": "[0..1]",
                    "join_keys": ["Customer"],
                }
            ],
        }

        v_meta = self.registry.register_view(raw_dict)
        self.assertEqual(v_meta.view_name, "I_SalesOrder")
        self.assertEqual(v_meta.vdm_type, "INTERFACE")
        self.assertEqual(v_meta.keys, ["SalesOrder"])
        self.assertEqual(len(v_meta.associations), 1)
        self.assertTrue(v_meta.associations[0].is_safe)
        self.assertEqual(self.registry.version, METADATA_VERSION)


if __name__ == "__main__":
    unittest.main()
