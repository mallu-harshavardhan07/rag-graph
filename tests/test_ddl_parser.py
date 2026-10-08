"""Unit tests for ABAP CDS DDL Parser and Feature Extractor."""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.ddl_parser import DDLParser


class TestDDLParser(unittest.TestCase):
    """Test parsing various ABAP CDS DDL constructs."""

    def test_parse_modern_cds_entity(self):
        sample_ddl = """
        @EndUserText.label: 'Sales Order Item Analytics'
        @AccessControl.authorizationCheck: #NOT_REQUIRED
        define view entity Z_SalesOrder_Item
          as select from vbap
          association [0..1] to I_Customer as _Customer
            on $projection.CustomerID = _Customer.Customer
          association [0..*] to I_Product as _Product
            on $projection.ProductID = _Product.Product
        {
          key vbeln as SalesOrder,
          key posnr as SalesOrderItem,
              kunnr as CustomerID,
              matnr as ProductID,
              netwr as NetAmount,
              _Customer,
              _Product
        }
        """
        parsed = DDLParser.parse(sample_ddl)
        self.assertEqual(parsed["view_name"], "Z_SALESORDER_ITEM")
        self.assertEqual(parsed["view_type"], "CUSTOM")
        self.assertEqual(parsed["description"], "Sales Order Item Analytics")
        self.assertIn("SD", parsed["domain"])
        self.assertEqual(parsed["base_data_sources"], ["VBAP"])

        # Check fields
        field_names = [f["name"] for f in parsed["fields"]]
        self.assertIn("SalesOrder", field_names)
        self.assertIn("SalesOrderItem", field_names)
        self.assertIn("CustomerID", field_names)
        self.assertIn("NetAmount", field_names)

        # Check keys
        keys = [f["name"] for f in parsed["fields"] if f["is_key"]]
        self.assertEqual(keys, ["SalesOrder", "SalesOrderItem"])

        # Check associations
        self.assertEqual(len(parsed["associations"]), 2)
        assoc_targets = [a["target_view"] for a in parsed["associations"]]
        self.assertIn("I_CUSTOMER", assoc_targets)
        self.assertIn("I_PRODUCT", assoc_targets)

        # Check association details
        cust_assoc = next(a for a in parsed["associations"] if a["target_view"] == "I_CUSTOMER")
        self.assertEqual(cust_assoc["alias"], "_Customer")
        self.assertEqual(cust_assoc["cardinality"], "[0..1]")
        self.assertIn("$projection.CustomerID = _Customer.Customer", cust_assoc["on_condition"])

    def test_parse_classic_cds_view(self):
        sample_ddl = """
        @AbapCatalog.sqlViewName: 'ZPOHEAD'
        @EndUserText.label: 'Purchase Order Header'
        define view Z_Purchase_Order as select from ekko
          association [0..1] to I_Supplier as _Supplier
            on $projection.Supplier = _Supplier.Supplier
        {
          key ebeln as PurchaseOrder,
              lifnr as Supplier,
              netwr as TotalAmount
        }
        """
        parsed = DDLParser.parse(sample_ddl)
        self.assertEqual(parsed["view_name"], "Z_PURCHASE_ORDER")
        self.assertEqual(parsed["sql_view_name"], "ZPOHEAD")
        self.assertEqual(parsed["description"], "Purchase Order Header")
        self.assertIn("MM", parsed["domain"])
        self.assertEqual(len(parsed["fields"]), 3)
        self.assertEqual(len(parsed["associations"]), 1)


if __name__ == "__main__":
    unittest.main()
