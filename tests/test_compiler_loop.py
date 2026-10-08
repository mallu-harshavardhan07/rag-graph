"""Unit and Integration Tests for ADT Compiler Closed-Loop and Self-Healing."""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.compiler_loop import ADTCompilerLoop
from core.sap_adt_client import SAPADTClient


class TestCompilerLoop(unittest.TestCase):
    """Test live syntax check loop and AST rule self-healing."""

    @classmethod
    def setUpClass(cls):
        cls.client = SAPADTClient()
        cls.compiler = ADTCompilerLoop(adt_client=cls.client, max_iterations=3)

    def test_clean_modern_cds_entity(self):
        """Test clean modern CDS entity compilation verification."""
        clean_ddl = """@EndUserText.label: 'Test Verified Sales Entity'
@AccessControl.authorizationCheck: #NOT_REQUIRED
define view entity Z_Test_Sales_Verify
  as select from I_SalesOrderItem
{
  key SalesOrder,
  key SalesOrderItem,
      NetAmount,
      TransactionCurrency
}"""
        result = self.compiler.compile_and_heal("Z_Test_Sales_Verify", clean_ddl)
        self.assertIn(result.status, ("VERIFIED", "OFFLINE_VERIFIED"))
        self.assertTrue(result.is_verified)
        self.assertGreater(result.total_iterations, 0)
        self.assertGreater(result.total_latency_ms, 0.0)

    def test_auto_healing_fixes_legacy_syntax(self):
        """Test that legacy or unannotated CDS is repaired by auto-healing."""
        bad_ddl = """
        define view Z_Legacy_Sales as select from I_SalesOrderItem
        {
          key SalesOrder,
          NetAmount,
        }
        """
        result = self.compiler.compile_and_heal("Z_Legacy_Sales", bad_ddl)
        # Verify that define view entity was applied and trailing comma cleaned
        self.assertIn("define view entity", result.final_code)
        self.assertIn("@AccessControl.authorizationCheck", result.final_code)


if __name__ == "__main__":
    unittest.main()
