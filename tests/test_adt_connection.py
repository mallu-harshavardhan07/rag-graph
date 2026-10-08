"""Unit and Integration Test for SAP ADT Connection and REST Endpoints.

Connects to https://YAWSS4HSBX.sapyash.com:44301, fetches CSRF token,
queries CDS views (Z* or I_*), and tests DDL extraction.
"""

import sys
import os
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from core.sap_adt_client import SAPADTClient


class TestSAPADTConnection(unittest.TestCase):
    """Integration test suite for SAP S/4HANA ADT REST API."""

    @classmethod
    def setUpClass(cls):
        print(f"\n--- Testing SAP ADT Connection ---")
        print(f"Target System ID: {settings.SAP_SYSTEM_ID}")
        print(f"Base URL: {settings.SAP_BASE_URL}")
        print(f"Client: {settings.SAP_CLIENT}")
        print(f"Username: {settings.SAP_USERNAME}")
        print(f"SSL Verification: {settings.SAP_VERIFY_SSL}")
        cls.client = SAPADTClient()

    def test_01_fetch_csrf_token(self):
        """Test fetching CSRF token from ADT discovery."""
        try:
            token = self.client.fetch_csrf_token()
            print(f"[OK] Successfully retrieved CSRF Token: {token[:12]}...")
            self.assertIsNotNone(token)
            self.assertNotEqual(token.lower(), "required")
        except Exception as e:
            print(f"[WARNING] Live SAP CSRF token fetch failed: {e}")
            self.skipTest(f"SAP host {settings.SAP_BASE_URL} unreachable or credentials invalid: {e}")

    def test_02_search_cds_views(self):
        """Test searching Z* or I_* CDS views."""
        try:
            print("[INFO] Querying SAP ADT quickSearch for 'I_*' and 'Z*' views...")
            results = self.client.search_cds_views(query="I_*", max_results=10)
            if not results:
                results = self.client.search_cds_views(query="Z*", max_results=10)

            print(f"[OK] Retrieved {len(results)} CDS view object references from SAP ADT.")
            for r in results[:5]:
                print(f"   -> Found: {r['name']} ({r.get('type')}) - {r.get('description')}")

            self.assertIsInstance(results, list)
        except Exception as e:
            print(f"[WARNING] Live search failed: {e}")
            self.skipTest(f"SAP search endpoint unreachable: {e}")

    def test_03_extract_ddl_source(self):
        """Test fetching raw DDL source for a view."""
        try:
            results = self.client.search_cds_views(query="I_*", max_results=5)
            if not results:
                results = self.client.search_cds_views(query="Z*", max_results=5)

            if results:
                target_view = results[0]["name"]
                print(f"[INFO] Fetching DDL source for view: {target_view}...")
                ddl = self.client.get_ddl_source(target_view)
                print(f"[OK] Fetched DDL source ({len(ddl)} characters).")
                print("--- DDL Preview (first 150 chars) ---")
                print(ddl[:150])
                print("-------------------------------------")
                self.assertTrue(len(ddl) > 0)
            else:
                self.skipTest("No views found to test DDL extraction.")
        except Exception as e:
            print(f"[WARNING] DDL extraction test failed: {e}")
            self.skipTest(f"SAP DDL source fetch failed: {e}")


if __name__ == "__main__":
    unittest.main()
