"""Unit Tests for SAP VDM Hierarchy Reranker, Penalization, and Unique Field Protection.

Verifies:
1. Interface/Core Views (I_*, #BASIC, #COMPOSITE) receive +0.1 Core boost.
2. Consumption Views (C_*, #CONSUMPTION) receive -0.2 Consumption penalty.
3. Value Help Views (*_VH, VALUEHELP, #VALUE_HELP) receive -0.5 Value Help penalty.
4. Value Help classification takes precedence over I_ prefix (e.g. I_SalesOrder_VH gets -0.5).
5. Explicit metadata (@VDM.viewType, raw_metadata['vdm_type']) takes precedence over naming.
6. Unique Source Protection: Heavily penalized views (e.g., VH) that uniquely cover a required
   field have penalties waived and receive protection boost to stay at the top of the candidate pool.
7. Soft signal only: Candidates are never eliminated or pruned from the pool.
8. Data integrity: Candidate fields, keys, field_dict, and associations remain strictly unmutated.
9. Full auditability: Both base_score and adjusted_score are preserved and accessible in metadata.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.composition_solver import CandidateView
from core.recommender import (
    classify_vdm_type,
    find_unique_source_views,
    rerank_candidates,
)


class TestVDMHierarchyReranker(unittest.TestCase):
    """Test suite for SAP VDM Hierarchy Reranking & Penalization."""

    def test_core_view_boost_i_salesorder(self):
        """I_SalesOrder receives +0.1 Interface/Core boost."""
        cand = CandidateView(
            name="I_SalesOrder",
            fields={"SALESORDER", "CUSTOMER"},
            retrieval_score=0.80,
        )
        self.assertEqual(classify_vdm_type(cand), "INTERFACE")
        reranked = rerank_candidates([cand])
        self.assertEqual(reranked[0].base_score, 0.80)
        self.assertEqual(reranked[0].adjusted_score, 0.90)
        self.assertEqual(reranked[0].retrieval_score, 0.90)
        self.assertEqual(reranked[0].raw_metadata.get("vdm_adjustment"), 0.10)

    def test_core_view_boost_i_customer(self):
        """I_Customer receives +0.1 Interface/Core boost."""
        cand = CandidateView(
            name="I_Customer",
            fields={"CUSTOMER", "CUSTOMERNAME"},
            retrieval_score=0.85,
        )
        self.assertEqual(classify_vdm_type(cand), "INTERFACE")
        reranked = rerank_candidates([cand])
        self.assertEqual(reranked[0].base_score, 0.85)
        self.assertEqual(reranked[0].adjusted_score, 0.95)
        self.assertEqual(reranked[0].retrieval_score, 0.95)

    def test_consumption_view_penalty_c_salesorder(self):
        """C_SalesOrder receives -0.2 Consumption penalty."""
        cand = CandidateView(
            name="C_SalesOrder",
            fields={"SALESORDER", "CUSTOMER"},
            retrieval_score=0.80,
        )
        self.assertEqual(classify_vdm_type(cand), "CONSUMPTION")
        reranked = rerank_candidates([cand])
        self.assertEqual(reranked[0].base_score, 0.80)
        self.assertEqual(reranked[0].adjusted_score, 0.60)
        self.assertEqual(reranked[0].retrieval_score, 0.60)
        self.assertEqual(reranked[0].raw_metadata.get("vdm_adjustment"), -0.20)

    def test_value_help_penalty_suffix_vh(self):
        """Views ending with _VH receive -0.5 Value Help penalty (even if starting with I_)."""
        cand = CandidateView(
            name="I_SalesOrder_VH",
            fields={"SALESORDER"},
            retrieval_score=0.90,
        )
        self.assertEqual(classify_vdm_type(cand), "VALUE_HELP")
        reranked = rerank_candidates([cand])
        self.assertEqual(reranked[0].base_score, 0.90)
        self.assertEqual(reranked[0].adjusted_score, 0.40)
        self.assertEqual(reranked[0].retrieval_score, 0.40)
        self.assertEqual(reranked[0].raw_metadata.get("vdm_adjustment"), -0.50)

    def test_value_help_penalty_valuehelp_keyword(self):
        """Views containing VALUEHELP in name receive -0.5 penalty."""
        cand = CandidateView(
            name="I_CUSTOMER_VALUEHELP",
            fields={"CUSTOMER"},
            retrieval_score=0.85,
        )
        self.assertEqual(classify_vdm_type(cand), "VALUE_HELP")
        reranked = rerank_candidates([cand])
        self.assertEqual(reranked[0].base_score, 0.85)
        self.assertEqual(reranked[0].adjusted_score, 0.35)
        self.assertEqual(reranked[0].retrieval_score, 0.35)
        self.assertEqual(reranked[0].raw_metadata.get("vdm_adjustment"), -0.50)

    def test_metadata_precedence_over_naming(self):
        """Explicit metadata annotations override naming convention heuristics."""
        # 1. View named C_CustomView but metadata states #BASIC -> INTERFACE (+0.1)
        cand_basic = CandidateView(
            name="C_CustomComposite",
            fields={"SALESORDER"},
            retrieval_score=0.70,
            raw_metadata={"vdm_type": "BASIC"},
        )
        self.assertEqual(classify_vdm_type(cand_basic), "INTERFACE")
        reranked_basic = rerank_candidates([cand_basic])
        self.assertEqual(reranked_basic[0].adjusted_score, 0.80)

        # 2. View named I_CustHelp but raw_ddl has @VDM.viewType: #VALUE_HELP -> VALUE_HELP (-0.5)
        cand_vh = CandidateView(
            name="I_CustHelp",
            fields={"CUSTOMER"},
            retrieval_score=0.80,
            raw_metadata={"raw_ddl": "define view I_CustHelp as select from kna1 @VDM.viewType: #VALUE_HELP"},
        )
        self.assertEqual(classify_vdm_type(cand_vh), "VALUE_HELP")
        reranked_vh = rerank_candidates([cand_vh])
        self.assertEqual(reranked_vh[0].adjusted_score, 0.30)

    def test_unique_source_protection_guarantees_retention(self):
        """Unique Source Protection waives penalty and keeps unique provider at top of pool.
        
        Scenario:
        - Required field: 'UniqueTaxCode'
        - Candidate A (I_Customer): Base 0.95, has 'Customer'. Core view -> +0.1 -> 1.05.
        - Candidate B (I_TaxCode_VH): Base 0.80, has 'Customer' and 'UniqueTaxCode'.
          Normally penalized by -0.5 to 0.30, causing it to fall behind or drop out.
        - Because I_TaxCode_VH is the ONLY candidate with 'UniqueTaxCode', penalty is waived
          and protection boost is applied (+1.0 -> 1.80), keeping it firmly at index 0.
        """
        cand_customer = CandidateView(
            name="I_Customer",
            fields={"CUSTOMER", "CUSTOMERNAME"},
            retrieval_score=0.95,
        )
        cand_vh = CandidateView(
            name="I_TaxCode_VH",
            fields={"CUSTOMER", "UNIQUETAXCODE"},
            retrieval_score=0.80,
        )

        candidates = [cand_customer, cand_vh]
        required_fields = ["Customer", "UniqueTaxCode"]

        # Check unique source detection
        uniques = find_unique_source_views(candidates, required_fields)
        self.assertIn("I_TAXCODE_VH", uniques)
        self.assertNotIn("I_CUSTOMER", uniques)

        reranked = rerank_candidates(candidates, required_fields=required_fields)

        # Candidate B must be retained at the top (index 0)
        self.assertEqual(reranked[0].name, "I_TaxCode_VH")
        self.assertGreater(reranked[0].retrieval_score, reranked[1].retrieval_score)
        self.assertTrue(reranked[0].raw_metadata.get("is_unique_source"))
        self.assertEqual(reranked[0].raw_metadata.get("vdm_adjustment"), 0.0)  # Waived
        self.assertEqual(reranked[0].adjusted_score, 1.80)  # 0.80 base + 0.0 vdm + 1.0 protection

        # Candidate A gets normal core boost
        self.assertEqual(reranked[1].name, "I_Customer")
        self.assertEqual(reranked[1].adjusted_score, 1.05)  # 0.95 + 0.10

    def test_soft_signals_only_no_candidate_pruning(self):
        """Reranking acts as a soft signal only and never eliminates candidates."""
        candidates = [
            CandidateView(name="I_Core", fields={"A"}, retrieval_score=0.90),
            CandidateView(name="C_Consumption", fields={"B"}, retrieval_score=0.70),
            CandidateView(name="Z_Custom", fields={"C"}, retrieval_score=0.50),
            CandidateView(name="I_Help_VH", fields={"D"}, retrieval_score=0.60),
            CandidateView(name="UnknownView", fields={"E"}, retrieval_score=0.40),
        ]

        reranked = rerank_candidates(candidates)
        # All 5 candidates must remain in the pool
        self.assertEqual(len(reranked), 5)
        names = [c.name for c in reranked]
        self.assertEqual(set(names), {"I_Core", "C_Consumption", "Z_Custom", "I_Help_VH", "UnknownView"})

    def test_candidate_metadata_and_fields_integrity(self):
        """Underlying canonical field metadata and associations are not mutated or dropped."""
        original_fields = {"SALESORDER", "CUSTOMER", "NETAMOUNT"}
        original_keys = ["SALESORDER"]
        original_field_dict = {"SALESORDER": "SalesOrder", "CUSTOMER": "Customer"}
        original_assocs = ["_Customer"]

        cand = CandidateView(
            name="I_SalesOrder",
            fields=set(original_fields),
            keys=list(original_keys),
            field_dict=dict(original_field_dict),
            associations=list(original_assocs),
            retrieval_score=0.85,
        )

        reranked = rerank_candidates([cand])
        result = reranked[0]

        # Field collections and keys must remain intact
        self.assertEqual(result.fields, original_fields)
        self.assertEqual(result.keys, original_keys)
        self.assertEqual(result.field_dict, original_field_dict)
        self.assertEqual(result.associations, original_assocs)

        # Audit metadata must be recorded
        self.assertIn("base_score", result.raw_metadata)
        self.assertIn("adjusted_score", result.raw_metadata)
        self.assertIn("vdm_type", result.raw_metadata)
        self.assertEqual(result.base_score, 0.85)
        self.assertEqual(result.adjusted_score, 0.95)


if __name__ == "__main__":
    unittest.main()
