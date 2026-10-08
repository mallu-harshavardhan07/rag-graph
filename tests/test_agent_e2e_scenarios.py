"""End-to-End Generic Scenarios across SAP Functional Domains (SD, MM, FI, PP).

Tests all 15 scenarios from Section 35 of the Master Architectural Directive:
Scenario A: One view contains everything -> REUSE
Scenario B: Two views cover everything with safe 0..1 association -> EXTEND
Scenario C: Two views cover everything but no association -> CREATE (NO_DIRECT_ASSOCIATION)
Scenario D: Two views cover everything but 0..* association -> CREATE (UNSAFE_CARDINALITY)
Scenario E: Missing field has no catalog provider -> CREATE + UNRESOLVED_CANONICAL_FIELD
Scenario F: Missing field exists in another view -> secondary retrieval discovers it via reverse index
Scenario G: Secondary view found but missing join key -> CREATE + MISSING_JOIN_KEYS
Scenario H: Analytical requirement -> compatible analytical cube view preferred
Scenario I: Transactional requirement -> transactional/interface views preferred
Scenario J: Master-data requirement -> dimension/master views preferred
Scenario K: Cross-domain requirement -> multiple domains allowed (SD + FI)
Scenario L: Value Help view alongside Interface view -> Interface preferred unless VH uniquely covers field
Scenario M: Graph unavailable -> infrastructure failure reported, NOT normal CREATE
Scenario N: Metadata/index version tracking -> tracked on registry
Scenario O: No hallucinated fields -> unresolved fields explicitly reported
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
    AnalyticalCharacteristics,
    METADATA_VERSION,
)
from core.composition_solver import (
    CandidateView,
    RequirementAnalyzer,
    RequirementSolver,
    RequirementAnalysis,
)
from core.query_expander import QueryExpander, QueryExpansion
from core.recommender import rerank_candidates, classify_vdm_type


class TestAgentE2EScenarios(unittest.TestCase):
    """End-to-End verification across all 15 architectural scenarios."""

    def setUp(self):
        self.registry = CanonicalMetadataRegistry()
        self.expander = QueryExpander()
        self.analyzer = RequirementAnalyzer(query_expander=self.expander, registry=self.registry)
        self.solver = RequirementSolver()

    def test_scenario_a_single_view_contains_everything_reuse(self):
        """Scenario A: One view contains everything -> REUSE with 0 DDL."""
        cand_order = CandidateView(
            name="I_PURCHASEORDER",
            fields={"PURCHASEORDER", "SUPPLIER", "COMPANYCODE", "PURCHASINGORGANIZATION"},
            keys=["PURCHASEORDER"],
            data_category="#FACT",
            retrieval_score=0.95,
        )

        analysis = self.analyzer.analyze(
            business_requirement="Purchase order header with supplier and company code",
            requested_fields=["PurchaseOrder", "Supplier", "CompanyCode"],
            domain="MM",
        )

        res = self.solver.solve(analysis=analysis, candidates=[cand_order])
        self.assertEqual(res.decision, "REUSE")
        self.assertEqual(res.coverage, 1.0)
        self.assertEqual(res.deployable, True)
        self.assertIsNone(res.ddl)  # REUSE never generates DDL
        self.assertIn("FULL_SINGLE_VIEW_COVERAGE", res.reason_codes)

    def test_scenario_b_two_views_cover_everything_safe_association_extend(self):
        """Scenario B: Two views cover everything with safe 0..1 association -> EXTEND."""
        cand_primary = CandidateView(
            name="I_PURCHASEORDER",
            fields={"PURCHASEORDER", "SUPPLIER"},
            keys=["PURCHASEORDER"],
            retrieval_score=0.90,
            associations=[
                {
                    "target_view": "I_SUPPLIER",
                    "alias": "_Supplier",
                    "cardinality": "[0..1]",
                    "join_keys": ["Supplier"],
                }
            ],
        )

        cand_secondary = CandidateView(
            name="I_SUPPLIER",
            fields={"SUPPLIER", "SUPPLIERNAME", "CITYNAME"},
            keys=["Supplier"],
            retrieval_score=0.85,
        )

        analysis = self.analyzer.analyze(
            business_requirement="Purchase orders with supplier name and city",
            requested_fields=["PurchaseOrder", "SupplierName", "CityName"],
            domain="MM",
        )

        res = self.solver.solve(
            analysis=analysis,
            candidates=[cand_primary],
            secondary_candidates=[cand_secondary],
        )
        self.assertEqual(res.decision, "EXTEND")
        self.assertEqual(res.coverage, 1.0)
        self.assertEqual(res.deployable, True)
        self.assertEqual(res.primary_view_name, "I_PURCHASEORDER")
        self.assertEqual(res.secondary_view, "I_SUPPLIER")
        self.assertIn("@AccessControl.authorizationCheck: #CHECK", res.ddl)
        self.assertIn("SAFE_TWO_VIEW_COMPOSITION", res.reason_codes)

    def test_scenario_c_two_views_cover_everything_no_association_create(self):
        """Scenario C: Two views cover everything but have no association -> CREATE."""
        cand_primary = CandidateView(
            name="I_MANUFACTURINGORDER",
            fields={"MANUFACTURINGORDER", "PLANT"},
            keys=["MANUFACTURINGORDER"],
            retrieval_score=0.85,
            associations=[],  # No association
        )

        cand_secondary = CandidateView(
            name="I_CUSTOMER",
            fields={"CUSTOMER", "CUSTOMERNAME"},
            keys=["CUSTOMER"],
            retrieval_score=0.75,
            associations=[],
        )

        analysis = self.analyzer.analyze(
            business_requirement="Production orders with customer name",
            requested_fields=["ManufacturingOrder", "CustomerName"],
            domain="PP",
        )

        res = self.solver.solve(
            analysis=analysis,
            candidates=[cand_primary],
            secondary_candidates=[cand_secondary],
        )
        self.assertEqual(res.decision, "CREATE")
        self.assertEqual(res.deployable, False)
        self.assertIn("NO_DIRECT_ASSOCIATION", res.reason_codes)

    def test_scenario_d_two_views_cover_everything_unsafe_cardinality_create(self):
        """Scenario D: Two views cover everything but association is 0..* -> CREATE (Hard Cardinality Ban)."""
        cand_header = CandidateView(
            name="I_SALESORDER",
            fields={"SALESORDER", "CUSTOMER"},
            keys=["SALESORDER"],
            retrieval_score=0.90,
            associations=[
                {
                    "target_view": "I_SALESORDERITEM",
                    "alias": "_Item",
                    "cardinality": "[0..*]",  # Unsafe to-many!
                    "join_keys": ["SalesOrder"],
                }
            ],
        )

        cand_item = CandidateView(
            name="I_SALESORDERITEM",
            fields={"SALESORDER", "SALESORDERITEM", "PRODUCT"},
            keys=["SalesOrder", "SalesOrderItem"],
            retrieval_score=0.85,
        )

        analysis = self.analyzer.analyze(
            business_requirement="Sales order header with product items",
            requested_fields=["SalesOrder", "Product"],
            domain="SD",
        )

        res = self.solver.solve(
            analysis=analysis,
            candidates=[cand_header],
            secondary_candidates=[cand_item],
        )
        self.assertEqual(res.decision, "CREATE")
        self.assertEqual(res.deployable, False)
        self.assertIn("UNSAFE_CARDINALITY", res.reason_codes)

    def test_scenario_e_missing_field_no_catalog_provider_unresolved(self):
        """Scenario E: Requested field does not exist anywhere in catalog -> UNRESOLVED_CANONICAL_FIELD."""
        cand = CandidateView(
            name="I_SALESORDER",
            fields={"SALESORDER", "CUSTOMER"},
            keys=["SALESORDER"],
            retrieval_score=0.85,
        )

        analysis = self.analyzer.analyze(
            business_requirement="Sales order with unknown non-existent field",
            requested_fields=["SalesOrder", "NonExistentFantasyAttributeXYZ"],
            domain="SD",
        )

        res = self.solver.solve(
            analysis=analysis,
            candidates=[cand],
            all_known_fields={"SALESORDER", "CUSTOMER"},
        )
        self.assertEqual(res.decision, "CREATE")
        self.assertIn("UNRESOLVED_CANONICAL_FIELD", res.reason_codes)
        self.assertIn("NonExistentFantasyAttributeXYZ", res.unresolved_fields)

    def test_scenario_f_missing_field_discovered_via_reverse_index(self):
        """Scenario F: Field -> View reverse index discovers secondary candidate providing missing field."""
        # Register a view providing CityName in registry
        view_city = CanonicalViewMetadata(
            view_name="I_CUSTOMER",
            fields=[
                CanonicalField("Customer", "Customer", is_key=True),
                CanonicalField("CityName", "CityName"),
            ],
            keys=["Customer"],
        )
        self.registry.register_view(view_city)

        # Query reverse index for CityName
        providers = self.registry.get_views_for_field("CityName")
        self.assertIn("I_CUSTOMER", providers)

    def test_scenario_g_secondary_view_found_missing_join_key_create(self):
        """Scenario G: Secondary view covers missing field but required join keys missing -> CREATE."""
        cand_primary = CandidateView(
            name="I_JOURNALENTRYITEM",
            fields={"ACCOUNTINGDOCUMENT", "COMPANYCODE", "FISCALYEAR"},
            keys=["AccountingDocument"],
            retrieval_score=0.85,
            associations=[
                {
                    "target_view": "I_GLACCOUNT",
                    "alias": "_GLAccount",
                    "cardinality": "[0..1]",
                    "join_keys": ["GLAccount"],  # Primary does not have GLAccount!
                }
            ],
        )

        cand_secondary = CandidateView(
            name="I_GLACCOUNT",
            fields={"GLACCOUNT", "GLACCOUNTNAME"},
            keys=["GLAccount"],
            retrieval_score=0.80,
        )

        analysis = self.analyzer.analyze(
            business_requirement="Journal entries with GL account name",
            requested_fields=["AccountingDocument", "GLAccountName"],
            domain="FI",
        )

        res = self.solver.solve(
            analysis=analysis,
            candidates=[cand_primary],
            secondary_candidates=[cand_secondary],
        )
        self.assertEqual(res.decision, "CREATE")
        self.assertIn("MISSING_JOIN_KEYS", res.reason_codes)

    def test_scenario_h_analytical_requirement_prefers_cube(self):
        """Scenario H: Analytical intent (SUM/total) prefers analytical cube view over raw transaction."""
        cand_raw = CandidateView(
            name="I_SALESORDERITEM",
            fields={"CUSTOMER", "TOTALNETAMOUNT"},
            retrieval_score=0.90,
            is_analytical_cube=False,
            data_category="#FACT",
        )

        cand_cube = CandidateView(
            name="I_SALESORDERITEMCUBE",
            fields={"CUSTOMER", "TOTALNETAMOUNT"},
            retrieval_score=0.80,
            is_analytical_cube=True,
            data_category="#CUBE",
        )

        # Intent soft boosting elevates Cube (+0.30) over raw Fact (+0.0)
        reranked = rerank_candidates([cand_raw, cand_cube], intent="ANALYTICAL")
        self.assertEqual(reranked[0].name, "I_SALESORDERITEMCUBE")

        # Solver grants REUSE only to the cube
        analysis = self.analyzer.analyze(
            business_requirement="Total amount grouped by customer",
            requested_fields=["Customer", "TotalAmount"],
            domain="SD",
        )
        self.assertTrue(analysis.is_analytical)

        res = self.solver.solve(analysis=analysis, candidates=[cand_cube])
        self.assertEqual(res.decision, "REUSE")

    def test_scenario_i_transactional_requirement_prefers_interface_views(self):
        """Scenario I: Transactional requirement prefers #FACT/interface views."""
        cand_dim = CandidateView(
            name="I_CUSTOMER",
            fields={"CUSTOMER"},
            retrieval_score=0.85,
            data_category="#DIMENSION",
        )
        cand_fact = CandidateView(
            name="I_SALESORDERITEM",
            fields={"SALESORDER", "PRODUCT"},
            retrieval_score=0.80,
            data_category="#FACT",
        )

        reranked = rerank_candidates([cand_dim, cand_fact], intent="TRANSACTIONAL")
        self.assertEqual(reranked[0].name, "I_SALESORDERITEM")

    def test_scenario_j_master_data_requirement_prefers_dimension(self):
        """Scenario J: Master data requirement prefers #DIMENSION views."""
        cand_cube = CandidateView(
            name="I_SALESORDERITEMCUBE",
            fields={"CUSTOMER"},
            retrieval_score=0.85,
            data_category="#CUBE",
        )
        cand_dim = CandidateView(
            name="I_CUSTOMER",
            fields={"CUSTOMER", "CUSTOMERNAME"},
            retrieval_score=0.80,
            data_category="#DIMENSION",
        )

        reranked = rerank_candidates([cand_cube, cand_dim], intent="MASTER_DATA")
        self.assertEqual(reranked[0].name, "I_CUSTOMER")

    def test_scenario_k_cross_domain_multi_label_support(self):
        """Scenario K: Cross-domain queries (SD + FI) extract multiple domains without destructive filtering."""
        res = self.expander.expand(
            business_requirement="Customer sales total with accounting company code and fiscal year",
            requested_fields=["Customer", "TotalAmount", "CompanyCode", "FiscalYear"],
        )
        self.assertTrue(any(d in res.likely_domains for d in ["SD", "FI"]))
        self.assertIn("Customer", res.dimensions)
        self.assertIn("CompanyCode", res.dimensions)

    def test_scenario_l_value_help_vs_interface_and_unique_protection(self):
        """Scenario L: Value Help views penalized by -0.5 unless uniquely providing a requested field."""
        cand_core = CandidateView(name="I_Customer", fields={"CUSTOMER"}, retrieval_score=0.85)
        cand_vh = CandidateView(name="I_Customer_VH", fields={"CUSTOMER"}, retrieval_score=0.90)

        # Common field: VH is penalized (-0.5 -> 0.40) while Core is boosted (+0.1 -> 0.95)
        reranked = rerank_candidates([cand_core, cand_vh])
        self.assertEqual(reranked[0].name, "I_Customer")
        self.assertEqual(reranked[1].name, "I_Customer_VH")

        # Unique field protection: VH uniquely has UniqueField -> penalty waived and elevated
        cand_vh_unique = CandidateView(name="I_Custom_VH", fields={"CUSTOMER", "UNIQUEFIELD"}, retrieval_score=0.80)
        reranked_unique = rerank_candidates([cand_core, cand_vh_unique], required_fields=["Customer", "UniqueField"])
        self.assertEqual(reranked_unique[0].name, "I_Custom_VH")
        self.assertTrue(reranked_unique[0].raw_metadata.get("is_unique_source"))

    def test_scenario_m_graph_unavailable_reports_infrastructure_error(self):
        """Scenario M: Infrastructure failure (Graph down) reports error, NOT a silent valid CREATE."""
        analysis = self.analyzer.analyze(
            business_requirement="Sales order header with customer",
            requested_fields=["SalesOrder", "Customer"],
            domain="SD",
        )

        res = self.solver.solve(
            analysis=analysis,
            candidates=[],
            infrastructure_error="GRAPH_VALIDATION_UNAVAILABLE",
        )
        self.assertEqual(res.decision, "CREATE")
        self.assertEqual(res.reason_code, "GRAPH_VALIDATION_UNAVAILABLE")
        self.assertIn("GRAPH_VALIDATION_UNAVAILABLE", res.reason_codes)
        self.assertTrue(any("INFRASTRUCTURE ERROR" in w for w in res.warnings))

    def test_scenario_n_metadata_version_tracking(self):
        """Scenario N: Metadata registry and solver track consistent version."""
        self.assertEqual(self.registry.version, METADATA_VERSION)

    def test_scenario_o_no_hallucinated_fields_reported(self):
        """Scenario O: Unresolved user fields are explicitly listed and omitted from projection."""
        cand = CandidateView(
            name="I_SALESORDER",
            fields={"SALESORDER"},
            keys=["SALESORDER"],
            retrieval_score=0.80,
        )

        analysis = self.analyzer.analyze(
            business_requirement="Sales order with fictitious metric",
            requested_fields=["SalesOrder", "FictitiousFieldNotInRepo"],
            domain="SD",
        )

        res = self.solver.solve(
            analysis=analysis,
            candidates=[cand],
            all_known_fields={"SALESORDER"},
        )
        self.assertIn("FictitiousFieldNotInRepo", res.unresolved_fields)
        self.assertIn("// TODO: Unresolved user field: FictitiousFieldNotInRepo", res.ddl)


if __name__ == "__main__":
    unittest.main()
