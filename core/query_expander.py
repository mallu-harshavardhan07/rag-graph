"""Pre-Retrieval LLM Query Expander & Semantic Router for CDS RAG Architecture.

Extracts SAP architectural intent (ANALYTICAL, TRANSACTIONAL, MASTER_DATA),
separates dimensions from quantitative measures, detects aggregations,
and synthesizes dense technical search keywords for metadata-aware retrieval.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

from config import settings

logger = logging.getLogger("query_expander")
logging.basicConfig(level=logging.INFO)

SEMANTIC_ROUTER_SCHEMA = {
    "name": "SemanticRouterOutput",
    "schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": ["ANALYTICAL", "TRANSACTIONAL", "MASTER_DATA"],
                "description": "SAP CDS intent: ANALYTICAL (cubes/queries with measures/aggregations), TRANSACTIONAL (facts/line-items), or MASTER_DATA (dimensions/attributes).",
            },
            "dimensions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Business attributes and entity keys in standard SAP CamelCase (e.g., Customer, Plant, SalesOrder, Material).",
            },
            "measures": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Quantitative metrics and amounts in standard SAP CamelCase (e.g., NetAmount, TargetQuantity, TotalNetAmount).",
            },
            "aggregations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Identified aggregation functions (e.g., SUM, COUNT, AVG, MIN, MAX).",
            },
            "likely_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Likely SAP functional areas (e.g., SD, MM, FI, PP, LE).",
            },
            "search_keywords": {
                "type": "string",
                "description": "Dense keyword search string containing table names (VBAK, VBAP, BSEG, EKKO), view names, dimensions, measures, and architectural intent terms for vector similarity.",
            },
        },
        "required": ["intent", "dimensions", "measures", "aggregations", "likely_domains", "search_keywords"],
    },
}

# Alias for backwards compatibility
QUERY_EXPANSION_SCHEMA = SEMANTIC_ROUTER_SCHEMA


@dataclass
class QueryExpansion:
    """Semantic Router Output representing user's SAP intent and normalized query contract."""
    intent: str  # "ANALYTICAL" | "TRANSACTIONAL" | "MASTER_DATA"
    dimensions: List[str]
    measures: List[str]
    aggregations: List[str]
    likely_domains: List[str]
    search_keywords: str
    business_objects: List[str] = field(default_factory=list)
    grain: List[str] = field(default_factory=list)
    canonical_seed_views: List[str] = field(default_factory=list)
    raw_response: Optional[str] = ""

    @property
    def expanded_fields(self) -> List[str]:
        """Requested canonical fields: combined dimensions + measures."""
        return list(dict.fromkeys(self.dimensions + self.measures))

    @property
    def dense_search_query(self) -> str:
        """Alias for search_keywords for backwards compatibility."""
        return self.search_keywords

    @property
    def predicted_domain(self) -> str:
        """Primary likely domain."""
        return self.likely_domains[0] if self.likely_domains else "General"

    def to_dict(self) -> Dict[str, Any]:
        """Produce the exact JSON schema defined in System Directive."""
        return {
            "intent": self.intent,
            "dimensions": self.dimensions,
            "measures": self.measures,
            "aggregations": self.aggregations,
            "likely_domains": self.likely_domains,
            "search_keywords": self.search_keywords,
            "business_objects": self.business_objects,
            "grain": self.grain,
            "canonical_seed_views": self.canonical_seed_views,
            "expanded_fields": self.expanded_fields,
            "predicted_domain": self.predicted_domain,
            "dense_search_query": self.dense_search_query,
        }


# Alias for SemanticRouterOutput
SemanticRouterOutput = QueryExpansion


class QueryExpander:
    """Pre-Retrieval Semantic Router and Query Expander with strict SAP domain dictionary and deterministic fallback."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        default_model: Optional[str] = None,
    ):
        self.openai_key = openai_api_key or settings.OPENAI_API_KEY
        self.anthropic_key = anthropic_api_key or settings.ANTHROPIC_API_KEY
        self.model = default_model or settings.DEFAULT_MODEL

        self.openai_client = None
        self.anthropic_client = None

        if self.openai_key and self.openai_key not in ("your-api-key", ""):
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.openai_key)
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client in QueryExpander: {e}")

        if self.anthropic_key and self.anthropic_key not in ("your-api-key", ""):
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_key)
            except Exception as e:
                logger.warning(f"Could not initialize Anthropic client in QueryExpander: {e}")

    def expand(
        self,
        business_requirement: str,
        requested_fields: Optional[List[str]] = None,
    ) -> QueryExpansion:
        """Route query semantically and translate input into dimensions, measures, and search keywords."""
        requested_fields = requested_fields or []
        clean_fields = [f.strip() for f in requested_fields if f.strip()]

        # 1. Attempt LLM Semantic Router Expansion
        llm_result = self._call_llm(business_requirement, clean_fields)
        if llm_result:
            return llm_result

        # 2. Deterministic Enterprise SAP Knowledge-Base Fallback
        logger.info("Using deterministic SAP Knowledge-Base heuristic semantic routing.")
        return self._heuristic_expand(business_requirement, clean_fields)

    def _call_llm(self, requirement: str, fields: List[str]) -> Optional[QueryExpansion]:
        """Invoke configured LLM (OpenAI or Anthropic) with strict Semantic Router schema."""
        prompt = self._build_prompt(requirement, fields)
        system_content = (
            "You are a Principal SAP S/4HANA Solution Architect and Semantic Router. "
            "Analyze user requirements, classify intent into ANALYTICAL, TRANSACTIONAL, or MASTER_DATA, "
            "separate attributes into dimensions and quantitative measures, identify aggregations, "
            "and synthesize dense technical search keywords.\n\n"
            "STRICT CLASSIFICATION GUIDELINES:\n"
            "- ANALYTICAL: Query asks for aggregations, totals, sums, KPI calculations, cube views, or group-by reporting "
            "(e.g., 'Total Amount by Customer', 'Sum of net price by plant').\n"
            "- TRANSACTIONAL: Query asks for transactional documents, line items, orders, deliveries, invoices, or journal entries "
            "(e.g., 'Sales order line items with net price', 'Purchase order items').\n"
            "- MASTER_DATA: Query asks for pure master data or dimension entities without transactional documents "
            "(e.g., 'Customer master data with name and city', 'Supplier details').\n\n"
            "STRICT FIELD SEPARATION:\n"
            "- dimensions: Attributes, entity identifiers, organizational units, dates (e.g., Customer, Plant, SalesOrder, Material, CompanyCode, CityName).\n"
            "- measures: Numerical quantitative amounts, quantities, prices, values (e.g., NetAmount, TargetQuantity, TotalNetAmount, BillingQuantity).\n"
            "- aggregations: Functions such as ['SUM'], ['COUNT'], ['AVG'] if requested.\n\n"
            "Return strictly valid JSON conforming to the SemanticRouterOutput schema."
        )

        # Strategy 1: OpenAI
        if self.openai_client:
            try:
                model_name = "gpt-4o" if "gpt" in self.model.lower() else "gpt-4o-mini"
                logger.info(f"Routing query with OpenAI ({model_name})...")
                resp = self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                raw_text = resp.choices[0].message.content or ""
                return self._parse_json_response(raw_text, requirement)
            except Exception as e:
                logger.warning(f"OpenAI Semantic Router failed: {e}. Falling back...")

        # Strategy 2: Anthropic Claude
        if self.anthropic_client:
            try:
                logger.info(f"Routing query with Anthropic ({self.model})...")
                resp = self.anthropic_client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=system_content,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                raw_text = resp.content[0].text
                return self._parse_json_response(raw_text, requirement)
            except Exception as e:
                logger.warning(f"Anthropic Semantic Router failed: {e}. Falling back...")

        return None

    def _build_prompt(self, requirement: str, fields: List[str]) -> str:
        return f"""Analyze the following SAP business requirement and requested fields:

<business_requirement>
{requirement}
</business_requirement>

<requested_fields>
{json.dumps(fields)}
</requested_fields>

Extract the user's SAP intent, separate dimensions from measures, detect aggregations, and identify likely SAP domains.

Output strictly valid JSON with EXACTLY this structure:
{{
    "intent": "ANALYTICAL" | "TRANSACTIONAL" | "MASTER_DATA",
    "dimensions": ["..."],
    "measures": ["..."],
    "aggregations": ["..."],
    "likely_domains": ["..."],
    "search_keywords": "..."
}}
"""

    def _parse_json_response(self, text: str, original_req: str) -> Optional[QueryExpansion]:
        """Defensively extract and validate SemanticRouterOutput JSON."""
        try:
            cleaned = text.strip()
            if "```" in cleaned:
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
                else:
                    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

            data = json.loads(cleaned)
            intent = str(data.get("intent", "TRANSACTIONAL")).strip().upper()
            if intent not in ("ANALYTICAL", "TRANSACTIONAL", "MASTER_DATA"):
                intent = "TRANSACTIONAL"

            dims = [str(d).strip() for d in data.get("dimensions", []) if d]
            meas = [str(m).strip() for m in data.get("measures", []) if m]
            aggs = [str(a).strip().upper() for a in data.get("aggregations", []) if a]
            domains = [str(dom).strip().upper() for dom in data.get("likely_domains", []) if dom]
            keywords = str(data.get("search_keywords", "")).strip()
            bos = [str(b).strip() for b in data.get("business_objects", []) if b]
            if not bos:
                bos = [d for d in dims if any(d.startswith(prefix) for prefix in ["SalesOrder", "Customer", "PurchaseOrder", "Supplier", "Material", "Product", "AccountingDocument", "ManufacturingOrder"])]
            grain = [str(g).strip() for g in data.get("grain", []) if g]
            if not grain:
                grain = dims[:1] if dims else []

            # Infer canonical seed views based on domains and intent
            canonical_seeds = self._infer_canonical_seeds(domains, intent, original_req)

            return QueryExpansion(
                intent=intent,
                dimensions=dims,
                measures=meas,
                aggregations=aggs,
                likely_domains=domains,
                search_keywords=keywords,
                business_objects=bos,
                grain=grain,
                canonical_seed_views=canonical_seeds,
                raw_response=text,
            )
        except Exception as e:
            logger.warning(f"Failed to parse LLM SemanticRouterOutput JSON: {e}")
            return None

    def _heuristic_expand(self, requirement: str, fields: List[str]) -> QueryExpansion:
        """Deterministic Enterprise SAP Knowledge-Base heuristic semantic router with domain isolation."""
        req_lower = requirement.lower()
        fields_text = " ".join([f.lower() for f in fields])
        full_text = f"{req_lower} {fields_text}"

        # 1. SAP Canonical Field Synonyms Dictionary
        FIELD_SYNONYMS = {
            # MM
            "vendor": "Supplier",
            "supplier": "Supplier",
            "creditor": "Supplier",
            "po": "PurchaseOrder",
            "purchase_order": "PurchaseOrder",
            "po_number": "PurchaseOrder",
            "po_item": "PurchaseOrderItem",
            "purchasing_org": "PurchasingOrganization",
            "material": "Product",
            "material_number": "Product",
            "product": "Product",
            "plant": "Plant",
            "factory": "Plant",
            "site": "Plant",
            # PP
            "production_order": "ManufacturingOrder",
            "manufacturing_order": "ManufacturingOrder",
            "work_center": "WorkCenter",
            "routing": "Routing",
            # LE
            "outbound_delivery": "DeliveryDocument",
            "inbound_delivery": "DeliveryDocument",
            "shipment": "DeliveryDocument",
            "handling_unit": "HandlingUnit",
            # SD
            "customer": "Customer",
            "debtor": "Customer",
            "client": "Customer",
            "customer_name": "CustomerName",
            "customername": "CustomerName",
            "city": "CityName",
            "city_name": "CityName",
            "cityname": "CityName",
            "so": "SalesOrder",
            "sales_order": "SalesOrder",
            "salesorder": "SalesOrder",
            "sales_order_item": "SalesOrderItem",
            "salesorderitem": "SalesOrderItem",
            "sales_org": "SalesOrganization",
            "sold_to_party": "SoldToParty",
            "payer": "PayerParty",
            "ship_to_party": "ShipToParty",
            "invoice": "BillingDocument",
            "invoice_number": "BillingDocument",
            "billing_document": "BillingDocument",
            "date": "CreationDate",
            "invoice_date": "BillingDocumentDate",
            "billing_date": "BillingDocumentDate",
            # Measures
            "total_amount": "TotalNetAmount",
            "totalamount": "TotalNetAmount",
            "net_amount": "TotalNetAmount",
            "netamount": "TotalNetAmount",
            "amount": "TotalNetAmount",
            "gross_amount": "TotalGrossAmount",
            "grossamount": "TotalGrossAmount",
            "net_price": "NetPriceAmount",
            "netprice": "NetPriceAmount",
            "price": "NetPriceAmount",
            "tax": "TaxAmount",
            "tax_amount": "TaxAmount",
            "taxamount": "TaxAmount",
            "tax_val": "TaxAmount",
            "quantity": "BillingQuantity",
            "qty": "BillingQuantity",
            "order_quantity": "OrderQuantity",
            "target_quantity": "TargetQuantity",
            "targetquantity": "TargetQuantity",
            # FI
            "company": "CompanyCode",
            "company_code": "CompanyCode",
            "companycode": "CompanyCode",
            "fiscal_year": "FiscalYear",
            "fiscalyear": "FiscalYear",
            "year": "FiscalYear",
            "gl_account": "GLAccount",
            "account": "GLAccount",
            "cost_center": "CostCenter",
            "profit_center": "ProfitCenter",
            "currency": "TransactionCurrency",
            "posting_date": "PostingDate",
            "creation_date": "CreationDate",
        }

        # Known Measure Patterns
        MEASURE_PATTERNS = [
            "amount", "netamount", "totalnetamount", "totalamount", "grossamount",
            "price", "netprice", "netpriceamount", "tax", "taxamount",
            "quantity", "qty", "billingquantity", "targetquantity", "orderquantity",
            "cost", "yield", "balance", "value",
        ]

        # 2. Intent Classification
        is_analytical = bool(
            re.search(r'\b(total|sum|group\s+by|aggregat|average|avg|count|kpi|measure|cube|analytics?|metric)\b', full_text)
        )
        is_transactional = bool(
            re.search(r'\b(sales\s+order|purchase\s+order|order|item|items|so\b|po\b|delivery|invoice|billing|document|journal|posting|line)\b', full_text)
        )
        is_master_data = bool(
            re.search(r'\b(customer|supplier|vendor|material|product|plant|partner|bank|master)\b', full_text)
        )

        if is_analytical:
            intent = "ANALYTICAL"
        elif is_transactional:
            intent = "TRANSACTIONAL"
        elif is_master_data:
            intent = "MASTER_DATA"
        else:
            intent = "TRANSACTIONAL"

        # 3. Detect Aggregations
        aggregations: List[str] = []
        if re.search(r'\b(total|sum)\b', full_text):
            aggregations.append("SUM")
        if re.search(r'\bcount\b', full_text):
            aggregations.append("COUNT")
        if re.search(r'\b(average|avg)\b', full_text):
            aggregations.append("AVG")
        if re.search(r'\bmin\b', full_text):
            aggregations.append("MIN")
        if re.search(r'\bmax\b', full_text):
            aggregations.append("MAX")

        # 4. Domain Trigger Evaluation
        def has_trigger(pattern: str, text: str) -> bool:
            return bool(re.search(r'\b' + re.escape(pattern) + r'\b', text, re.IGNORECASE))

        score_mm = sum(1 for t in ["purchase order", "po", "supplier", "vendor", "purchasing", "procurement", "ekko", "ekpo"] if has_trigger(t, full_text))
        score_pp = sum(1 for t in ["production order", "manufacturing", "work center", "routing", "factory", "afko", "afpo"] if has_trigger(t, full_text))
        score_fi = sum(1 for t in ["journal entry", "gl", "ledger", "company code", "financial", "accounting document", "bkpf", "bseg", "acdoca"] if has_trigger(t, full_text))
        score_le = sum(1 for t in ["outbound delivery", "inbound delivery", "shipment", "handling unit", "likp", "lips", "vekp"] if has_trigger(t, full_text))
        score_sd = sum(1 for t in ["sales order", "so", "customer", "billing", "delivery", "invoice", "vbak", "vbap", "vbrk", "vbrp"] if has_trigger(t, full_text))

        domain_scores = [
            ("PP", score_pp),
            ("MM", score_mm),
            ("FI", score_fi),
            ("LE", score_le),
            ("SD", score_sd),
        ]
        sorted_domains = sorted(domain_scores, key=lambda x: x[1], reverse=True)
        top_domain = sorted_domains[0][0] if sorted_domains[0][1] > 0 else "SD"
        likely_domains = [d for d, s in sorted_domains if s > 0]
        if not likely_domains:
            likely_domains = [top_domain]

        # 5. Separate Dimensions from Measures
        dimensions: List[str] = []
        measures: List[str] = []

        for f in fields:
            f_norm = f.strip().lower()
            translated = FIELD_SYNONYMS.get(f_norm)
            if not translated:
                if any(c.isupper() for c in f[1:]):
                    translated = f
                else:
                    translated = "".join(part.capitalize() for part in re.split(r"[_\s]+", f_norm) if part) or f

            # Classify as measure or dimension
            is_meas = any(p in translated.lower() for p in MEASURE_PATTERNS)
            if is_meas:
                if translated not in measures:
                    measures.append(translated)
            else:
                if translated not in dimensions:
                    dimensions.append(translated)

        # If analytical intent but user didn't specify measure explicitly in input fields,
        # extract measure from business requirement
        if intent == "ANALYTICAL" and not measures:
            if any(w in req_lower for w in ["amount", "total", "net amount", "revenue", "price"]):
                measures.append("TotalNetAmount")
            elif any(w in req_lower for w in ["quantity", "qty", "volume"]):
                measures.append("BillingQuantity" if top_domain == "SD" else "TargetQuantity")

        # Ensure mandatory primary keys for transactional queries only if user provided no fields
        if intent == "TRANSACTIONAL" and not fields:
            if top_domain == "SD" and "SalesOrder" not in dimensions and not any(w in req_lower for w in ["customer", "master"]):
                dimensions.insert(0, "SalesOrder")
            elif top_domain == "MM" and "PurchaseOrder" not in dimensions:
                dimensions.insert(0, "PurchaseOrder")
            elif top_domain == "FI" and "CompanyCode" not in dimensions:
                dimensions.insert(0, "CompanyCode")
            elif top_domain == "PP" and "ManufacturingOrder" not in dimensions:
                dimensions.insert(0, "ManufacturingOrder")

        # 6. Infer Canonical Seed Views
        canonical_seed_views = self._infer_canonical_seeds(likely_domains, intent, requirement)

        # 7. Synthesize Search Keywords
        kw_parts = [
            requirement,
            " ".join(dimensions),
            " ".join(measures),
            " ".join(likely_domains),
            " ".join(canonical_seed_views),
        ]
        if intent == "ANALYTICAL":
            kw_parts.append("Analytics Cube Aggregation")
        elif intent == "TRANSACTIONAL":
            kw_parts.append("Transaction Item")
        elif intent == "MASTER_DATA":
            kw_parts.append("Master Data Dimension")

        search_keywords = " ".join(dict.fromkeys(" ".join(kw_parts).split()))

        # 8. Infer Business Objects and Grain
        KNOWN_BO_TYPES = [
            "SalesOrder", "Customer", "PurchaseOrder", "Supplier", "Product",
            "Material", "AccountingDocument", "CompanyCode", "ManufacturingOrder",
            "BillingDocument", "DeliveryDocument", "Plant", "GLAccount", "CostCenter",
        ]
        bos = [bo for bo in KNOWN_BO_TYPES if any(bo.lower() in d.lower() for d in dimensions)]
        if not bos and dimensions:
            bos = [dimensions[0]]
        grain = [dimensions[0]] if dimensions else []

        return QueryExpansion(
            intent=intent,
            dimensions=dimensions,
            measures=measures,
            aggregations=aggregations,
            likely_domains=likely_domains,
            search_keywords=search_keywords,
            business_objects=bos,
            grain=grain,
            canonical_seed_views=canonical_seed_views,
            raw_response="Generated via deterministic SAP Semantic Router heuristics.",
        )

    def _infer_canonical_seeds(self, domains: List[str], intent: str, requirement: str) -> List[str]:
        """Infer high-probability canonical standard views based on domains, intent, and requirement."""
        primary_domain = domains[0] if domains else "SD"
        req_lower = requirement.lower()

        if intent == "ANALYTICAL":
            if primary_domain == "SD":
                return ["I_SalesOrderItemCube", "I_SalesOrderItem"]
            elif primary_domain == "MM":
                return ["I_PurchaseOrderItemCube", "I_PurchaseOrderItem"]
            elif primary_domain == "FI":
                return ["I_JournalEntryItemCube", "I_JournalEntryItem"]
            elif primary_domain == "PP":
                return ["I_MfgOrderOperationCube", "I_ManufacturingOrder"]
            elif primary_domain == "LE":
                return ["I_OutboundDeliveryItem", "I_DeliveryDocument"]

        elif intent == "MASTER_DATA":
            if primary_domain == "SD":
                return ["I_Customer"]
            elif primary_domain == "MM":
                return ["I_Supplier"]
            elif primary_domain == "FI":
                return ["I_CompanyCode", "I_GLAccount"]
            elif primary_domain == "PP":
                return ["I_Plant"]
            elif primary_domain == "LE":
                return ["I_ShippingPoint"]

        # Default: TRANSACTIONAL
        if primary_domain == "SD":
            if any(w in req_lower for w in ["invoice", "billing", "bill", "tax"]):
                return ["I_BillingDocumentItem", "I_BillingDocument"]
            return ["I_SalesOrderItem", "I_SalesOrder"]
        elif primary_domain == "MM":
            return ["I_PurchaseOrderItem", "I_PurchaseOrder"]
        elif primary_domain == "FI":
            return ["I_JournalEntryItem", "I_GLAccountLineItem"]
        elif primary_domain == "PP":
            return ["I_ManufacturingOrder", "I_ManufacturingOrderItem"]
        elif primary_domain == "LE":
            return ["I_OutboundDeliveryItem", "I_DeliveryDocument"]

        return ["I_SalesOrderItem"]
