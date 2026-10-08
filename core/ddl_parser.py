"""DDL Parser and Feature Extractor for SAP ABAP CDS Views.

Extracts structured metadata, annotations, fields, data sources, and associations
from raw ABAP CDS DDL source code. Supports both modern CDS View Entities
(define view entity) and classic CDS Views (define view).
"""

import re
from typing import Dict, List, Any, Optional


class DDLParser:
    """Robust AST / Regex parser for SAP CDS DDL source files."""

    @staticmethod
    def strip_comments(ddl_text: str) -> str:
        """Strip single-line and multi-line comments from DDL code."""
        # Multi-line /* ... */
        cleaned = re.sub(r'/\*.*?\*/', '', ddl_text, flags=re.DOTALL)
        # Single-line // ...
        cleaned = re.sub(r'//.*', '', cleaned)
        # Single-line -- ...
        cleaned = re.sub(r'--.*', '', cleaned)
        return cleaned

    @classmethod
    def parse(cls, ddl_code: str, fallback_name: Optional[str] = None) -> Dict[str, Any]:
        """Parse raw DDL code into structured CDS schema metadata."""
        if not ddl_code or not ddl_code.strip():
            return {
                "view_name": (fallback_name or "UNKNOWN").upper(),
                "sql_view_name": None,
                "view_type": "CUSTOM" if fallback_name and fallback_name.upper().startswith(("Z", "Y")) else "STANDARD",
                "description": "",
                "domain": "Cross-Functional",
                "base_data_sources": [],
                "fields": [],
                "associations": [],
                "raw_ddl": ddl_code,
            }

        cleaned = cls.strip_comments(ddl_code)

        # 1. View Name
        view_name = cls._extract_view_name(cleaned) or (fallback_name or "UNKNOWN")
        view_name = view_name.strip().upper()

        # 2. View Type
        view_type = "CUSTOM" if view_name.startswith(("Z", "Y")) else "STANDARD"

        # 3. SQL View Name (@AbapCatalog.sqlViewName)
        sql_view_name = cls._extract_annotation_value(cleaned, r'@AbapCatalog\.sqlViewName\s*:\s*[\'"]?([A-Za-z0-9_]+)[\'"]?')

        # 4. Description (@EndUserText.label)
        description = cls._extract_annotation_value(cleaned, r'@EndUserText\.label\s*:\s*[\'"]([^\'"]+)[\'"]')
        if not description:
            # Fallback description
            description = view_name.replace("_", " ").title()

        # 5. Extract Base Data Sources (select from ...)
        base_data_sources = cls._extract_base_data_sources(cleaned)

        # 6. Infer Domain
        domain = cls._infer_domain(view_name, description, base_data_sources, cleaned)

        # 7. Extract Associations
        associations = cls._extract_associations(cleaned)

        # 8. Extract Fields
        fields = cls._extract_fields(cleaned)

        return {
            "view_name": view_name,
            "sql_view_name": sql_view_name,
            "view_type": view_type,
            "description": description,
            "domain": domain,
            "base_data_sources": base_data_sources,
            "fields": fields,
            "associations": associations,
            "raw_ddl": ddl_code,
        }

    @staticmethod
    def _extract_annotation_value(text: str, pattern: str) -> Optional[str]:
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    @staticmethod
    def _extract_view_name(text: str) -> Optional[str]:
        # Match 'define [root] view entity <name>' or 'define [root] view <name>'
        match = re.search(
            r'define\s+(?:root\s+)?(?:transient\s+)?(?:custom\s+)?view\s+(?:entity\s+)?([A-Za-z0-9_/]+)',
            text,
            re.IGNORECASE,
        )
        return match.group(1) if match else None

    @staticmethod
    def _extract_base_data_sources(text: str) -> List[str]:
        sources = []
        # Match 'as select from <source> [as alias]'
        from_match = re.search(r'as\s+select\s+from\s+([A-Za-z0-9_]+)', text, re.IGNORECASE)
        if from_match:
            sources.append(from_match.group(1).upper())

        # Match any join sources: 'inner join <source>', 'left outer join <source>'
        join_matches = re.findall(
            r'(?:inner|left\s+outer|right\s+outer|join)\s+([A-Za-z0-9_]+)\s+as',
            text,
            re.IGNORECASE,
        )
        for jm in join_matches:
            src = jm.upper()
            if src not in sources:
                sources.append(src)

        return sources

    @staticmethod
    def _extract_associations(text: str) -> List[Dict[str, str]]:
        associations = []
        # Pattern for: association [cardinality] to <target_view> as <alias> on <on_condition>
        # Uses lookahead (?=(?:association|composition|\{)) to cleanly separate multiple associations
        assoc_pattern = re.compile(
            r'(?:association|composition)\s*(?:\[([^\]]+)\])?\s+(?:to|of)\s+([A-Za-z0-9_]+)\s+as\s+([A-Za-z0-9_]+)\s+on\s+([\s\S]*?)(?=(?:association|composition|\{))',
            re.IGNORECASE,
        )

        for match in assoc_pattern.finditer(text):
            card = match.group(1).strip() if match.group(1) else "0..1"
            if not card.startswith("["):
                card = f"[{card}]"
            target_view = match.group(2).strip().upper()
            alias = match.group(3).strip()
            on_cond = re.sub(r'\s+', ' ', match.group(4).strip())

            associations.append({
                "target_view": target_view,
                "alias": alias,
                "cardinality": card,
                "on_condition": on_cond,
            })

        return associations

    @classmethod
    def _extract_fields(cls, text: str) -> List[Dict[str, Any]]:
        fields = []
        # Locate the select list block enclosed between { and }
        select_block_match = re.search(r'\{([\s\S]*)\}\s*;?$', text)
        if not select_block_match:
            # Try finding first matching outer brace block
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start != -1 and brace_end > brace_start:
                body = text[brace_start + 1 : brace_end]
            else:
                body = ""
        else:
            body = select_block_match.group(1)

        if not body:
            return fields

        # Split items by comma while taking care not to split inside parenthesis or brackets
        raw_items = cls._split_field_expressions(body)

        for item in raw_items:
            item = item.strip()
            if not item:
                continue

            # Skip standalone association exposures like `_Customer,` or `_Item`
            if item.startswith("_") and " as " not in item.lower() and "." not in item:
                continue

            # Skip header-level annotations in field list
            if item.startswith("@") and not any(k in item for k in ["key", "as", "cast"]):
                continue

            is_key = bool(re.search(r'\bkey\b', item, re.IGNORECASE))
            clean_item = re.sub(r'\bkey\b', '', item, flags=re.IGNORECASE).strip()

            # Remove field annotations if any (e.g., @Semantics...)
            clean_item = re.sub(r'@[A-Za-z0-9_.]+(?:\s*\([^)]*\))?', '', clean_item).strip()

            # Check for 'as <alias>'
            as_match = re.search(r'\bas\s+([A-Za-z0-9_]+)\b', clean_item, re.IGNORECASE)
            if as_match:
                alias_name = as_match.group(1)
                expr = clean_item[: as_match.start()].strip()
            else:
                alias_name = clean_item.split(".")[-1].strip()
                expr = clean_item

            # Remove casting if present to find data element or raw name
            data_element = ""
            cast_match = re.search(r'cast\s*\([^as]+\bas\s+([A-Za-z0-9_]+)', clean_item, re.IGNORECASE)
            if cast_match:
                data_element = cast_match.group(1)

            # Clean name
            field_name = alias_name.strip(" ;")
            # If name still contains invalid characters, extract identifier
            id_match = re.match(r'^[A-Za-z0-9_]+', field_name)
            if id_match:
                field_name = id_match.group(0)

            if field_name and not field_name.startswith(("@", "/*", "//")):
                fields.append({
                    "name": field_name,
                    "alias": alias_name,
                    "is_key": is_key,
                    "data_element": data_element,
                    "expression": expr,
                })

        return fields

    @staticmethod
    def _split_field_expressions(body: str) -> List[str]:
        """Split select list by comma respecting parenthesis and brackets nesting."""
        items = []
        current = []
        paren_depth = 0
        bracket_depth = 0

        for char in body:
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth = max(0, paren_depth - 1)
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif char == "," and paren_depth == 0 and bracket_depth == 0:
                item = "".join(current).strip()
                if item:
                    items.append(item)
                current = []
                continue
            current.append(char)

        last_item = "".join(current).strip()
        if last_item:
            items.append(last_item)

        return items

    @staticmethod
    def _infer_domain(
        view_name: str,
        description: str,
        data_sources: List[str],
        raw_code: str,
    ) -> str:
        """Infer business domain (SD, MM, FI, PP, etc.) from names and annotations."""
        combined = f"{view_name} {description} {' '.join(data_sources)} {raw_code}".upper()

        if any(w in combined for w in ["PURCHASE", "EKKO", "EKPO", "VENDOR", "LIFNR", "PO_", "MATERIAL", "MARA"]):
            return "MM (Materials Management / Sourcing)"
        if any(w in combined for w in ["SALES", "VBAK", "VBAP", "CUSTOMER", "KUNNR", "BILLING", "INVOICE", "SO_"]):
            return "SD (Sales & Distribution)"
        if any(w in combined for w in ["ACCOUNTING", "BKPF", "BSEG", "FINANCE", "GL", "COMPANY", "BUKRS", "PAYMENT"]):
            return "FI (Finance & Controlling)"
        if any(w in combined for w in ["DELIVERY", "LIKP", "LIPS", "SHIPPING"]):
            return "LE (Logistics Execution)"
        if any(w in combined for w in ["PRODUCTION", "AFKO", "AFPO", "ROUTING", "PLANT"]):
            return "PP (Production Planning)"

        return "Cross-Functional / General"

    @staticmethod
    def generate_compact_signature(view_metadata: Dict[str, Any]) -> str:
        """Generate token-safe compact semantic signature for embedding models.

        Keeps under 256 tokens for models like all-MiniLM-L6-v2:
        View: {view_name} | Type: {view_type} | Domain: {domain} | Label: {description}
        Fields: {', '.join(field_names)}
        """
        v_name = view_metadata.get("view_name", "UNKNOWN")
        v_type = view_metadata.get("view_type", "STANDARD")
        dom = view_metadata.get("domain", "General")
        desc = view_metadata.get("description", "")
        f_names = [f["name"] for f in view_metadata.get("fields", [])]
        return (
            f"View: {v_name} | Type: {v_type} | Domain: {dom} | Label: {desc}\n"
            f"Fields: {', '.join(f_names)}"
        )
