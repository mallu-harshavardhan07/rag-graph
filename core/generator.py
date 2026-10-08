"""LLM Mega-Prompt Generator for Modern SAP ABAP CDS View Entities.

Assembles graph-grounded prompts using primary views, supporting views, and
verified Knowledge Graph join conditions. Supports Anthropic, OpenAI, and
deterministic fallback generation for 100% Zero-Hallucination Joins.
"""

import os
import re
import logging
from typing import Dict, List, Any, Optional

from config import settings
from core.recommender import RecommendationResult, resolve_field_join_plan

logger = logging.getLogger("generator")
logging.basicConfig(level=logging.INFO)


class CDSGenerator:
    """Enterprise ABAP CDS View Entity generator with GraphRAG grounding."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        default_model: Optional[str] = None,
    ):
        self.openai_key = openai_api_key or settings.OPENAI_API_KEY
        self.anthropic_key = anthropic_api_key or settings.ANTHROPIC_API_KEY
        self.model = default_model or settings.DEFAULT_MODEL

        # Initialize clients lazily or if keys are present
        self.openai_client = None
        self.anthropic_client = None

        if self.openai_key and self.openai_key != "your-api-key":
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.openai_key)
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}")

        if self.anthropic_key and self.anthropic_key != "your-api-key":
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_key)
            except Exception as e:
                logger.warning(f"Could not initialize Anthropic client: {e}")

    def generate(
        self,
        new_view_name: str,
        business_requirement: str,
        domain: str,
        required_fields: List[str],
        recommendation: RecommendationResult,
    ) -> str:
        """Assemble the GraphRAG Mega-Prompt and invoke the LLM with strict join-tree plan."""
        new_view_name = self._sanitize_view_name(new_view_name)
        prompt = self.build_mega_prompt(
            new_view_name=new_view_name,
            business_requirement=business_requirement,
            domain=domain,
            required_fields=required_fields,
            recommendation=recommendation,
        )

        ddl_code = ""

        # Strategy 1: Anthropic Claude (e.g. claude-3-5-sonnet)
        if self.anthropic_client and "claude" in self.model.lower():
            try:
                logger.info(f"Generating DDL with Anthropic model {self.model}...")
                response = self.anthropic_client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    system="You are a Principal SAP ABAP Developer. Generate valid modern ABAP CDS View Entities.",
                    messages=[{"role": "user", "content": prompt}],
                )
                ddl_code = response.content[0].text
            except Exception as e:
                logger.warning(f"Anthropic generation error: {e}. Falling back...")

        # Strategy 2: OpenAI (e.g. gpt-4o)
        if not ddl_code and self.openai_client:
            try:
                logger.info("Generating DDL with OpenAI model...")
                model_name = "gpt-4o" if "gpt" in self.model.lower() else "gpt-4o"
                response = self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a Principal SAP ABAP Developer. Generate valid modern ABAP CDS View Entities."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                )
                ddl_code = response.choices[0].message.content
            except Exception as e:
                logger.warning(f"OpenAI generation error: {e}. Falling back...")

        # Strategy 3: Deterministic Zero-Hallucination Template Synthesis (Fallback)
        if not ddl_code:
            logger.info("Using Graph-grounded deterministic CDS entity synthesizer.")
            ddl_code = self._synthesize_from_graph(
                new_view_name=new_view_name,
                business_requirement=business_requirement,
                domain=domain,
                required_fields=required_fields,
                recommendation=recommendation,
            )

        return self.clean_ddl(ddl_code)

    def build_mega_prompt(
        self,
        new_view_name: str,
        business_requirement: str,
        domain: str,
        required_fields: List[str],
        recommendation: RecommendationResult,
    ) -> str:
        """Construct the prompt strictly enforcing the deterministic join-tree plan."""
        join_plan = getattr(recommendation, "join_plan", None) or {}
        prim_name = join_plan.get("primary_view", recommendation.primary_view or "I_SalesOrder")
        field_mapping = join_plan.get("field_mapping", {})
        req_associations = join_plan.get("required_associations", recommendation.supporting_views)
        unresolved_fields = join_plan.get("unresolved_fields", recommendation.uncovered_fields)

        # Format field mapping instructions
        mapping_lines = []
        for f, m in field_mapping.items():
            mapping_lines.append(f"  - Field '{f}' -> project as `{m['field_expr']}` (Source: {m['source_view']})")
        formatted_field_mapping = "\n".join(mapping_lines) if mapping_lines else "None"

        # Format associations
        assoc_lines = []
        for a in req_associations:
            assoc_lines.append(
                f"  - association {a.get('cardinality', '[0..1]')} to {a['target_view']} as {a['alias']}\n"
                f"    on {a['on_condition']}"
            )
        formatted_associations = "\n".join(assoc_lines) if assoc_lines else "No additional associations required."

        # Unresolved fields warning
        if unresolved_fields:
            unresolved_section = f"""<unresolved_fields_warning>
CRITICAL ZERO-HALLUCINATION ENFORCEMENT:
The following requested fields DO NOT exist in the Knowledge Graph or any connected view:
{unresolved_fields}
DO NOT project the following fields as they do not exist in the graph: {unresolved_fields}.
Do NOT invent tables or foreign keys for them.
</unresolved_fields_warning>"""
            rule_unresolved = f"DO NOT project the following fields as they do not exist in the graph: {unresolved_fields}"
        else:
            unresolved_section = ""
            rule_unresolved = "All requested fields are verified and mapped."

        prompt = f"""You are a Principal SAP ABAP Developer. Generate a modern SAP ABAP CDS View Entity.

<business_requirement>
{business_requirement}
</business_requirement>

<domain>{domain}</domain>

<join_plan>
Primary View: {prim_name} (as _Primary)

Verified Associations to Include:
{formatted_associations}

Deterministic Field Mapping (MANDATORY):
{formatted_field_mapping}
</join_plan>

{unresolved_section}

<strict_rules>
1. Generate valid modern ABAP CDS syntax: `define view entity {new_view_name} as select from {prim_name} as _Primary ...`
2. Include ONLY the verified associations and ON conditions provided in <join_plan>. Do NOT invent foreign keys or tables.
3. Field Aliasing Rule: Prefix EVERY projected field with the EXACT alias defined in the Deterministic Field Mapping (e.g. `_Primary.<field>` or `<alias>.<field>`).
4. {rule_unresolved}
5. Include header annotations:
   @EndUserText.label: '{business_requirement[:60]}'
   @AccessControl.authorizationCheck: #CHECK
6. Output ONLY the raw executable DDL source code. No explanations.
</strict_rules>
"""
        return prompt

    def _synthesize_from_graph(
        self,
        new_view_name: str,
        business_requirement: str,
        domain: str,
        required_fields: List[str],
        recommendation: RecommendationResult,
    ) -> str:
        """Deterministic zero-hallucination CDS view entity synthesizer using the join-tree plan.

        Strictly enforces:
        - NEVER falls back to _Primary for missing or unresolved fields.
        - Unresolved fields are omitted from the projection list.
        - Exact aliases from field_mapping are enforced.
        """
        join_plan = getattr(recommendation, "join_plan", None) or {}
        prim_view = join_plan.get("primary_view", recommendation.primary_view or "I_SalesOrder")
        field_mapping = join_plan.get("field_mapping", {})
        req_associations = join_plan.get("required_associations", recommendation.supporting_views)
        unresolved_fields = set(f.upper() for f in join_plan.get("unresolved_fields", recommendation.uncovered_fields))

        # Check primary view key
        prim_meta = recommendation.primary_view_metadata or {}
        raw_keys = [f["name"] for f in prim_meta.get("fields", []) if f.get("is_key")]
        valid_keys = []
        for k in raw_keys:
            m = re.search(r'([A-Za-z0-9_]+)\s*$', str(k).strip())
            k_name = m.group(1) if m else str(k).strip()
            if k_name.lower() not in ("select", "from", "where", "join", "on", "as", "case", "when"):
                valid_keys.append(k_name)

        lines = [
            f"@EndUserText.label: '{business_requirement[:60]}'",
            "@AccessControl.authorizationCheck: #CHECK",
            f"define view entity {new_view_name}",
            f"  as select from {prim_view} as _Primary",
        ]

        # Add associations
        for assoc in req_associations:
            target = assoc["target_view"]
            alias = assoc["alias"]
            card = assoc.get("cardinality", "[0..1]")
            cond = assoc.get("on_condition", "")
            lines.append(f"  association {card} to {target} as {alias}")
            lines.append(f"    on {cond}")

        lines.append("{")

        # Expose primary key if available and not already in projected fields
        if valid_keys:
            pk = valid_keys[0]
            if pk.upper() not in [rf.strip().upper() for rf in required_fields]:
                lines.append(f"  key _Primary.{pk} as {pk},")

        # Expose mapped fields
        for rf in required_fields:
            rf_clean = rf.strip()
            rf_upper = rf_clean.upper()

            # Skip unresolved fields (NEVER fall back to _Primary!)
            if rf_upper in unresolved_fields or rf_clean not in field_mapping:
                lines.append(f"  /* Field '{rf_clean}' omitted: unresolvable in Knowledge Graph */")
                continue

            mapping = field_mapping[rf_clean]
            lines.append(f"  {mapping['field_expr']},")

        # Expose associations in projection
        for assoc in req_associations:
            lines.append(f"  /* Association exposure */")
            lines.append(f"  {assoc['alias']},")

        # Clean trailing commas
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].endswith(","):
                lines[i] = lines[i][:-1]
                break

        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def _sanitize_view_name(name: str) -> str:
        name = re.sub(r'[^A-Za-z0-9_]', '_', name.strip()).upper()
        if not name.startswith(("Z", "Y")):
            name = f"Z_{name}"
        return name[:30]

    @staticmethod
    def clean_ddl(ddl_code: str) -> str:
        """Strip markdown codeblocks and leading/trailing chatter."""
        ddl = re.sub(r'^```[a-zA-Z]*\n', '', ddl_code.strip())
        ddl = re.sub(r'\n```$', '', ddl)
        return ddl.strip()
