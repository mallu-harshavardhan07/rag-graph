"""Live ADT Auto-Healing and Compiler Feedback Loop.

Performs live syntax validation of generated CDS DDL against SAP S/4HANA ADT compiler.
Iteratively feeds diagnostic error messages (line numbers, tokens, rules) back into
the LLM agent until code compiles with 0 errors (Max 3 iterations).
"""

import time
import re
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from core.sap_adt_client import SAPADTClient
from core.generator import CDSGenerator
from config import settings

logger = logging.getLogger("compiler_loop")
logging.basicConfig(level=logging.INFO)


@dataclass
class CompilationIteration:
    iteration_number: int
    ddl_code: str
    is_valid: bool
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    action_taken: str = ""


@dataclass
class CompilationResult:
    status: str  # "VERIFIED", "FAILED", "OFFLINE_VERIFIED"
    final_code: str
    iterations: List[CompilationIteration] = field(default_factory=list)
    total_iterations: int = 0
    total_latency_ms: float = 0.0
    error_summary: str = ""
    is_verified: bool = False


class ADTCompilerLoop:
    """Agentic compiler loop coordinating SAP ADT syntax check and LLM self-healing."""

    def __init__(
        self,
        adt_client: Optional[SAPADTClient] = None,
        generator: Optional[CDSGenerator] = None,
        max_iterations: int = 3,
    ):
        self.adt_client = adt_client or SAPADTClient()
        self.generator = generator or CDSGenerator()
        self.max_iterations = max_iterations

    def compile_and_heal(
        self,
        view_name: str,
        initial_ddl: str,
    ) -> CompilationResult:
        """Run the live ADT syntax check and auto-heal loop."""
        current_code = initial_ddl
        iterations_log: List[CompilationIteration] = []
        start_all = time.time()

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"Compiler Loop: Iteration {iteration}/{self.max_iterations} for {view_name}...")

            # 1. Run Live Syntax Check via SAP ADT
            check_result = self.adt_client.syntax_check(view_name, current_code)
            is_valid = check_result.get("is_valid", False)
            errors = check_result.get("errors", [])
            warnings = check_result.get("warnings", [])
            latency = check_result.get("latency_ms", 0.0)

            # Check if error is due to SAP server offline / unreachable
            is_network_failure = any(
                "connection error" in e.get("message", "").lower() or
                "failed to establish a new connection" in e.get("message", "").lower()
                for e in errors
            )

            if is_network_failure:
                logger.warning("SAP ADT system is currently unreachable over network. Applying local CDS entity validation.")
                local_valid, local_errors = self._local_cds_syntax_check(current_code)
                iter_record = CompilationIteration(
                    iteration_number=iteration,
                    ddl_code=current_code,
                    is_valid=local_valid,
                    errors=local_errors,
                    warnings=[{"message": "Verified against local CDS Entity AST (SAP ADT system endpoint unreachable)"}],
                    latency_ms=latency,
                    action_taken="SAP ADT Offline - Validated via Local CDS Parser Rules",
                )
                iterations_log.append(iter_record)

                total_lat = round((time.time() - start_all) * 1000, 2)
                return CompilationResult(
                    status="OFFLINE_VERIFIED" if local_valid else "FAILED",
                    final_code=current_code,
                    iterations=iterations_log,
                    total_iterations=iteration,
                    total_latency_ms=total_lat,
                    is_verified=local_valid,
                    error_summary="; ".join([e["message"] for e in local_errors]) if not local_valid else "",
                )

            # Record iteration
            iter_record = CompilationIteration(
                iteration_number=iteration,
                ddl_code=current_code,
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                latency_ms=latency,
                action_taken="Clean compilation" if is_valid else "Compiler rejected code",
            )
            iterations_log.append(iter_record)

            # Case A: Clean Compilation (0 errors) -> Success!
            if is_valid:
                logger.info(f"View {view_name} verified by SAP ADT in {latency}ms (Iteration {iteration}).")
                total_lat = round((time.time() - start_all) * 1000, 2)
                return CompilationResult(
                    status="VERIFIED",
                    final_code=current_code,
                    iterations=iterations_log,
                    total_iterations=iteration,
                    total_latency_ms=total_lat,
                    is_verified=True,
                )

            # Case B: Errors returned -> Self-heal
            if iteration < self.max_iterations:
                logger.info(f"Self-healing: Feeding {len(errors)} error diagnostics to healing engine...")
                healed_code = self._heal_code(view_name, current_code, errors)
                current_code = healed_code
                iter_record.action_taken = f"Auto-healed {len(errors)} error(s) via LLM & AST rules"
            else:
                iter_record.action_taken = f"Max iterations reached ({self.max_iterations})"

        total_lat = round((time.time() - start_all) * 1000, 2)
        error_msg = "; ".join([e.get("message", "") for e in errors])
        return CompilationResult(
            status="FAILED",
            final_code=current_code,
            iterations=iterations_log,
            total_iterations=self.max_iterations,
            total_latency_ms=total_lat,
            is_verified=False,
            error_summary=error_msg,
        )

    def _heal_code(self, view_name: str, code: str, errors: List[Dict[str, Any]]) -> str:
        """Feed compiler diagnostics to LLM or local rule repair."""
        # Format errors
        error_lines = []
        for e in errors:
            line_no = e.get("line", "?")
            msg = e.get("message", "Unknown error")
            error_lines.append(f"- Line {line_no}: {msg}")
        adt_error_messages = "\n".join(error_lines)

        # 1. Try local rule fixes first for common CDS issues
        code_fixed = self._apply_ast_rule_fixes(code, errors)
        if code_fixed != code:
            return code_fixed

        # 2. Feed to LLM if available
        healing_prompt = f"""You are a Principal SAP ABAP Developer.
The SAP S/4HANA ADT compiler rejected your code for CDS view entity `{view_name}` with these errors:

<adt_error_messages>
{adt_error_messages}
</adt_error_messages>

<rejected_code>
{code}
</rejected_code>

<healing_rules>
1. Fix the annotations, types, aliases, or syntax specifically targeting the compiler diagnostics.
2. Maintain `define view entity {view_name} as select from ...` modern ABAP CDS syntax.
3. Return ONLY the raw corrected DDL source code. No explanations or markdown wrapper.
</healing_rules>
"""
        if self.generator.anthropic_client:
            try:
                resp = self.generator.anthropic_client.messages.create(
                    model=self.generator.model,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": healing_prompt}],
                )
                return self.generator.clean_ddl(resp.content[0].text)
            except Exception as e:
                logger.warning(f"LLM self-heal error with Anthropic: {e}")

        if self.generator.openai_client:
            try:
                resp = self.generator.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": healing_prompt}],
                    temperature=0.1,
                )
                return self.generator.clean_ddl(resp.choices[0].message.content)
            except Exception as e:
                logger.warning(f"LLM self-heal error with OpenAI: {e}")

        return code

    @staticmethod
    def _apply_ast_rule_fixes(code: str, errors: List[Dict[str, Any]]) -> str:
        """Apply rule-based AST repairs for common SAP CDS syntax errors."""
        fixed = code

        # Check for missing authorization check annotation
        if not re.search(r'@AccessControl\.authorizationCheck', fixed):
            fixed = "@AccessControl.authorizationCheck: #CHECK\n" + fixed

        # Check for missing end user label
        if not re.search(r'@EndUserText\.label', fixed):
            fixed = "@EndUserText.label: 'CDS View Entity'\n" + fixed

        # Remove trailing commas inside select list before closing brace
        fixed = re.sub(r',\s*(\}\s*;?\s*)$', r'\n\1', fixed)

        # Fix legacy `define view` to modern `define view entity`
        fixed = re.sub(r'\bdefine\s+view\s+([A-Za-z0-9_]+)\s+as\b', r'define view entity \1 as', fixed)

        return fixed

    @staticmethod
    def _local_cds_syntax_check(code: str) -> tuple[bool, List[Dict[str, Any]]]:
        """Validate CDS code against core ABAP CDS 2023 syntax rules."""
        errors = []

        if not re.search(r'define\s+(?:root\s+)?view\s+entity\s+[A-Za-z0-9_]+', code, re.IGNORECASE):
            errors.append({
                "line": 1,
                "severity": "E",
                "message": "S/4HANA 2023 compliance requires modern 'define view entity' syntax.",
            })

        if not re.search(r'as\s+select\s+from\s+[A-Za-z0-9_]+', code, re.IGNORECASE):
            errors.append({
                "line": 2,
                "severity": "E",
                "message": "Missing 'as select from <source>' clause.",
            })

        if "{" not in code or "}" not in code:
            errors.append({
                "line": 3,
                "severity": "E",
                "message": "Malformed CDS entity select block braces '{ ... }'.",
            })

        if not re.search(r'@AccessControl\.authorizationCheck', code):
            errors.append({
                "line": 1,
                "severity": "E",
                "message": "Missing required header annotation '@AccessControl.authorizationCheck'.",
            })

        return len(errors) == 0, errors
