"""Full-Stack Production-Grade SAP CDS View Recommender & Generator UI.

GraphRAG + Live ADT Agentic Loop Web Interface powered by Streamlit.
Refactored strictly with linear enterprise workflow and deterministic join-tree enforcement.
"""

import sys
import os
import time
import json
import streamlit as st

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from core.sap_adt_client import SAPADTClient
from core.ddl_parser import DDLParser
from core.database import StorageEngine
from core.recommender import ReusabilityEvaluator
from core.generator import CDSGenerator
from core.compiler_loop import ADTCompilerLoop
from scripts.ingest_all_views import run_full_crawler

# Page Configuration
st.set_page_config(
    page_title="SAP CDS View Recommender & Generator (GraphRAG)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Enterprise Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .badge-reuse {
        background-color: #107c41;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-extend {
        background-color: #0078d4;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-create {
        background-color: #d83b01;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-generate {
        background-color: #0078d4;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-verified {
        background-color: #13a10e;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-warning {
        background-color: #d83b01;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_system_components():
    """Initialize core engines singleton."""
    storage = StorageEngine()
    client = SAPADTClient()
    evaluator = ReusabilityEvaluator(storage_engine=storage)
    generator = CDSGenerator()
    compiler = ADTCompilerLoop(adt_client=client, generator=generator)
    return storage, client, evaluator, generator, compiler


storage, client, evaluator, generator, compiler = get_system_components()


# ==========================================
# SIDEBAR: SAP SYSTEM & ARCHITECTURE STATUS
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/59/SAP_2011_logo.svg", width=90)
    st.title("S/4HANA ADT GraphRAG")
    st.caption("Production AI Architecture Engine")

    st.markdown("---")
    st.subheader("📡 Live SAP ADT Status")
    st.write(f"**System ID:** `{settings.SAP_SYSTEM_ID}` (Client `{settings.SAP_CLIENT}`)")
    st.write(f"**Release:** S/4HANA {settings.SAP_S4HANA_RELEASE}")
    st.write(f"**Target Host:** `{settings.SAP_BASE_URL.split('://')[-1]}`")

    # Live Connection test button
    if st.button("🔄 Test Live ADT Ping"):
        with st.spinner("Pinging SAP ADT REST endpoint..."):
            try:
                token = client.fetch_csrf_token(force_refresh=True)
                st.success(f"Connected! CSRF: `{token[:10]}...`")
            except Exception as e:
                st.error(f"Connection error: {str(e)[:100]}")

    st.markdown("---")
    st.subheader("📚 Repository Knowledge Base")
    total_indexed = storage.get_total_indexed_count()
    st.metric(label="Indexed CDS Views", value=f"{total_indexed:,}")

    if st.button("⚡ Run Adaptive Prefix Crawler"):
        with st.spinner("Running dynamic BFS trie crawler on SAP ADT..."):
            cnt = run_full_crawler(max_cap=200, workers=5)
            st.success(f"Adaptive crawl finished! {cnt} total views indexed.")
            st.rerun()

    st.markdown("---")
    st.caption("Zero-Hallucination Joins via Knowledge Graph & Live ADT Auto-Healing.")


# ==========================================
# MAIN INTERFACE
# ==========================================
st.title("⚡ Production-Grade SAP CDS View Recommender & Generator")
st.markdown(
    "Dual **VectorDB (ChromaDB)** + **Knowledge Graph (Neo4j)** GraphRAG index connected to "
    "**SAP S/4HANA 2023 ADT** with **Live Compiler Closed-Loop Verification**."
)

tab1, tab2, tab3 = st.tabs([
    "🎯 CDS Reusability & Multi-View Generation",
    "🔍 Live SAP ADT Explorer & Compiler",
    "🕸️ Knowledge Graph & Topology",
])


# -------------------------------------------------------------
# TAB 1: REUSABILITY EVALUATION & MULTI-VIEW GENERATION
# -------------------------------------------------------------
with tab1:
    st.subheader("📋 CDS Entity Request Specification")

    # STRICT LINEAR TOP-TO-BOTTOM WORKFLOW (No template presets)
    with st.container():
        # 1. Select Domain
        domain = st.selectbox(
            "1. Select Domain:",
            [
                "SD (Sales & Distribution)",
                "MM (Materials Management / Sourcing)",
                "FI (Finance & Controlling)",
                "LE (Logistics Execution)",
                "PP (Production Planning)",
                "Cross-Functional / General",
            ],
            index=0,
            help="Business functional area for the CDS view entity.",
        )

        # 2. Required Output Fields
        raw_fields_input = st.text_input(
            "2. Required Output Fields (comma-separated):",
            value="SalesOrder, SalesOrderItem, NetPrice, Customer, CustomerName, CityName",
            help="Exact list of business fields required in the target CDS view projection list.",
        )

        # 3. Business Requirement Specification
        business_requirement = st.text_area(
            "3. Business Requirement Specification:",
            value="Need a composite CDS view exposing sales order items with item net price and customer location details.",
            height=100,
            help="Detailed functional description used for semantic vector retrieval against CDS signatures.",
        )

        # 4. Search Scope Filter
        scope_display = st.radio(
            "4. Search Scope Filter:",
            ["All Views", "Custom Views", "Standard Views"],
            index=0,
            horizontal=True,
            help="Filter search scope across custom (Z*/Y*) and standard (I_*/C_*) CDS repositories.",
        )
        scope_map = {
            "All Views": "ALL",
            "Custom Views": "CUSTOM_ONLY",
            "Standard Views": "STANDARD_ONLY",
        }
        scope = scope_map[scope_display]

        # 5. New View Entity Name
        view_entity_name = st.text_input(
            "5. New View Entity Name:",
            value="Z_SalesOrder_Customer_Analytics",
            help="Target CDS View Entity name complying with modern S/4HANA conventions.",
        )

        parsed_fields = [f.strip() for f in raw_fields_input.split(",") if f.strip()]

        # 6. Generate / Evaluate Button
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            eval_clicked = st.button("🔍 Evaluate Reusability", use_container_width=True, type="secondary")
        with col_btn2:
            gen_clicked = st.button("⚡ Generate & Verify CDS View Entity", use_container_width=True, type="primary")

    # EXECUTION & RESULTS DISPLAY
    if eval_clicked or gen_clicked or ("eval_result" in st.session_state and not eval_clicked):
        if eval_clicked or gen_clicked:
            with st.spinner("Evaluating reusability against ChromaDB signatures and Knowledge Graph..."):
                result = evaluator.evaluate(
                    business_requirement=business_requirement,
                    domain=domain,
                    required_fields=parsed_fields,
                    scope=scope,
                )
                st.session_state["eval_result"] = result

        result = st.session_state["eval_result"]

        st.markdown("---")
        st.subheader("📊 Reusability Decision & Routing")

        # PRE-RETRIEVAL SEMANTIC ROUTER & QUERY EXPANSION DISPLAY
        if getattr(result, "query_expansion", None):
            qe = result.query_expansion
            intent = qe.get("intent", "TRANSACTIONAL")
            dims = qe.get("dimensions", [])
            meas = qe.get("measures", [])
            aggs = qe.get("aggregations", [])
            likely_doms = qe.get("likely_domains", [qe.get("predicted_domain", "SD")])
            search_kws = qe.get("search_keywords", qe.get("dense_search_query", ""))

            with st.expander("🔍 Semantic Router & Metadata-Aware Intent Extraction", expanded=True):
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    st.write(f"**Extracted SAP Intent:** `{intent}`")
                    st.write(f"**Likely Domains:** {', '.join([f'`{d}`' for d in likely_doms])}")
                    st.write(f"**Dimensions:** {', '.join([f'`{d}`' for d in dims]) if dims else 'None'}")
                with col_exp2:
                    st.write(f"**Quantitative Measures:** {', '.join([f'`{m}`' for m in meas]) if meas else 'None'}")
                    st.write(f"**Aggregations:** {', '.join([f'`{a}`' for a in aggs]) if aggs else 'None'}")
                    st.caption(f"**Search Keywords:** `{search_kws[:140]}...`")

        # RECOMMENDATION BANNER (TRI-STATE DECISION ENGINE)
        if result.decision == "REUSE":
            st.markdown(
                f"""
                <div style="background-color: #0b3d1f; border-left: 6px solid #107c41; padding: 18px; border-radius: 8px; margin-bottom: 20px;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <span class="badge-reuse">100% REUSABLE MATCH (REUSE)</span>
                            <h3 style="color: #6cf7a2; margin: 8px 0 4px 0;">Existing Standard View: <code>{result.primary_view_name}</code></h3>
                            <p style="color: #d1e7dd; margin: 0;">{result.rationale}</p>
                            <small style="color: #6cf7a2;">✔ Deployable: True (Zero technical debt; direct reuse of active catalog view)</small>
                        </div>
                        <div style="text-align: right;">
                            <h2 style="color: #6cf7a2; margin: 0;">100%</h2>
                            <small style="color: #a3cfbb;">Field Coverage</small>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif result.decision == "EXTEND":
            sec_name = getattr(result, "secondary_view", "") or (result.supporting_views[0]["target_view"] if result.supporting_views else "Secondary")
            assoc_card = result.association.get("cardinality", "0..1") if getattr(result, "association", None) else "0..1"
            st.markdown(
                f"""
                <div style="background-color: #0a2540; border-left: 6px solid #0078d4; padding: 18px; border-radius: 8px; margin-bottom: 20px;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <span class="badge-extend">SAFE TWO-VIEW COMPOSITION (EXTEND)</span>
                            <h3 style="color: #70baff; margin: 8px 0 4px 0;">Primary: <code>{result.primary_view_name}</code> + Secondary: <code>{sec_name}</code> [{assoc_card}]</h3>
                            <p style="color: #cfe2ff; margin: 0;">{result.rationale}</p>
                            <small style="color: #70baff;">✔ Deployable: True (Verified safe association)</small>
                        </div>
                        <div style="text-align: right;">
                            <h2 style="color: #70baff; margin: 0;">100%</h2>
                            <small style="color: #9ec5fe;">Combined Coverage</small>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for w in result.warnings:
                st.warning(f"⚠️ {w}")
        else:  # CREATE
            st.markdown(
                f"""
                <div style="background-color: #3b1908; border-left: 6px solid #d83b01; padding: 18px; border-radius: 8px; margin-bottom: 20px;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <span class="badge-create">FALLBACK SYNTHESIS (CREATE)</span>
                            <h3 style="color: #ff9f73; margin: 8px 0 4px 0;">Partial Match Primary: <code>{result.primary_view_name}</code></h3>
                            <p style="color: #ffe0d1; margin: 0;">{result.rationale}</p>
                            <small style="color: #ff9f73;">⛔ Deployable: False (Automatic ADT deployment blocked - manual review required)</small>
                        </div>
                        <div style="text-align: right;">
                            <h2 style="color: #ff9f73; margin: 0;">{result.match_percentage}%</h2>
                            <small style="color: #ffb899;">Primary Coverage</small>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for w in result.warnings:
                st.warning(f"⚠️ {w}")

        # WHY DID THE AGENT DECIDE THIS? (DECISION TRACE EXPANDER - SECTION 26)
        decision_trace = getattr(result, "decision_trace", {}) or {}
        with st.expander("🧐 Why did the agent decide this? (Decision Trace)", expanded=True):
            st.markdown(
                f"**Final Decision:** `{result.decision}` | "
                f"**Reason Codes:** `{', '.join(getattr(result, 'reason_codes', [result.reason_code]))}` | "
                f"**Evidence Confidence:** `{result.confidence}`"
            )

            # 1. Canonical Field Mapping & Validation Status
            canon_map = decision_trace.get("canonical_field_mapping", [])
            if canon_map:
                st.write("##### 1. Canonical Field Mapping & Validation Status")
                st.dataframe(
                    [
                        {
                            "Requested Term": m.get("requested_term"),
                            "Canonical SAP Field": m.get("canonical_name"),
                            "Validation Status": "✅ Validated in Catalog" if m.get("validated") else "⚠️ Unvalidated Proposal",
                            "Resolution Source": m.get("source"),
                            "Confidence": m.get("confidence"),
                        }
                        for m in canon_map
                    ],
                    use_container_width=True,
                )

            # 2. Primary Candidates Pool
            prim_cands = decision_trace.get("primary_candidates", [])
            if prim_cands:
                st.write("##### 2. Evaluated Primary Candidates")
                st.dataframe(
                    [
                        {
                            "Candidate View": c.get("view"),
                            "Base Score": c.get("base_score"),
                            "Adjusted Score (VDM/Intent)": c.get("adjusted_score"),
                            "Field Coverage": f"{int(c.get('coverage', 0) * 100)}%",
                        }
                        for c in prim_cands[:5]
                    ],
                    use_container_width=True,
                )

            # 3. Missing Fields & Secondary Retrieval Pool
            missing_f = decision_trace.get("missing_fields", [])
            sec_cands = decision_trace.get("secondary_candidates", [])
            if missing_f:
                st.write(f"##### 3. Missing Fields from Primary: {', '.join([f'`{f}`' for f in missing_f])}")
                if sec_cands:
                    st.caption("Targeted Secondary Retrieval via Field -> View Reverse Index:")
                    st.dataframe(
                        [
                            {
                                "Secondary Candidate View": s.get("view"),
                                "Missing-Field Coverage": f"{int(s.get('missing_field_coverage', 0) * 100)}%",
                                "Retrieval Score": s.get("score"),
                            }
                            for s in sec_cands[:5]
                        ],
                        use_container_width=True,
                    )

            # 4. Evaluated View Pairs & Safety Gates
            pair_evals = decision_trace.get("pair_evaluations", [])
            if pair_evals:
                st.write("##### 4. Evaluated Two-View Compositions & Graph Safety Gates")
                st.dataframe(
                    [
                        {
                            "Primary View": p.get("primary"),
                            "Secondary View": p.get("secondary"),
                            "Combined Coverage": f"{int(p.get('combined_coverage', 0) * 100)}%",
                            "Direct Association": "✅ Yes" if p.get("association") else "❌ No",
                            "Cardinality": p.get("cardinality"),
                            "Join Keys": ", ".join(p.get("join_keys", [])) if p.get("join_keys") else "None",
                            "Gate Outcome": "🟢 ACCEPTED" if p.get("result") == "ACCEPTED" else "🔴 REJECTED",
                            "Reason Code": p.get("reason_code"),
                        }
                        for p in pair_evals
                    ],
                    use_container_width=True,
                )

        # DETERMINISTIC FIELD JOIN-TREE MAPPING TABLE
        st.write("#### 📑 Deterministic Field Join-Tree Plan")
        join_plan = getattr(result, "join_plan", {}) or {}

        # Strictly use expanded_fields if available to eliminate raw conversational field clutter
        if getattr(result, "query_expansion", None) and result.query_expansion.get("expanded_fields"):
            effective_fields = result.query_expansion["expanded_fields"]
        else:
            effective_fields = parsed_fields

        if not join_plan and getattr(result, "primary_view_name", None):
            try:
                from core.recommender import resolve_field_join_plan
                join_plan = resolve_field_join_plan(
                    primary_view=result.primary_view_name,
                    requested_fields=effective_fields,
                    graph=evaluator.storage.graph_store,
                )
                result.join_plan = join_plan
            except Exception as e:
                logger.warning(f"Could not reconstruct join_plan: {e}")
                join_plan = {}
        if not join_plan:
            join_plan = {}
        field_mapping = join_plan.get("field_mapping", {})
        unresolved = join_plan.get("unresolved_fields", [])

        # Display ONLY the canonical expanded fields in the Join-Tree Plan table
        mapping_rows = []
        for ef in effective_fields:
            if ef in field_mapping:
                m = field_mapping[ef]
                mapping_rows.append({
                    "Canonical Field": ef,
                    "Resolution Status": "✅ Resolved",
                    "Assigned Source View": m["source_view"],
                    "Enforced Alias": m["alias"],
                    "Deterministic Projection": m["field_expr"],
                })
            else:
                mapping_rows.append({
                    "Canonical Field": ef,
                    "Resolution Status": "❌ Unresolved in Graph",
                    "Assigned Source View": "None (Excluded)",
                    "Enforced Alias": "-",
                    "Deterministic Projection": "/* Omitted from projection */",
                })
        st.dataframe(mapping_rows, use_container_width=True)

        if unresolved:
            st.warning(f"⚠️ Zero-Hallucination Guard: The following fields do not exist in the graph and will NOT be projected: **{', '.join(unresolved)}**")

        # GRAPH TRAVERSAL JOINS
        if result.supporting_views:
            st.write("#### 🔗 Knowledge Graph Verified Associations & ON Conditions")
            st.caption("Exact join conditions and cardinalities resolved from the Knowledge Graph topology:")
            for s in result.supporting_views:
                st.markdown(
                    f"""
                    - **Target View:** `{s['target_view']}` (Alias: `{s['alias']}`)  
                      **Fields Covered:** `{', '.join(s.get('covered_fields', []))}`  
                      **ON Condition:** `{s['on_condition']}`  
                      **Cardinality:** `{s.get('cardinality', '[0..1]')}`
                    """
                )

        # GENERATION / COMPILER LOOP (ENFORCING TRI-STATE RULES)
        if result.decision == "REUSE":
            st.markdown("---")
            st.subheader("⚡ Action: Zero Code Generation Required (REUSE)")
            st.success(
                f"Existing view `{result.primary_view_name}` satisfies 100% of functional requirements. "
                "Per enterprise governance rules, DDL generation and ADT compiler deployment are strictly bypassed."
            )
        elif result.decision == "CREATE":
            st.markdown("---")
            st.subheader("⚡ Action: Best-Effort Informational DDL (CREATE - Non-Deployable)")
            st.info(
                "⚠️ **Deployable: False.** Synthesis required because fields could not be safely joined through verified 1..1 or 0..1 associations. "
                "Automatic deployment is disabled. The following best-effort DDL is provided for developer review only:"
            )
            st.code(result.ddl, language="abap")
        elif result.decision == "EXTEND":
            st.markdown("---")
            st.subheader("⚡ Action: Safe Two-View Composition (EXTEND)")
            if gen_clicked or "compilation_result" in st.session_state:
                if gen_clicked:
                    with st.spinner("Compiling and verifying wrapper CDS View Entity via S/4HANA ADT..."):
                        target_ddl = result.ddl or generator.generate(
                            new_view_name=view_entity_name,
                            business_requirement=business_requirement,
                            domain=domain,
                            required_fields=result.query_expansion.get("expanded_fields", parsed_fields) if getattr(result, "query_expansion", None) else parsed_fields,
                            recommendation=result,
                        )
                        comp_res = compiler.compile_and_heal(view_name=view_entity_name, initial_ddl=target_ddl)
                        st.session_state["compilation_result"] = comp_res

                comp_res = st.session_state.get("compilation_result")
                if comp_res:
                    st.write("#### ⚡ Live S/4HANA ADT Verified DDL Code")

            col_badge, col_stats = st.columns([1, 1])
            with col_badge:
                if comp_res.status == "VERIFIED":
                    st.markdown(
                        f"""
                        <div style="background-color: #0b3d1f; padding: 10px; border-radius: 6px; border: 1px solid #13a10e;">
                            <span class="badge-verified">✔ SAP COMPILER VERIFIED</span>
                            <strong style="color: #6cf7a2; margin-left: 10px;">0 Errors ({comp_res.total_latency_ms} ms)</strong>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif comp_res.status == "OFFLINE_VERIFIED":
                    st.markdown(
                        f"""
                        <div style="background-color: #0d2c4d; padding: 10px; border-radius: 6px; border: 1px solid #0078d4;">
                            <span class="badge-generate">✔ LOCAL CDS AST VERIFIED</span>
                            <strong style="color: #9ec5fe; margin-left: 10px;">S/4HANA 2023 Rules Validated</strong>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="background-color: #3b110a; padding: 10px; border-radius: 6px; border: 1px solid #d83b01;">
                            <span class="badge-warning">⚠ COMPILER REJECTED</span>
                            <span style="color: #ff9999; margin-left: 10px;">{comp_res.error_summary}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with col_stats:
                st.caption(f"Compiler Loop Iterations: **{comp_res.total_iterations}** | Total Verification Latency: **{comp_res.total_latency_ms} ms**")

            # Code Editor
            st.code(comp_res.final_code, language="abap")

            # Compiler Iteration Logs
            with st.expander("🔍 View Compiler Loop Diagnostic Trajectory"):
                for it in comp_res.iterations:
                    st.write(f"**Iteration {it.iteration_number}:** Status: `{'Clean' if it.is_valid else 'Rejected'}` | Action: *{it.action_taken}* | Latency: `{it.latency_ms} ms`")
                    if it.errors:
                        st.json(it.errors)


# -------------------------------------------------------------
# TAB 2: LIVE SAP ADT EXPLORER & COMPILER
# -------------------------------------------------------------
with tab2:
    st.subheader("🔍 Live SAP S/4HANA 2023 ADT Object Search")
    st.markdown("Query the active SAP dictionary repository in real-time via Eclipse ADT REST APIs.")

    col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
    with col_s1:
        adt_query = st.text_input("QuickSearch Pattern:", value="I_GRC*")
    with col_s2:
        obj_type = st.selectbox("Object Type:", ["DDLS/DF", "DDLS", "All Types"], index=0)
    with col_s3:
        max_res = st.number_input("Max Results:", min_value=5, max_value=200, value=20)

    if st.button("Query SAP ADT", type="primary"):
        with st.spinner("Querying SAP Informationsystem REST API..."):
            try:
                res_objs = client.search_cds_views(
                    query=adt_query,
                    max_results=max_res,
                    object_type=obj_type if obj_type != "All Types" else "",
                )
                st.session_state["live_search_results"] = res_objs
                st.success(f"Retrieved {len(res_objs)} object references from SAP S/4HANA.")
            except Exception as e:
                st.error(f"Failed querying SAP ADT: {e}")

    if "live_search_results" in st.session_state:
        res_list = st.session_state["live_search_results"]
        if res_list:
            view_names = [r["name"] for r in res_list]
            sel_view = st.selectbox("Select View to Inspect DDL Source:", view_names)

            if sel_view:
                matched_item = next(r for r in res_list if r["name"] == sel_view)
                st.write(f"**URI:** `{matched_item.get('uri')}` | **Package:** `{matched_item.get('package')}`")

                col_v1, col_v2 = st.columns([1, 1])
                with col_v1:
                    if st.button(f"📥 Fetch Raw DDL: {sel_view}"):
                        with st.spinner("Extracting DDL from SAP ADT..."):
                            ddl = client.get_ddl_source(sel_view, uri=matched_item.get("uri"))
                            st.session_state[f"ddl_{sel_view}"] = ddl

                if f"ddl_{sel_view}" in st.session_state:
                    source_code = st.session_state[f"ddl_{sel_view}"]
                    st.code(source_code, language="abap")

                    # Live Syntax Check against SAP ADT
                    with col_v2:
                        if st.button(f"⚡ Check Live Syntax on SAP ADT"):
                            with st.spinner("Executing syntax check on S/4HANA..."):
                                chk = client.syntax_check(sel_view, source_code, uri=matched_item.get("uri"))
                                if chk["is_valid"]:
                                    st.success(f"✔ SAP Compiler Clean! Latency: {chk['latency_ms']} ms")
                                else:
                                    st.warning(f"Compiler Messages ({chk['latency_ms']} ms):")
                                    st.json(chk["errors"] or chk["warnings"])


@st.cache_data(ttl=60)
def _get_cached_view_names() -> list:
    return sorted(list(storage.get_all_indexed_view_names()))

# -------------------------------------------------------------
# TAB 3: KNOWLEDGE GRAPH & TOPOLOGY
# -------------------------------------------------------------
with tab3:
    st.subheader("🕸️ CDS View Knowledge Graph Topology")
    st.markdown("Visual representation of views, exposed fields, and foreign associations.")

    views_available = _get_cached_view_names()
    if views_available:
        selected_view_for_graph = st.selectbox("Select View for Subgraph Traversal:", views_available, index=0)

        subgraph = storage.graph_store.get_view_subgraph(selected_view_for_graph, depth=2)
        st.write(f"**Nodes in Subgraph:** {len(subgraph['nodes'])} | **Edges:** {len(subgraph['edges'])}")

        vis_nodes = []
        for n in subgraph["nodes"]:
            is_view = n["type"] == "CDSView"
            vis_nodes.append({
                "id": n["id"],
                "label": n["label"],
                "color": "#0078d4" if is_view else "#107c41",
                "shape": "box" if is_view else "ellipse",
                "font": {"color": "#ffffff"},
            })

        vis_edges = []
        for e in subgraph["edges"]:
            vis_edges.append({
                "from": e["from"],
                "to": e["to"],
                "label": e["relation"],
                "arrows": "to",
                "color": {"color": "#888888"},
            })

        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
          <style type="text/css">
            #network {{
              width: 100%;
              height: 480px;
              background-color: #161b22;
              border: 1px solid #30363d;
              border-radius: 8px;
            }}
          </style>
        </head>
        <body>
        <div id="network"></div>
        <script type="text/javascript">
          var nodes = new vis.DataSet({json.dumps(vis_nodes)});
          var edges = new vis.DataSet({json.dumps(vis_edges)});
          var container = document.getElementById('network');
          var data = {{ nodes: nodes, edges: edges }};
          var options = {{
            physics: {{
              solver: 'forceAtlas2Based',
              forceAtlas2Based: {{ gravitationalConstant: -30, springLength: 80 }}
            }},
            nodes: {{ borderWidth: 2, shadow: true }},
            edges: {{ font: {{ color: '#cccccc', size: 10 }} }}
          }};
          var network = new vis.Network(container, data, options);
        </script>
        </body>
        </html>
        """
        st.components.v1.html(html_code, height=500)
    else:
        st.info("No views currently indexed.")
