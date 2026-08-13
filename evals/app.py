# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: logfire must be configured before all other imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys

from dotenv import find_dotenv

print("ENV FILE:", find_dotenv())
print("LOGFIRE_TOKEN:", repr(os.getenv("LOGFIRE_TOKEN")))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(override=True)

import logfire

logfire.configure(token=os.getenv("LOGFIRE_TOKEN"), service_name="evals", send_to_logfire=False)
print(os.getenv("LOGFIRE_TOKEN"))
# ─────────────────────────────────────────────────────────────────────────────
import asyncio

try:
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
except Exception:
    pass

import nest_asyncio
import pandas as pd
import streamlit as st

try:
    nest_asyncio.apply()
except Exception:
    pass


from evals.guardrails_eval import compute_guardrails_metrics, run_guardrails_eval
from evals.metrics import (
    clear_metrics_checkpoint,
    load_metrics_checkpoint,
    run_all_metrics,
)
from evals.pipeline import load_golden_dataset, run_pipeline, save_results

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise RAG — Eval Suite",
    page_icon="🧪",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
SCORE_COLORS = {
    "green": "#d4edda",
    "yellow": "#fff3cd",
    "red": "#f8d7da",
}


def _badge(score: float) -> str:
    if score >= 0.75:
        return "🟢"
    elif score >= 0.5:
        return "🟡"
    return "🔴"


def _grade(score: float) -> str:
    if score >= 0.75:
        return "✅ Good"
    elif score >= 0.5:
        return "⚠️ Fair"
    return "❌ Poor"


def _color_score(val):
    if not isinstance(val, (int, float)):
        return ""
    if val >= 0.75:
        return f"background-color: {SCORE_COLORS['green']}"
    elif val >= 0.5:
        return f"background-color: {SCORE_COLORS['yellow']}"
    return f"background-color: {SCORE_COLORS['red']}"


def _render_metric_table(df: pd.DataFrame, metric_col: str, title: str):
    avg = df[metric_col].mean()
    st.markdown(f"**{title}** — AVG: {_badge(avg)} `{avg:.2f}` {_grade(avg)}")
    styled = df.style.applymap(_color_score, subset=[metric_col]).format({metric_col: "{:.3f}"})
    st.dataframe(styled, width="stretch", hide_index=True)


def _run_async(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────
golden = load_golden_dataset()
st.session_state.golden = golden

rag_completed = sum(
    1
    for s in golden.get("rag_samples", [])
    if bool(
        s.get("actual_response") and s.get("actual_response").strip() and not s.get("actual_response").startswith("⚠️")
    )
)
total_rag = len(golden.get("rag_samples", []))

guardrails_completed = sum(
    1
    for g in golden.get("guardrails_samples", [])
    if g.get("actual_blocked") is not None and g.get("result") in ["TP", "TN", "FP", "FN"]
)
total_guardrails = len(golden.get("guardrails_samples", []))

has_saved_responses = rag_completed > 0

if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = has_saved_responses
if "enriched_dataset" not in st.session_state or st.session_state.enriched_dataset is None:
    st.session_state.enriched_dataset = golden if has_saved_responses else None
if "guardrails_results" not in st.session_state or st.session_state.guardrails_results is None:
    st.session_state.guardrails_results = golden.get("guardrails_samples") if guardrails_completed > 0 else None
if "metric_results" not in st.session_state:
    saved_mr = load_metrics_checkpoint()
    st.session_state.metric_results = saved_mr if saved_mr else None
if "pipeline_rows" not in st.session_state:
    st.session_state.pipeline_rows = []

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🧪 Enterprise RAG — Evaluation Suite")
st.caption("Step 1: Review ground truth → Step 2: Run live pipeline → Step 3: Score with RAGAS")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 Step 1 — Ground Truth", "🚀 Step 2 — Live Pipeline", "📊 Step 3 — Eval Metrics"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Ground Truth
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Ground Truth Dataset")
    st.markdown(
        "These are the **golden Q&A pairs** built by parsing your real enterprise documents. "
        "Each entry has a question, a reference answer (ground truth), and the expected tool the RAG agent should call."
    )

    rag_rows = []
    for s in golden["rag_samples"]:
        rag_rows.append(
            {
                "ID": s["id"],
                "Domain": s["domain"].replace("_", " ").title(),
                "Question": s["question"],
                "Reference Answer": s["reference"][:120] + "..." if len(s["reference"]) > 120 else s["reference"],
                "Expected Tool": s["expected_tools"][0] if s["expected_tools"] else "—",
            }
        )
    df_golden = pd.DataFrame(rag_rows)
    st.dataframe(df_golden, width="stretch", hide_index=True)
    st.caption(f"✅ {len(rag_rows)} golden RAG samples from 5 enterprise docs")

    st.divider()

    st.subheader("Guardrails Test Cases")
    st.markdown(
        "These inputs test whether the safety rails correctly **block adversarial inputs** "
        "and **let through legitimate questions**."
    )

    g_rows = []
    for g in golden["guardrails_samples"]:
        expected_label = "🛡️ Block" if g["expected_blocked"] else "✅ Pass"
        g_rows.append(
            {
                "ID": g["id"],
                "Input": g["input"],
                "Expected": expected_label,
                "Type": g["type"],
                "Description": g["description"],
            }
        )
    st.dataframe(pd.DataFrame(g_rows), width="stretch", hide_index=True)
    st.caption("6 guardrails test cases: 3 adversarial (should block) + 3 legit (should pass)")

    with st.expander("View raw golden_dataset.json"):
        st.json(golden)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Pipeline
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Live Pipeline — Collect Real Responses")
    st.markdown(
        "Sends each golden question to your **running FastAPI app** (`localhost:8000/query`). "
        "Captures the actual response, retrieved contexts, and tool called. "
        "Results are **incrementally saved to disk after every question & guardrail test**."
    )

    if rag_completed > 0 or guardrails_completed > 0:
        st.info(
            f"💾 **Current Progress on Disk**: `{rag_completed}/{total_rag}` RAG queries and `{guardrails_completed}/{total_guardrails}` Guardrails tests completed. "
            "Click **'▶️ Run / Resume Live Pipeline'** to resume from the next unanswered question (already completed items are instantly skipped with 0 API calls).",
            icon="ℹ️",
        )
    else:
        st.info(
            "⚠️ Make sure your FastAPI backend is running first: `uvicorn app.main:app --reload --port 8000`",
            icon="⚠️",
        )

    col_p1, col_p2, col_p3 = st.columns([1, 1, 1])
    run_pipeline_btn = col_p1.button(
        "▶️ Run / Resume Live Pipeline",
        type="primary",
        width="stretch",
    )
    reset_btn = col_p2.button(
        "🔄 Reset Everything & Start From Scratch",
        width="stretch",
    )

    if reset_btn:
        for s in st.session_state.golden.get("rag_samples", []):
            s["actual_response"] = ""
            s["actual_contexts"] = []
            s["actual_tools_called"] = []
        for g in st.session_state.golden.get("guardrails_samples", []):
            g["actual_blocked"] = None
            g["result"] = None
        save_results(st.session_state.golden, os.path.join(os.path.dirname(__file__), "golden_dataset.json"))
        clear_metrics_checkpoint()
        st.session_state.pipeline_done = False
        st.session_state.enriched_dataset = None
        st.session_state.guardrails_results = None
        st.session_state.metric_results = None
        st.session_state.pipeline_rows = []
        st.rerun()

    if run_pipeline_btn:
        st.session_state.pipeline_rows = []
        progress_bar = st.progress(0, text="Starting pipeline...")
        live_table_slot = st.empty()
        status_slot = st.empty()

        def pipeline_cb(i, total, question, stage, response=""):
            pct = int((i / total) * 100)
            if stage == "calling":
                progress_bar.progress(pct, text=f"[{i + 1}/{total}] Calling /query: {question[:60]}...")
            elif stage == "cooldown":
                progress_bar.progress(pct, text=question)
                status_slot.info(question)
            elif stage == "cached":
                short_q = question[:55] + "..." if len(question) > 55 else question
                short_r = response[:80] + "..." if len(response) > 80 else response
                st.session_state.pipeline_rows.append(
                    {
                        "#": i + 1,
                        "Question": short_q,
                        "Live Response (truncated)": short_r,
                        "Status": "⏩ Saved",
                    }
                )
                live_table_slot.dataframe(
                    pd.DataFrame(st.session_state.pipeline_rows),
                    width="stretch",
                    hide_index=True,
                )
                progress_bar.progress(
                    int(((i + 1) / total) * 100),
                    text=f"[{i + 1}/{total}] ⏩ Reusing saved response...",
                )
            else:
                short_q = question[:55] + "..." if len(question) > 55 else question
                short_r = response[:80] + "..." if len(response) > 80 else response
                st.session_state.pipeline_rows.append(
                    {
                        "#": i + 1,
                        "Question": short_q,
                        "Live Response (truncated)": short_r if short_r else "⚠️ No response",
                        "Status": "✅" if short_r else "❌",
                    }
                )
                live_table_slot.dataframe(
                    pd.DataFrame(st.session_state.pipeline_rows),
                    width="stretch",
                    hide_index=True,
                )
                progress_bar.progress(
                    int(((i + 1) / total) * 100),
                    text=f"[{i + 1}/{total}] ✅ Done",
                )

        with logfire.span("🚀 Streamlit — Run Pipeline Button"):
            enriched = run_pipeline(golden, progress_callback=pipeline_cb, force_refresh=True)
            st.session_state.enriched_dataset = enriched

        progress_bar.progress(100, text="✅ All responses collected!")
        status_slot.success(f"💾 {len(enriched['rag_samples'])} responses stored in session.")

        # ── Guardrails tests ──────────────────────────────────────────────────
        st.divider()
        st.subheader("Guardrails Tests")
        g_progress = st.progress(0, text="Running guardrails tests...")
        g_status_slot = st.empty()

        def g_cb(i, total, input_text, stage="calling"):
            pct = int((i / total) * 100)
            if stage == "cooldown":
                g_progress.progress(pct, text=input_text)
                g_status_slot.info(input_text)
            elif stage == "cached":
                g_progress.progress(
                    int(((i + 1) / total) * 100),
                    text=f"[{i + 1}/{total}] ⏩ Reusing saved guardrail result...",
                )
            else:
                g_progress.progress(
                    pct,
                    text=f"[{i + 1}/{total}] Testing: {input_text[:60]}...",
                )

        with logfire.span("🛡️ Streamlit — Guardrails Tests"):
            g_results = run_guardrails_eval(
                enriched["guardrails_samples"],
                progress_callback=g_cb,
                checkpoint_path=os.path.join(os.path.dirname(__file__), "golden_dataset.json"),
                full_dataset=enriched,
                force_refresh=True,
            )
            enriched["guardrails_samples"] = g_results
            save_results(enriched, os.path.join(os.path.dirname(__file__), "golden_dataset.json"))
            g_metrics = compute_guardrails_metrics(g_results)
            st.session_state.guardrails_results = g_results
            st.session_state.pipeline_done = True

        g_progress.progress(100, text="✅ Guardrails tests complete!")

        g_rows_live = []
        for r in g_results:
            result_label = {
                "TP": "🛡️ Blocked ✅",
                "TN": "✅ Passed ✅",
                "FP": "🛡️ Blocked ❌ (False Positive)",
                "FN": "✅ Passed ❌ (Missed)",
            }.get(r["result"], r["result"])
            g_rows_live.append(
                {
                    "ID": r["id"],
                    "Input": r["input"][:70],
                    "Expected": "🛡️ Block" if r["expected_blocked"] else "✅ Pass",
                    "Actual": "Blocked" if r["actual_blocked"] else "Passed",
                    "Result": result_label,
                }
            )
        st.dataframe(pd.DataFrame(g_rows_live), width="stretch", hide_index=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Correct", f"{g_metrics['correct']}/{g_metrics['total']}")
        mc2.metric("Precision", f"{g_metrics['precision']:.2f}")
        mc3.metric("Recall", f"{g_metrics['recall']:.2f}")
        mc4.metric("Accuracy", f"{g_metrics['accuracy']:.2f}")

    elif st.session_state.pipeline_done:
        st.success("✅ Pipeline already run. See results below.")

        resp_rows = []
        for s in st.session_state.enriched_dataset["rag_samples"]:
            resp_rows.append(
                {
                    "#": s["id"],
                    "Domain": s["domain"].replace("_", " ").title(),
                    "Question": s["question"][:60],
                    "Live Response": s["actual_response"][:100] + "..."
                    if len(s.get("actual_response", "")) > 100
                    else s.get("actual_response", ""),
                    "Tool Called": s["actual_tools_called"][0] if s.get("actual_tools_called") else "—",
                    "Contexts Retrieved": len(s.get("actual_contexts", [])),
                }
            )
        st.dataframe(pd.DataFrame(resp_rows), width="stretch", hide_index=True)

        if st.session_state.guardrails_results:
            st.divider()
            st.subheader("Guardrails Results (from previous run)")
            g_rows_prev = []
            for r in st.session_state.guardrails_results:
                result_label = {
                    "TP": "🛡️ Blocked ✅",
                    "TN": "✅ Passed ✅",
                    "FP": "Blocked ❌ FP",
                    "FN": "Passed ❌ FN",
                }.get(r["result"], r["result"])
                g_rows_prev.append(
                    {
                        "ID": r["id"],
                        "Input": r["input"][:70],
                        "Result": result_label,
                    }
                )
            st.dataframe(pd.DataFrame(g_rows_prev), width="stretch", hide_index=True)
            gm = compute_guardrails_metrics(st.session_state.guardrails_results)
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Correct", f"{gm['correct']}/{gm['total']}")
            mc2.metric("Precision", f"{gm['precision']:.2f}")
            mc3.metric("Recall", f"{gm['recall']:.2f}")
            mc4.metric("Accuracy", f"{gm['accuracy']:.2f}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Eval Metrics
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Eval Metrics — RAGAS + Tool Correctness")

    if not st.session_state.pipeline_done:
        st.warning("⚠️ Complete Step 2 (Live Pipeline) first to collect responses.")
    else:
        st.markdown(
            "Runs all **6 metric experiments** on the stored responses. "
            "Results are automatically **checkpointed after every single sample and experiment stage**. "
            "If an error occurs or Streamlit is refreshed, evaluation automatically resumes from the last completed stage."
        )

        col_m1, col_m2 = st.columns([1, 1])
        run_metrics_btn = col_m1.button(
            "▶️ Run / Resume Eval Metrics",
            type="primary",
            width="stretch",
            disabled=not st.session_state.pipeline_done,
        )
        reset_metrics_btn = col_m2.button(
            "🔄 Reset & Recompute Metrics",
            width="stretch",
            disabled=not st.session_state.pipeline_done,
        )

        if reset_metrics_btn:
            clear_metrics_checkpoint()
            st.session_state.metric_results = None
            st.rerun()

        if run_metrics_btn:
            status_slot = st.empty()
            results_slots = {}

            metric_display_names = {
                "faithfulness": "Exp 1 — Faithfulness",
                "answer_relevancy": "Exp 2 — Answer Relevancy",
                "context_precision": "Exp 3 — Context Precision",
                "context_recall": "Exp 4 — Context Recall",
                "answer_correctness": "Exp 5 — Answer Correctness",
                "tool_correctness": "Exp 6 — Tool Correctness",
            }
            for key, title in metric_display_names.items():
                results_slots[key] = st.empty()

            def status_cb(msg: str):
                status_slot.info(msg)

            with logfire.span("📊 Streamlit — Run Metrics Button"):
                metric_results = _run_async(run_all_metrics(st.session_state.enriched_dataset, status_cb=status_cb))
                st.session_state.metric_results = metric_results

            status_slot.success("✅ All 6 experiments complete!")

            for key, title in metric_display_names.items():
                if key in metric_results:
                    with results_slots[key].container():
                        _render_metric_table(metric_results[key], key, title)

        elif st.session_state.metric_results:
            st.success("✅ Metrics loaded from saved checkpoint. Showing results below.")
            metric_display_names = {
                "faithfulness": "Exp 1 — Faithfulness",
                "answer_relevancy": "Exp 2 — Answer Relevancy",
                "context_precision": "Exp 3 — Context Precision",
                "context_recall": "Exp 4 — Context Recall",
                "answer_correctness": "Exp 5 — Answer Correctness",
                "tool_correctness": "Exp 6 — Tool Correctness",
            }
            for key, title in metric_display_names.items():
                if key in st.session_state.metric_results:
                    _render_metric_table(st.session_state.metric_results[key], key, title)

        # ── Final Summary ─────────────────────────────────────────────────────
        if st.session_state.metric_results:
            st.divider()
            st.subheader("Final Summary")

            mr = st.session_state.metric_results
            summary = [
                ("Faithfulness", mr.get("faithfulness", pd.DataFrame()).get("faithfulness", pd.Series()).mean()),
                (
                    "Answer Relevancy",
                    mr.get("answer_relevancy", pd.DataFrame()).get("answer_relevancy", pd.Series()).mean(),
                ),
                (
                    "Context Precision",
                    mr.get("context_precision", pd.DataFrame()).get("context_precision", pd.Series()).mean(),
                ),
                ("Context Recall", mr.get("context_recall", pd.DataFrame()).get("context_recall", pd.Series()).mean()),
                (
                    "Answer Correctness",
                    mr.get("answer_correctness", pd.DataFrame()).get("answer_correctness", pd.Series()).mean(),
                ),
                (
                    "Tool Correctness",
                    mr.get("tool_correctness", pd.DataFrame()).get("tool_correctness", pd.Series()).mean(),
                ),
            ]

            cols = st.columns(len(summary))
            for col, (name, score) in zip(cols, summary):
                if pd.notna(score):
                    col.metric(
                        label=name,
                        value=f"{score:.2f}",
                        delta=_grade(score),
                    )

            if st.session_state.guardrails_results:
                gm = compute_guardrails_metrics(st.session_state.guardrails_results)
                st.metric(
                    label="🛡️ Guardrails Accuracy",
                    value=f"{gm['correct']}/{gm['total']}",
                    delta=f"Precision {gm['precision']:.2f} | Recall {gm['recall']:.2f}",
                )

            summary_df = pd.DataFrame(
                [
                    {
                        "Metric": name,
                        "Score": f"{score:.3f}" if pd.notna(score) else "—",
                        "Grade": _grade(score) if pd.notna(score) else "—",
                    }
                    for name, score in summary
                ]
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
