import asyncio
import json
import sys

sys.path.insert(0, ".")
from dotenv import load_dotenv

from evals.guardrails_eval import compute_guardrails_metrics
from evals.metrics import (
    METRICS_CHECKPOINT_PATH,
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    _build_judge,
    _save_checkpoint_data,
    _score_df,
)

load_dotenv(override=True)


async def run_metric_with_retry(metric, inputs, name):
    print(f"--> Scoring {name} across {len(inputs)} samples...")
    scores = []
    for idx, inp in enumerate(inputs):
        for attempt in range(5):
            try:
                res = await metric.ascore(**inp)
                val = getattr(res, "value", res)
                val_f = float(val) if val is not None else 0.0
                scores.append(val_f)
                break
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate limit" in err_str:
                    wait_s = 5 * (attempt + 1)
                    print(f"    [Sample {idx + 1}] Rate limit in {name}, waiting {wait_s}s...")
                    await asyncio.sleep(wait_s)
                else:
                    print(f"    [Sample {idx + 1}] Warning in {name}: {e}")
                    scores.append(0.0)
                    break
        await asyncio.sleep(0.5)
    return scores


async def main():
    print("==========================================================")
    print("🚀 RUNNING END-TO-END RAG EVALUATION & BENCHMARK SUITE")
    print("==========================================================")

    with open("evals/golden_dataset.json") as f:
        golden = json.load(f)

    samples = golden["rag_samples"]
    print(f"Loaded {len(samples)} golden RAG samples.")

    judge_llm, ragas_embeddings = _build_judge()
    print("Judge LLM initialized via OpenRouter 70B.")

    checkpoint_data = {"experiments": {}, "sample_cache": {}}

    # 1. Faithfulness
    inputs_f = [
        {
            "user_input": s["question"],
            "response": s["actual_response"],
            "retrieved_contexts": s["actual_contexts"],
        }
        for s in samples
    ]
    scores_f = await run_metric_with_retry(Faithfulness(llm=judge_llm), inputs_f, "Faithfulness")
    df_f = _score_df("faithfulness", samples, scores_f)
    checkpoint_data["experiments"]["faithfulness"] = df_f.to_dict(orient="records")

    # 2. Answer Relevancy
    inputs_ar = [
        {
            "user_input": s["question"],
            "response": s["actual_response"],
        }
        for s in samples
    ]
    scores_ar = await run_metric_with_retry(
        AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings), inputs_ar, "Answer Relevancy"
    )
    df_ar = _score_df("answer_relevancy", samples, scores_ar)
    checkpoint_data["experiments"]["answer_relevancy"] = df_ar.to_dict(orient="records")

    # 3. Context Precision
    inputs_cp = [
        {
            "user_input": s["question"],
            "reference": s["reference"],
            "retrieved_contexts": s["actual_contexts"],
        }
        for s in samples
    ]
    scores_cp = await run_metric_with_retry(ContextPrecision(llm=judge_llm), inputs_cp, "Context Precision")
    df_cp = _score_df("context_precision", samples, scores_cp)
    checkpoint_data["experiments"]["context_precision"] = df_cp.to_dict(orient="records")

    # 4. Context Recall
    inputs_cr = [
        {
            "user_input": s["question"],
            "reference": s["reference"],
            "retrieved_contexts": s["actual_contexts"],
        }
        for s in samples
    ]
    scores_cr = await run_metric_with_retry(ContextRecall(llm=judge_llm), inputs_cr, "Context Recall")
    df_cr = _score_df("context_recall", samples, scores_cr)
    checkpoint_data["experiments"]["context_recall"] = df_cr.to_dict(orient="records")

    # 5. Answer Correctness
    inputs_ac = [
        {
            "user_input": s["question"],
            "reference": s["reference"],
            "response": s["actual_response"],
        }
        for s in samples
    ]
    scores_ac = await run_metric_with_retry(
        AnswerCorrectness(llm=judge_llm, embeddings=ragas_embeddings), inputs_ac, "Answer Correctness"
    )
    df_ac = _score_df("answer_correctness", samples, scores_ac)
    checkpoint_data["experiments"]["answer_correctness"] = df_ac.to_dict(orient="records")

    # 6. Tool Correctness
    scores_tc = []
    for s in samples:
        exp = s.get("expected_tools", ["retrieve_documents"])
        act = s.get("actual_tools_called", ["retrieve_documents"])
        scores_tc.append(1.0 if set(exp) == set(act) else 0.0)
    df_tc = _score_df("tool_correctness", samples, scores_tc)
    checkpoint_data["experiments"]["tool_correctness"] = df_tc.to_dict(orient="records")

    # Save to metric_results.json
    _save_checkpoint_data(checkpoint_data, METRICS_CHECKPOINT_PATH)
    print("\n✅ Checkpoint saved to evals/metric_results.json")

    # Guardrails metrics
    gm = compute_guardrails_metrics(golden.get("guardrails_samples", []))

    # Print Final Summary Table
    def get_status(score, good=0.75, fair=0.55):
        if score >= good:
            return "✅ Good"
        elif score >= fair:
            return "⚠️ Fair"
        return "❌ Poor"

    avg_f = df_f["faithfulness"].mean()
    avg_ar = df_ar["answer_relevancy"].mean()
    avg_cp = df_cp["context_precision"].mean()
    avg_cr = df_cr["context_recall"].mean()
    avg_ac = df_ac["answer_correctness"].mean()
    avg_tc = df_tc["tool_correctness"].mean()
    g_acc = gm.get("accuracy", 1.0)
    g_correct = gm.get("correct", 6)
    g_total = gm.get("total", 6)

    print("\n" + "=" * 60)
    print("                    FINAL SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<25} | {'Score':<8} | {'Status'}")
    print("-" * 60)
    print(f"{'Faithfulness':<25} | {avg_f:<8.2f} | {get_status(avg_f)}")
    print(f"{'Answer Relevancy':<25} | {avg_ar:<8.2f} | {get_status(avg_ar)}")
    print(f"{'Context Precision':<25} | {avg_cp:<8.2f} | {get_status(avg_cp)}")
    print(f"{'Context Recall':<25} | {avg_cr:<8.2f} | {get_status(avg_cr)}")
    print(f"{'Answer Correctness':<25} | {avg_ac:<8.2f} | {get_status(avg_ac)}")
    print(f"{'Tool Correctness':<25} | {avg_tc:<8.2f} | {get_status(avg_tc)}")
    print(
        f"{'🛡️ Guardrails Accuracy':<25} | {g_correct}/{g_total:<6} | {get_status(g_acc)} (Precision {gm.get('precision', 1.0):.2f} | Recall {gm.get('recall', 1.0):.2f})"
    )
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
