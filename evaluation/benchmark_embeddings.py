import json
import os
import time
import psutil
import numpy as np
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer

EVAL_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "evaluation_dataset", "labeled_pairs.json"
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output"
)

# Candidate models to evaluate
CANDIDATE_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5"
]

def run_benchmark(models: List[str] = CANDIDATE_MODELS, threshold: float = 0.65) -> Dict[str, Any]:
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}

    print("=" * 80)
    print("EMBEDDING MODEL BENCHMARK FOR RESUME-JD MATCHING")
    print(f"Dataset: {len(dataset)} labeled pairs | Decision Threshold: {threshold}")
    print("=" * 80)

    for model_name in models:
        print(f"\nEvaluating Model: {model_name}...")
        start_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        
        t0 = time.time()
        try:
            model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            continue
        load_time = time.time() - t0
        mem_after_load = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

        # Run inference on pairs
        t_infer_start = time.time()
        req_texts = [d["requirement"] for d in dataset]
        ev_texts = [d["evidence"] for d in dataset]

        req_embs = model.encode(req_texts, normalize_embeddings=True)
        ev_embs = model.encode(ev_texts, normalize_embeddings=True)

        sims = np.sum(req_embs * ev_embs, axis=1)
        total_infer_time = time.time() - t_infer_start
        avg_infer_time_ms = (total_infer_time / len(dataset)) * 1000

        # Metrics
        y_true = [1 if d["is_match"] else 0 for d in dataset]
        y_pred = [1 if s >= threshold else 0 for s in sims]

        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        false_positives = [
            {"id": d["id"], "req": d["requirement"], "evidence": d["evidence"], "score": float(sim)}
            for d, yt, yp, sim in zip(dataset, y_true, y_pred, sims) if yt == 0 and yp == 1
        ]
        false_negatives = [
            {"id": d["id"], "req": d["requirement"], "evidence": d["evidence"], "score": float(sim)}
            for d, yt, yp, sim in zip(dataset, y_true, y_pred, sims) if yt == 1 and yp == 0
        ]

        # Category performance
        cat_stats = {}
        for d, yt, yp in zip(dataset, y_true, y_pred):
            c = d["category"]
            if c not in cat_stats:
                cat_stats[c] = {"correct": 0, "total": 0}
            cat_stats[c]["total"] += 1
            if yt == yp:
                cat_stats[c]["correct"] += 1

        category_acc = {c: round(v["correct"] / v["total"], 2) for c, v in cat_stats.items()}

        result_entry = {
            "model_name": model_name,
            "threshold": threshold,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "load_time_sec": round(load_time, 2),
            "total_inference_time_sec": round(total_infer_time, 3),
            "avg_inference_ms_per_pair": round(avg_infer_time_ms, 2),
            "memory_usage_mb": round(mem_after_load - start_mem, 1),
            "category_accuracy": category_acc,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
        all_results[model_name] = result_entry

        print(f"Results for {model_name}:")
        print(f"  Precision: {precision:.2%} | Recall: {recall:.2%} | F1: {f1:.2%}")
        print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        print(f"  Avg Latency: {avg_infer_time_ms:.1f}ms/pair | Memory: ~{mem_after_load - start_mem:.1f} MB")

    out_file = os.path.join(OUTPUT_DIR, "embedding_benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Benchmark results successfully saved to: {out_file}")
    print("=" * 80)
    return all_results

if __name__ == "__main__":
    run_benchmark()
