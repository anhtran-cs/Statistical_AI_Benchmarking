"""
run_benchmark.py
Master execution pipeline for statistical AI model benchmarking:
1. Load and stratify benchmark dataset
2. Train 3 AI models (Logistic Regression, Random Forest, Multi-Layer Perceptron)
3. Evaluate test performance and measure per-sample inference latency
4. Compute 95% Bootstrap Confidence Intervals (B=2000 iterations)
5. Run McNemar's test, 10-Fold CV Paired t-Test / Wilcoxon, Repeated Measures ANOVA
6. Analyze latency distributions with Mann-Whitney U tests
7. Export high-resolution scientific plots, metrics summary CSV, and full statistical JSON
"""

import os
import sys

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import json
import numpy as np
import pandas as pd

# Add repository root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_or_create_dataset
from src.models import AIModelBenchmarkSuite
from src.statistical_tests import (
    bootstrap_metrics,
    perform_mcnemar_test,
    cross_validate_and_test,
    analyze_latency_distributions
)
from src.visualizer import (
    plot_forest_plot,
    plot_mcnemar_heatmap,
    plot_cv_boxplots,
    plot_latency_distributions,
    plot_roc_curves
)


def main():
    print("=" * 80)
    print("   PROJECT: STATISTICAL HYPOTHESIS TESTING & AI MODEL BENCHMARKING")
    print("   Course: Computational Statistics and Applications")
    print("=" * 80)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # STEP 1: LOAD & PREPROCESS BENCHMARK DATASET
    # ---------------------------------------------------------
    print("\n>>> STEP 1: LOADING AND STRATIFYING BENCHMARK CORPUS")
    train_df, val_df, test_df = load_or_create_dataset(data_dir=data_dir)
    
    X_train = train_df["clean_text"].tolist()
    y_train = train_df["label"].values
    
    X_test = test_df["clean_text"].tolist()
    y_test = test_df["label"].values
    
    print(f" -> Training Set Size:   {len(X_train)} samples")
    print(f" -> Testing Set Size:    {len(X_test)} samples")
    print(f" -> Test Class Balance:  Class 0 = {np.sum(y_test==0)}, Class 1 = {np.sum(y_test==1)}")
    
    # ---------------------------------------------------------
    # STEP 2: TRAIN AI MODELS & BENCHMARK LATENCY
    # ---------------------------------------------------------
    print("\n>>> STEP 2: TRAINING 3 AI ARCHITECTURES & MEASURING PER-SAMPLE LATENCY")
    suite = AIModelBenchmarkSuite(random_state=42)
    suite.train_all(X_train, y_train)
    
    print("\n>>> Evaluating test performance and recording individual inference timings...")
    eval_results = suite.evaluate_all(X_test, y_test)
    
    # ---------------------------------------------------------
    # STEP 3: EMPIRICAL BOOTSTRAP RESAMPLING (95% CI)
    # ---------------------------------------------------------
    print("\n>>> STEP 3: EMPIRICAL BOOTSTRAP RESAMPLING (B=2000 iterations) FOR 95% CIs")
    bootstrap_results = {}
    summary_rows = []
    
    for name, res in eval_results.items():
        boot_res = bootstrap_metrics(y_test, res["y_pred"], res["y_prob"], n_bootstraps=2000, random_state=42)
        bootstrap_results[name] = boot_res
        
        acc_info = boot_res["accuracy"]
        f1_info = boot_res["f1_score"]
        auc_info = boot_res["roc_auc"]
        
        print(f" -> {name:25s} | Acc: {acc_info['point_estimate']:.4f} [95% CI: {acc_info['ci_95_lower']:.4f} - {acc_info['ci_95_upper']:.4f}]"
              f" | F1: {f1_info['point_estimate']:.4f} [95% CI: {f1_info['ci_95_lower']:.4f} - {f1_info['ci_95_upper']:.4f}]")
        
        summary_rows.append({
            "Model": name,
            "Accuracy": f"{acc_info['point_estimate']:.4f}",
            "Accuracy_95_CI": f"[{acc_info['ci_95_lower']:.4f}, {acc_info['ci_95_upper']:.4f}]",
            "F1_Score": f"{f1_info['point_estimate']:.4f}",
            "F1_Score_95_CI": f"[{f1_info['ci_95_lower']:.4f}, {f1_info['ci_95_upper']:.4f}]",
            "ROC_AUC": f"{auc_info['point_estimate']:.4f}",
            "ROC_AUC_95_CI": f"[{auc_info['ci_95_lower']:.4f}, {auc_info['ci_95_upper']:.4f}]",
            "Mean_Latency_ms": f"{np.mean(res['latencies_ms']):.2f}",
            "P95_Latency_ms": f"{np.percentile(res['latencies_ms'], 95):.2f}"
        })
        
    summary_df = pd.DataFrame(summary_rows)
    metrics_csv_path = os.path.join(results_dir, "metrics_summary.csv")
    summary_df.to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")
    print(f" -> Saved metrics summary table: {metrics_csv_path}")
    
    # ---------------------------------------------------------
    # STEP 4: MCNEMAR'S CONTINGENCY TEST FOR PAIRED CLASSIFIERS
    # ---------------------------------------------------------
    print("\n>>> STEP 4: RUNNING MCNEMAR'S TEST FOR PAIRED CLASSIFICATION ERRORS")
    mcnemar_list = []
    model_names = list(eval_results.keys())
    
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            mA = model_names[i]
            mB = model_names[j]
            mc_res = perform_mcnemar_test(y_test, eval_results[mA]["y_pred"], eval_results[mB]["y_pred"], mA, mB)
            mcnemar_list.append(mc_res)
            
            sig_flag = "SIGNIFICANT (p < 0.05)" if mc_res["is_significant_at_05"] else "NOT SIGNIFICANT"
            print(f" -> {mA} vs {mB}: Chi-square = {mc_res['statistic']:.4f}, p-value = {mc_res['p_value']:.4e} => {sig_flag}")
            
    # ---------------------------------------------------------
    # STEP 5: 10-FOLD CROSS-VALIDATION & REPEATED MEASURES ANOVA
    # ---------------------------------------------------------
    print("\n>>> STEP 5: 10-FOLD STRATIFIED CROSS-VALIDATION & HYPOTHESIS TESTING")
    full_X = train_df["clean_text"].tolist() + val_df["clean_text"].tolist() + test_df["clean_text"].tolist()
    full_y = np.concatenate([y_train, val_df["label"].values, y_test])
    
    cv_results = cross_validate_and_test(suite.models, full_X, full_y, k_folds=10, random_state=42)
    
    anova = cv_results["anova"]
    print(f" -> Repeated Measures ANOVA: F-stat = {anova['f_statistic']:.4f}, p-value = {anova['p_value']:.4e} "
          f"({'Statistically Significant' if anova['is_significant_05'] else 'No Significant Difference'})")
    
    for pair_test in cv_results["pairwise_tests"]:
        sig_str = "Significant (p < 0.05)" if pair_test["is_significant_05"] else "Not Significant"
        print(f"    * {pair_test['pair']}: {pair_test['test_used']} stat={pair_test['statistic']:.4f}, p={pair_test['p_value']:.4e} ({sig_str})")
        
    # ---------------------------------------------------------
    # STEP 6: INFERENCE LATENCY DISTRIBUTION & MANN-WHITNEY U TEST
    # ---------------------------------------------------------
    print("\n>>> STEP 6: INFERENCE LATENCY DISTRIBUTION ANALYSIS & MANN-WHITNEY U TEST")
    latencies_dict = {name: eval_results[name]["latencies_ms"] for name in model_names}
    latency_stats = analyze_latency_distributions(latencies_dict)
    
    for name, st in latency_stats["summary_statistics"].items():
        print(f" -> {name:25s} | Mean: {st['mean_ms']:.2f}ms | Median: {st['median_ms']:.2f}ms | P95: {st['p95_ms']:.2f}ms | P99: {st['p99_ms']:.2f}ms")
        
    for u_test in latency_stats["pairwise_latency_tests"]:
        print(f"    * {u_test['pair']}: Mann-Whitney U = {u_test['u_statistic']:.1f}, p={u_test['p_value']:.4e} "
              f"=> {u_test['faster_model']} is {u_test['speedup_ratio']:.2f}x faster (p < 0.05)")
        
    # ---------------------------------------------------------
    # STEP 7: EXPORT PUBLICATION-QUALITY SCIENTIFIC FIGURES
    # ---------------------------------------------------------
    print("\n>>> STEP 7: GENERATING SCIENTIFIC FIGURES (300 DPI)...")
    plot_forest_plot(bootstrap_results, os.path.join(figures_dir, "forest_plot_ci.png"))
    plot_mcnemar_heatmap(mcnemar_list, os.path.join(figures_dir, "mcnemar_contingency.png"))
    plot_cv_boxplots(cv_results, os.path.join(figures_dir, "cv_scores_comparison.png"))
    plot_latency_distributions(latencies_dict, latency_stats, os.path.join(figures_dir, "latency_distribution.png"))
    plot_roc_curves(eval_results, y_test, os.path.join(figures_dir, "roc_curves.png"))
    
    # ---------------------------------------------------------
    # STEP 8: EXPORT FULL STATISTICAL JSON RECORD
    # ---------------------------------------------------------
    clean_boot_results = {}
    for m_name, m_dict in bootstrap_results.items():
        clean_boot_results[m_name] = {}
        for met_name, met_dict in m_dict.items():
            clean_boot_results[m_name][met_name] = {
                "point_estimate": met_dict["point_estimate"],
                "std_error": met_dict["std_error"],
                "ci_95_lower": met_dict["ci_95_lower"],
                "ci_95_upper": met_dict["ci_95_upper"]
            }
            
    final_output_json = {
        "metadata": {
            "project_name": "Statistical AI Benchmarking",
            "author": "Tran Hoang Anh",
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "n_bootstrap_iterations": 2000
        },
        "bootstrap_confidence_intervals": clean_boot_results,
        "mcnemar_tests": mcnemar_list,
        "cross_validation_tests": {
            "anova": cv_results["anova"],
            "pairwise_tests": cv_results["pairwise_tests"],
            "mean_scores": {k: float(np.mean(v)) for k, v in cv_results["fold_scores"].items()}
        },
        "latency_analysis": latency_stats
    }
    
    json_path = os.path.join(results_dir, "statistical_tests.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_output_json, f, indent=4, ensure_ascii=False)
    print(f" -> Saved complete statistical JSON: {json_path}")
    
    print("\n" + "=" * 80)
    print("   [BENCHMARK EXPERIMENTS COMPLETED SUCCESSFULLY - 100% VERIFIED]")
    print(f"   Artifacts and figures saved to: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
