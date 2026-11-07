"""
visualizer.py
Scientific visualization module for statistical evaluation and model benchmarking:
1. Forest Plot: 95% Bootstrap Confidence Intervals for AI models.
2. McNemar Contingency Heatmap: 2x2 contingency tables displaying cross-classification errors.
3. Cross-Validation Boxplot: 10-Fold CV score distributions with Repeated Measures ANOVA.
4. Latency Distribution Plot: Empirical KDE of inference latency with P95/P99 percentiles.
5. ROC-AUC Curves: Receiver Operating Characteristic curves comparing discriminative power.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Scientific plotting style configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11


def plot_forest_plot(bootstrap_results: dict, output_path: str = "results/figures/forest_plot_ci.png"):
    """
    Plots a Forest Plot displaying Point Estimates and 95% Bootstrap Confidence Intervals.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    metrics = [("f1_score", "Weighted F1-Score"), ("accuracy", "Accuracy")]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    
    for ax_idx, (metric_key, metric_title) in enumerate(metrics):
        ax = axes[ax_idx]
        model_names = list(bootstrap_results.keys())
        y_pos = np.arange(len(model_names))
        
        points = [bootstrap_results[m][metric_key]["point_estimate"] for m in model_names]
        lowers = [bootstrap_results[m][metric_key]["ci_95_lower"] for m in model_names]
        uppers = [bootstrap_results[m][metric_key]["ci_95_upper"] for m in model_names]
        
        xerr = [np.array(points) - np.array(lowers), np.array(uppers) - np.array(points)]
        
        for i in range(len(model_names)):
            ax.errorbar(points[i], y_pos[i], xerr=[[xerr[0][i]], [xerr[1][i]]],
                        fmt='o', color=colors[i], ecolor=colors[i], elinewidth=2.5,
                        capsize=6, capthick=2, markersize=8)
            ax.annotate(f"{points[i]:.4f}\n[{lowers[i]:.4f}, {uppers[i]:.4f}]",
                        (points[i], y_pos[i] + 0.15),
                        ha='center', fontsize=9.5, fontweight='bold', color=colors[i])
            
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names, fontsize=11, fontweight='bold')
        ax.set_xlabel(f"{metric_title}", fontsize=12)
        ax.set_title(f"95% Bootstrap Confidence Intervals - {metric_title}", fontsize=13, fontweight='bold')
        ax.set_xlim(min(lowers) - 0.03, max(uppers) + 0.03)
        ax.grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Saved Forest Plot: {output_path}")


def plot_mcnemar_heatmap(mcnemar_results: list[dict], output_path: str = "results/figures/mcnemar_contingency.png"):
    """
    Plots contingency heatmaps for paired model comparisons in McNemar's test.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_pairs = len(mcnemar_results)
    fig, axes = plt.subplots(1, n_pairs, figsize=(5.5 * n_pairs, 4.5))
    if n_pairs == 1:
        axes = [axes]
        
    for idx, res in enumerate(mcnemar_results):
        ax = axes[idx]
        tbl = res["contingency_table"]
        matrix = np.array([
            [tbl["both_correct_n11"], tbl["A_correct_B_wrong_n10"]],
            [tbl["A_wrong_B_correct_n01"], tbl["both_wrong_n00"]]
        ])
        
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    annot_kws={'size': 13, 'weight': 'bold'})
        
        mA = res["model_A"].split()[0]
        mB = res["model_B"].split()[0]
        ax.set_xticklabels([f'{mB} Correct', f'{mB} Incorrect'], fontsize=10)
        ax.set_yticklabels([f'{mA} Correct', f'{mA} Incorrect'], fontsize=10, rotation=0)
        
        sig_str = "Significant (p < 0.05)" if res["is_significant_at_05"] else "Not Significant (p >= 0.05)"
        p_val_fmt = f"{res['p_value']:.4e}" if res['p_value'] < 0.0001 else f"{res['p_value']:.4f}"
        ax.set_title(f"{mA} vs {mB}\nChi-sq={res['statistic']:.2f}, p={p_val_fmt}\n({sig_str})", 
                     fontsize=11, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Saved McNemar Heatmap: {output_path}")


def plot_cv_boxplots(cv_results: dict, output_path: str = "results/figures/cv_scores_comparison.png"):
    """
    Plots Boxplots + Jittered observations for 10-Fold Stratified Cross-Validation.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fold_scores = cv_results["fold_scores"]
    
    df_list = []
    for name, scores in fold_scores.items():
        for s in scores:
            df_list.append({"Model": name, "F1-Score": s})
    df_cv = pd.DataFrame(df_list)
    
    plt.figure(figsize=(9, 5))
    palette = ["#1f77b4", "#2ca02c", "#d62728"]
    
    sns.boxplot(x="Model", y="F1-Score", data=df_cv, hue="Model", palette=palette, width=0.45,
                legend=False, boxprops=dict(alpha=0.8), showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"yellow", "markeredgecolor":"black", "markersize":"8"})
    sns.stripplot(x="Model", y="F1-Score", data=df_cv, color='black', alpha=0.6, jitter=0.15, size=7)
    
    anova = cv_results["anova"]
    plt.title(f"10-Fold Stratified Cross-Validation F1-Score Distribution\n"
              f"Repeated Measures ANOVA: F = {anova['f_statistic']:.3f}, p-value = {anova['p_value']:.4e}",
              fontsize=13, fontweight='bold')
    plt.xlabel("AI Architecture", fontsize=12)
    plt.ylabel("Weighted F1-Score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Saved CV Boxplots: {output_path}")


def plot_latency_distributions(latencies_dict: dict, latency_stats: dict, 
                               output_path: str = "results/figures/latency_distribution.png"):
    """
    Plots empirical KDE of inference latency alongside Median, P95, and P99 benchmarks.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    colors = {"Logistic Regression": "#1f77b4", "Random Forest": "#2ca02c", "Multi-Layer Perceptron": "#d62728"}
    
    for name, lat in latencies_dict.items():
        arr = np.array(lat)
        stats_item = latency_stats["summary_statistics"][name]
        sns.kdeplot(arr, label=f"{name} (Median: {stats_item['median_ms']:.2f}ms, P95: {stats_item['p95_ms']:.2f}ms)",
                    color=colors[name], linewidth=2.2, fill=True, alpha=0.15)
        
        # P95 line marker
        plt.axvline(stats_item['p95_ms'], color=colors[name], linestyle=':', alpha=0.7)
        
    plt.title("Per-Sample Inference Latency Distribution (Empirical Density)\n"
              "Vertical dotted lines (:) indicate the 95th Percentile (P95 SLA Benchmark)", fontsize=13, fontweight='bold')
    plt.xlabel("Inference Latency (milliseconds per sample)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.legend(fontsize=10, loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Saved Latency Distribution: {output_path}")


def plot_roc_curves(eval_results: dict, y_test: np.ndarray, output_path: str = "results/figures/roc_curves.png"):
    """
    Plots Receiver Operating Characteristic (ROC) curves comparing model discrimination.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    colors = {"Logistic Regression": "#1f77b4", "Random Forest": "#2ca02c", "Multi-Layer Perceptron": "#d62728"}
    
    for name, res in eval_results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[name], lw=2.2, label=f"{name} (AUC = {roc_auc:.4f})")
        
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.5000)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Saved ROC Curves: {output_path}")
