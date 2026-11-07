"""
visualizer.py
Module xuất bản các biểu đồ khoa học trực quan hóa kết quả kiểm định thống kê:
1. Forest Plot: Khoảng tin cậy 95% Bootstrap cho các mô hình.
2. McNemar Contingency Heatmap: Bảng tiếp liên thể hiện phân bố mẫu đúng/sai chéo.
3. Cross-Validation Boxplot: Biểu đồ hộp phân bố điểm F1 qua 10 fold.
4. Latency Distribution Plot: Phân bố thời gian suy luận (KDE) kèm mốc P95/P99.
5. ROC-AUC Curves: Đường cong ROC so sánh năng lực phân loại của 3 mô hình.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Cấu hình phong cách biểu đồ học thuật
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11


def plot_forest_plot(bootstrap_results: dict, output_path: str = "results/figures/forest_plot_ci.png"):
    """
    Vẽ biểu đồ Forest Plot thể hiện điểm ước lượng và khoảng tin cậy 95% Bootstrap của các mô hình.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    metrics = [("f1_score", "F1-Score (Weighted)"), ("accuracy", "Accuracy")]
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
        ax.set_xlabel(f"Chỉ số {metric_title}", fontsize=12)
        ax.set_title(f"Khoảng tin cậy 95% Bootstrap - {metric_title}", fontsize=13, fontweight='bold')
        ax.set_xlim(min(lowers) - 0.03, max(uppers) + 0.03)
        ax.grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Đã lưu Forest Plot: {output_path}")


def plot_mcnemar_heatmap(mcnemar_results: list[dict], output_path: str = "results/figures/mcnemar_contingency.png"):
    """
    Vẽ ma trận tiếp liên (Contingency Matrix) cho các cặp so sánh trong kiểm định McNemar.
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
        ax.set_xticklabels([f'{mB} Đúng', f'{mB} Sai'], fontsize=10)
        ax.set_yticklabels([f'{mA} Đúng', f'{mA} Sai'], fontsize=10, rotation=0)
        
        sig_str = "Có ý nghĩa (p < 0.05)" if res["is_significant_at_05"] else "Không ý nghĩa (p >= 0.05)"
        p_val_fmt = f"{res['p_value']:.4e}" if res['p_value'] < 0.0001 else f"{res['p_value']:.4f}"
        ax.set_title(f"{mA} vs {mB}\nChi-sq={res['statistic']:.2f}, p={p_val_fmt}\n({sig_str})", 
                     fontsize=11, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Đã lưu McNemar Heatmap: {output_path}")


def plot_cv_boxplots(cv_results: dict, output_path: str = "results/figures/cv_scores_comparison.png"):
    """
    Vẽ biểu đồ Boxplot + Jitter Points cho kết quả 10-Fold Cross-Validation.
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
    
    sns.boxplot(x="Model", y="F1-Score", data=df_cv, palette=palette, width=0.45,
                boxprops=dict(alpha=0.8), showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"yellow", "markeredgecolor":"black", "markersize":"8"})
    sns.stripplot(x="Model", y="F1-Score", data=df_cv, color='black', alpha=0.6, jitter=0.15, size=7)
    
    anova = cv_results["anova"]
    plt.title(f"Phân bố Điểm F1 qua 10-Fold Stratified Cross-Validation\n"
              f"Repeated Measures ANOVA: F = {anova['f_statistic']:.3f}, p-value = {anova['p_value']:.4e}",
              fontsize=13, fontweight='bold')
    plt.xlabel("Mô hình AI", fontsize=12)
    plt.ylabel("Weighted F1-Score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Đã lưu CV Boxplots: {output_path}")


def plot_latency_distributions(latencies_dict: dict, latency_stats: dict, 
                               output_path: str = "results/figures/latency_distribution.png"):
    """
    Vẽ biểu đồ phân phối độ trễ suy luận (KDE) kèm mốc trung vị và bách phân vị P95, P99.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    colors = {"Logistic Regression": "#1f77b4", "Random Forest": "#2ca02c", "Multi-Layer Perceptron": "#d62728"}
    
    for name, lat in latencies_dict.items():
        arr = np.array(lat)
        stats_item = latency_stats["summary_statistics"][name]
        sns.kdeplot(arr, label=f"{name} (Median: {stats_item['median_ms']:.2f}ms, P95: {stats_item['p95_ms']:.2f}ms)",
                    color=colors[name], linewidth=2.2, fill=True, alpha=0.15)
        
        # Mốc P95
        plt.axvline(stats_item['p95_ms'], color=colors[name], linestyle=':', alpha=0.7)
        
    plt.title("Phân phối Thời gian Suy luận (Inference Latency Distribution per Sample)\n"
              "Đường chấm nét dọc (:) thể hiện mốc Bách phân vị P95", fontsize=13, fontweight='bold')
    plt.xlabel("Thời gian suy luận (mili-giây / sample)", fontsize=12)
    plt.ylabel("Mật độ xác suất (Density)", fontsize=12)
    plt.legend(fontsize=10, loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Đã lưu Latency Distribution: {output_path}")


def plot_roc_curves(eval_results: dict, y_test: np.ndarray, output_path: str = "results/figures/roc_curves.png"):
    """
    Vẽ biểu đồ ROC-AUC so sánh khả năng phân biệt của 3 mô hình.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    colors = {"Logistic Regression": "#1f77b4", "Random Forest": "#2ca02c", "Multi-Layer Perceptron": "#d62728"}
    
    for name, res in eval_results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[name], lw=2.2, label=f"{name} (AUC = {roc_auc:.4f})")
        
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Ngẫu nhiên (AUC = 0.5000)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tỷ lệ Dương tính Giả (False Positive Rate)', fontsize=12)
    plt.ylabel('Tỷ lệ Dương tính Thật (True Positive Rate)', fontsize=12)
    plt.title('Đường cong Đặc trưng Hoạt động của Bộ thu (ROC Curves)', fontsize=13, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Đã lưu ROC Curves: {output_path}")
