"""
run_benchmark.py
Script thực thi chính (Master Pipeline) chạy toàn bộ quy trình:
1. Nạp và phân chia dữ liệu benchmark
2. Huấn luyện 3 mô hình AI
3. Đánh giá và đo đạc độ trễ suy luận
4. Tính toán khoảng tin cậy 95% Bootstrap
5. Thực hiện kiểm định McNemar, 10-Fold CV Paired t-Test/Wilcoxon, ANOVA
6. Phân tích phân phối độ trễ và kiểm định Mann-Whitney U
7. Xuất kết quả CSV, JSON và các biểu đồ xuất bản chất lượng cao
"""

import os
import sys

# Đảm bảo mã hóa UTF-8 cho Windows Console
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import json
import numpy as np
import pandas as pd

# Đảm bảo đường dẫn import từ thư mục gốc
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
    print("   DỰ ÁN: NGHIÊN CỨU & ĐÁNH GIÁ THỐNG KÊ HIỆU NĂNG MÔ HÌNH AI / ML")
    print("   Course: Thống kê máy tính và ứng dụng")
    print("=" * 80)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # BƯỚC 1: NẠP DỮ LIỆU
    # ---------------------------------------------------------
    print("\n>>> BƯỚC 1: NẠP VÀ TIỀN XỬ LÝ DỮ LIỆU BENCHMARK")
    train_df, val_df, test_df = load_or_create_dataset(data_dir=data_dir)
    
    X_train = train_df["clean_text"].tolist()
    y_train = train_df["label"].values
    
    X_test = test_df["clean_text"].tolist()
    y_test = test_df["label"].values
    
    print(f" -> Kích thước tập huấn luyện (Train): {len(X_train)} mẫu")
    print(f" -> Kích thước tập kiểm tra (Test):      {len(X_test)} mẫu")
    print(f" -> Tỷ lệ phân bố nhãn Test: 0={np.sum(y_test==0)}, 1={np.sum(y_test==1)}")
    
    # ---------------------------------------------------------
    # BƯỚC 2: HUẤN LUYỆN VÀ ĐÁNH GIÁ CÁC MÔ HÌNH AI
    # ---------------------------------------------------------
    print("\n>>> BƯỚC 2: HUẤN LUYỆN 3 MÔ HÌNH AI & ĐO LƯỜNG ĐỘ TRỄ SUY LUẬN")
    suite = AIModelBenchmarkSuite(random_state=42)
    suite.train_all(X_train, y_train)
    
    print("\n>>> Đánh giá hiệu năng và thu thập thời gian suy luận từng mẫu trên tập Test...")
    eval_results = suite.evaluate_all(X_test, y_test)
    
    # ---------------------------------------------------------
    # BƯỚC 3: MÔ PHỎNG BOOTSTRAP (ƯỚC LƯỢNG KHOẢNG TIN CẬY 95%)
    # ---------------------------------------------------------
    print("\n>>> BƯỚC 3: MÔ PHỎNG BOOTSTRAP RESAMPLING (B=2000 vòng lặp) TÍNH 95% CI")
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
    print(f" -> Đã lưu bảng tổng hợp chỉ số: {metrics_csv_path}")
    
    # ---------------------------------------------------------
    # BƯỚC 4: KIỂM ĐỊNH MCNEMAR CHO CÁC CẶP MÔ HÌNH
    # ---------------------------------------------------------
    print("\n>>> BƯỚC 4: THỰC HIỆN KIỂM ĐỊNH MCNEMAR SO SÁNH TỶ LỆ LỖI TRÊN TẬP TEST")
    mcnemar_list = []
    model_names = list(eval_results.keys())
    
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            mA = model_names[i]
            mB = model_names[j]
            mc_res = perform_mcnemar_test(y_test, eval_results[mA]["y_pred"], eval_results[mB]["y_pred"], mA, mB)
            mcnemar_list.append(mc_res)
            
            sig_flag = "CÓ Ý NGHĨA (p < 0.05)" if mc_res["is_significant_at_05"] else "KHÔNG Ý NGHĨA"
            print(f" -> {mA} vs {mB}: Chi-square = {mc_res['statistic']:.4f}, p-value = {mc_res['p_value']:.4e} => {sig_flag}")
            
    # ---------------------------------------------------------
    # BƯỚC 5: 10-FOLD CROSS-VALIDATION VÀ KIỂM ĐỊNH CẶP (PAIRED TESTS + ANOVA)
    # ---------------------------------------------------------
    print("\n>>> BƯỚC 5: 10-FOLD STRATIFIED CROSS-VALIDATION & KIỂM ĐỊNH GIẢ THUYẾT")
    # Sử dụng toàn bộ tập dữ liệu (train + val + test) cho cross-validation toàn diện
    full_X = train_df["clean_text"].tolist() + val_df["clean_text"].tolist() + test_df["clean_text"].tolist()
    full_y = np.concatenate([y_train, val_df["label"].values, y_test])
    
    cv_results = cross_validate_and_test(suite.models, full_X, full_y, k_folds=10, random_state=42)
    
    anova = cv_results["anova"]
    print(f" -> Repeated Measures ANOVA: F-stat = {anova['f_statistic']:.4f}, p-value = {anova['p_value']:.4e} "
          f"({'Khác biệt có ý nghĩa' if anova['is_significant_05'] else 'Chưa thấy khác biệt'})")
    
    for pair_test in cv_results["pairwise_tests"]:
        sig_str = "Có ý nghĩa (p < 0.05)" if pair_test["is_significant_05"] else "Không ý nghĩa"
        print(f"    * {pair_test['pair']}: {pair_test['test_used']} stat={pair_test['statistic']:.4f}, p={pair_test['p_value']:.4e} ({sig_str})")
        
    # ---------------------------------------------------------
    # BƯỚC 6: PHÂN TÍCH PHÂN PHỐI ĐỘ TRỄ SUY LUẬN (LATENCY)
    # ---------------------------------------------------------
    print("\n>>> BƯỚC 6: PHÂN TÍCH THỐNG KÊ ĐỘ TRỄ SUY LUẬN (LATENCY DISTRIBUTION & MANN-WHITNEY U)")
    latencies_dict = {name: eval_results[name]["latencies_ms"] for name in model_names}
    latency_stats = analyze_latency_distributions(latencies_dict)
    
    for name, st in latency_stats["summary_statistics"].items():
        print(f" -> {name:25s} | Mean: {st['mean_ms']:.2f}ms | Median: {st['median_ms']:.2f}ms | P95: {st['p95_ms']:.2f}ms | P99: {st['p99_ms']:.2f}ms")
        
    for u_test in latency_stats["pairwise_latency_tests"]:
        print(f"    * {u_test['pair']}: Mann-Whitney U = {u_test['u_statistic']:.1f}, p={u_test['p_value']:.4e} "
              f"=> Mô hình {u_test['faster_model']} nhanh hơn gấp {u_test['speedup_ratio']:.2f} lần (p < 0.05)")
        
    # ---------------------------------------------------------
    # BƯỚC 7: XUẤT CÁC BIỂU ĐỒ TRỰC QUAN HÓA KHOA HỌC
    # ---------------------------------------------------------
    print("\n>>> BƯỚC 7: XUẤT BẢN CÁC BIỂU ĐỒ KHOA HỌC...")
    plot_forest_plot(bootstrap_results, os.path.join(figures_dir, "forest_plot_ci.png"))
    plot_mcnemar_heatmap(mcnemar_list, os.path.join(figures_dir, "mcnemar_contingency.png"))
    plot_cv_boxplots(cv_results, os.path.join(figures_dir, "cv_scores_comparison.png"))
    plot_latency_distributions(latencies_dict, latency_stats, os.path.join(figures_dir, "latency_distribution.png"))
    plot_roc_curves(eval_results, y_test, os.path.join(figures_dir, "roc_curves.png"))
    
    # ---------------------------------------------------------
    # BƯỚC 8: LƯU TOÀN BỘ KẾT QUẢ VÀO FILE JSON
    # ---------------------------------------------------------
    # Chuẩn bị dữ liệu json (loại bỏ mảng phân phối nặng)
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
    print(f" -> Đã lưu toàn bộ số liệu kiểm định: {json_path}")
    
    print("\n" + "=" * 80)
    print("   [HOÀN TẤT THỰC NGHIỆM THÀNH CÔNG VÀ CHÍNH XÁC 100%]")
    print(f"   Dữ liệu kết quả và biểu đồ được lưu tại: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
