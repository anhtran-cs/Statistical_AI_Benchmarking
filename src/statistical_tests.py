"""
statistical_tests.py
Advanced inferential and computational statistical testing module for AI benchmarking:
1. Bootstrap Resampling: 95% Empirical Confidence Intervals for Accuracy, F1, Precision, Recall, AUC.
2. McNemar's Test: Paired classification error testing (Edwards' continuity correction & exact binomial).
3. 10-Fold Stratified Cross-Validation + Paired t-Test / Wilcoxon: Distribution-tested paired evaluation.
4. Repeated Measures ANOVA & Post-hoc: Simultaneous multi-model variance analysis.
5. Latency Distribution Analysis: Heavy-tailed Log-Normal modeling & non-parametric Mann-Whitney U test.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, 
                      n_bootstraps: int = 2000, alpha: float = 0.05, 
                      random_state: int = 42) -> dict:
    """
    Ước lượng khoảng tin cậy 95% cho 5 chỉ số hiệu năng (Accuracy, F1, Precision, Recall, ROC-AUC)
    bằng phương pháp mô phỏng lặp lại Bootstrap (Empirical Bootstrap).
    """
    np.random.seed(random_state)
    n = len(y_true)
    
    boot_acc = []
    boot_f1 = []
    boot_prec = []
    boot_rec = []
    boot_auc = []
    
    for _ in range(n_bootstraps):
        idx = np.random.choice(n, size=n, replace=True)
        # Đảm bảo mẫu bootstrap có đủ 2 lớp để tính ROC-AUC
        if len(np.unique(y_true[idx])) < 2:
            continue
            
        boot_acc.append(accuracy_score(y_true[idx], y_pred[idx]))
        boot_f1.append(f1_score(y_true[idx], y_pred[idx], average='weighted'))
        boot_prec.append(precision_score(y_true[idx], y_pred[idx], average='weighted', zero_division=0))
        boot_rec.append(recall_score(y_true[idx], y_pred[idx], average='weighted', zero_division=0))
        boot_auc.append(roc_auc_score(y_true[idx], y_prob[idx]))
        
    def get_summary(arr, point_val):
        arr = np.array(arr)
        se = float(np.std(arr, ddof=1))
        ci_lower = float(np.percentile(arr, 100 * (alpha / 2)))
        ci_upper = float(np.percentile(arr, 100 * (1 - alpha / 2)))
        return {
            "point_estimate": float(point_val),
            "std_error": se,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "distribution": arr
        }
        
    return {
        "accuracy": get_summary(boot_acc, accuracy_score(y_true, y_pred)),
        "f1_score": get_summary(boot_f1, f1_score(y_true, y_pred, average='weighted')),
        "precision": get_summary(boot_prec, precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        "recall": get_summary(boot_rec, recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        "roc_auc": get_summary(boot_auc, roc_auc_score(y_true, y_prob))
    }


def perform_mcnemar_test(y_true: np.ndarray, y_pred_A: np.ndarray, y_pred_B: np.ndarray,
                         name_A: str = "Model A", name_B: str = "Model B") -> dict:
    """
    Thực hiện kiểm định McNemar so sánh 2 mô hình phân loại trên cùng tập dữ liệu test.
    Cặp giả thuyết:
    H0: p_01 = p_10 (Xác suất phân loại sai cận biên của 2 mô hình là như nhau)
    H1: p_01 != p_10 (Hai mô hình có sự khác biệt có ý nghĩa thống kê về tỷ lệ phân loại sai)
    """
    correct_A = (y_pred_A == y_true)
    correct_B = (y_pred_B == y_true)
    
    # Bảng tiếp liên 2x2
    n00 = int(np.sum(~correct_A & ~correct_B)) # Cả 2 đều sai
    n01 = int(np.sum(~correct_A & correct_B))  # A sai, B đúng
    n10 = int(np.sum(correct_A & ~correct_B))  # A đúng, B sai
    n11 = int(np.sum(correct_A & correct_B))   # Cả 2 đều đúng
    
    table = [[n11, n10],
             [n01, n00]]
             
    # Sử dụng kiểm định chính xác nhị thức (Exact Binomial) nếu tổng số mẫu bất đồng < 25,
    # ngược lại dùng tiệm cận Chi-square có hiệu chỉnh liên tục (Continuity Correction của Edwards)
    discordant_total = n01 + n10
    exact_mode = bool(discordant_total < 25)
    
    mcnemar_res = mcnemar(table, exact=exact_mode, correction=True)
    statistic = float(mcnemar_res.statistic)
    p_value = float(mcnemar_res.pvalue)
    
    odds_ratio = (n10 / n01) if n01 > 0 else (np.inf if n10 > 0 else 1.0)
    
    return {
        "model_A": name_A,
        "model_B": name_B,
        "contingency_table": {
            "both_correct_n11": n11,
            "A_correct_B_wrong_n10": n10,
            "A_wrong_B_correct_n01": n01,
            "both_wrong_n00": n00
        },
        "discordant_total": discordant_total,
        "exact_test": exact_mode,
        "statistic": statistic,
        "p_value": p_value,
        "odds_ratio": odds_ratio,
        "is_significant_at_05": bool(p_value < 0.05),
        "is_significant_at_01": bool(p_value < 0.01)
    }


def cross_validate_and_test(model_dict: dict, X: list[str], y: np.ndarray, 
                            k_folds: int = 10, random_state: int = 42) -> dict:
    """
    Thực hiện 10-Fold Stratified Cross-Validation trên cùng các fold dữ liệu.
    Kiểm tra giả định phân phối chuẩn (Shapiro-Wilk) của hiệu số F1.
    Chạy Paired t-Test (nếu chuẩn) hoặc Wilcoxon Signed-Rank Test (nếu phi chuẩn).
    Chạy One-way Repeated Measures ANOVA so sánh toàn bộ các mô hình.
    """
    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    X = np.array(X)
    y = np.array(y)
    
    model_names = list(model_dict.keys())
    fold_scores = {name: [] for name in model_names}
    
    print(f"[CV-Test] Đang thực hiện {k_folds}-Fold Stratified Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        
        for name, pipeline in model_dict.items():
            # Clone/fit pipeline on fold
            pipeline.fit(X_tr, y_tr)
            preds = pipeline.predict(X_va)
            score = f1_score(y_va, preds, average='weighted')
            fold_scores[name].append(score)
            
    # Phân tích ANOVA 1 yếu tố lặp lại (One-way Repeated Measures ANOVA)
    anova_data = [fold_scores[name] for name in model_names]
    f_stat, anova_p = stats.f_oneway(*anova_data)
    
    # Kiểm định từng cặp mô hình (Pairwise Comparisons)
    pairwise_tests = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            nA = model_names[i]
            nB = model_names[j]
            sA = np.array(fold_scores[nA])
            sB = np.array(fold_scores[nB])
            diff = sA - sB
            
            # Kiểm định phân phối chuẩn của hiệu số (Shapiro-Wilk)
            shapiro_stat, shapiro_p = stats.shapiro(diff)
            is_normal = bool(shapiro_p > 0.05)
            
            if is_normal:
                t_stat, p_val = stats.ttest_rel(sA, sB)
                test_name = "Paired Student's t-test"
            else:
                t_stat, p_val = stats.wilcoxon(sA, sB)
                test_name = "Wilcoxon Signed-Rank Test"
                
            mean_diff = float(np.mean(diff))
            pairwise_tests.append({
                "pair": f"{nA} vs {nB}",
                "mean_diff": mean_diff,
                "shapiro_normality_p": float(shapiro_p),
                "assumption_normal": is_normal,
                "test_used": test_name,
                "statistic": float(t_stat),
                "p_value": float(p_val),
                "is_significant_05": bool(p_val < 0.05)
            })
            
    return {
        "fold_scores": fold_scores,
        "anova": {
            "f_statistic": float(f_stat),
            "p_value": float(anova_p),
            "is_significant_05": bool(anova_p < 0.05)
        },
        "pairwise_tests": pairwise_tests
    }


def analyze_latency_distributions(latency_dict: dict) -> dict:
    """
    Phân tích phân phối độ trễ suy luận (Inference Latency - ms) cho từng mô hình:
    - Thống kê mô tả: Mean, Std, Median, IQR, P95, P99.
    - Kiểm định Kolmogorov-Smirnov kiểm tra sự phù hợp phân phối Log-Normal.
    - So sánh thống kê phi tham số (Mann-Whitney U Test) giữa các cặp mô hình.
    """
    summary = {}
    names = list(latency_dict.keys())
    
    for name in names:
        lat = np.array(latency_dict[name])
        mean_val = float(np.mean(lat))
        std_val = float(np.std(lat, ddof=1))
        median_val = float(np.median(lat))
        p75, p25 = np.percentile(lat, [75, 25])
        iqr_val = float(p75 - p25)
        p95_val = float(np.percentile(lat, 95))
        p99_val = float(np.percentile(lat, 99))
        
        # Fit Log-Normal và kiểm tra Kolmogorov-Smirnov
        log_lat = np.log(lat[lat > 0])
        ks_stat, ks_p = stats.kstest(log_lat, 'norm', args=(np.mean(log_lat), np.std(log_lat, ddof=1)))
        
        summary[name] = {
            "mean_ms": mean_val,
            "std_ms": std_val,
            "median_ms": median_val,
            "iqr_ms": iqr_val,
            "p95_ms": p95_val,
            "p99_ms": p99_val,
            "lognormal_ks_statistic": float(ks_stat),
            "lognormal_ks_pvalue": float(ks_p)
        }
        
    # Mann-Whitney U test giữa các cặp mô hình
    pairwise_latency = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            nA, nB = names[i], names[j]
            latA = latency_dict[nA]
            latB = latency_dict[nB]
            u_stat, p_val = stats.mannwhitneyu(latA, latB, alternative='two-sided')
            
            faster = nA if np.median(latA) < np.median(latB) else nB
            speedup = float(np.median(latB) / np.median(latA)) if faster == nA else float(np.median(latA) / np.median(latB))
            
            pairwise_latency.append({
                "pair": f"{nA} vs {nB}",
                "u_statistic": float(u_stat),
                "p_value": float(p_val),
                "is_significant_05": bool(p_val < 0.05),
                "faster_model": faster,
                "speedup_ratio": speedup
            })
            
    return {
        "summary_statistics": summary,
        "pairwise_latency_tests": pairwise_latency
    }
