# 📊 Statistical AI Benchmarking: Empirical Evaluation & Hypothesis Testing for AI/ML Models

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-003366?style=flat)](https://www.statsmodels.org/)

> **Course:** Computational Statistics and Applications  
> **Researcher:** Tran Hoang Anh (Student ID: 24684541)  
> **Domain:** Machine Learning, Natural Language Processing, Statistical Inference, AI System Benchmarking  

---

## 🎯 1. Problem Statement & Motivation

In modern Artificial Intelligence and Data Science engineering, practitioners frequently make critical architectural decisions based exclusively on **point estimate metrics (e.g., single Accuracy or F1-Score numbers) evaluated on a static test split**.

From a rigorous mathematical standpoint, this practice suffers from severe methodological flaws:
1. **Sampling Variance:** A small observed performance difference (e.g., $95.17\%$ vs. $94.80\%$) is often indistinguishable from random sampling noise.
2. **Intractable Analytic Distributions:** Complex harmonic metrics like $F_1\text{-score}$ do not have closed-form sampling distributions, rendering standard asymptotic Wald intervals ($\hat{p} \pm z\sqrt{\hat{p}(1-\hat{p})/n}$) invalid.
3. **Paired Dependency:** Evaluating multiple competing models on the same test corpus creates pairwise dependencies that violate standard two-sample independent test assumptions.

### 🔬 Core Objectives:
This repository provides a production-grade, reproducible **Statistical AI Benchmarking Framework** that unifies Classical Inferential Statistics with Modern AI Evaluation:
* **Bootstrap Resampling ($B = 2,000$ iterations)** to construct empirical 95% Confidence Intervals for Accuracy, Precision, Recall, F1, and ROC-AUC.
* **McNemar's Contingency Test with Continuity Correction** to rigorously evaluate classification error differences between paired models.
* **10-Fold Stratified Cross-Validation + Shapiro-Wilk Normality Screening, Paired t-Test / Wilcoxon Signed-Rank Test, and Repeated Measures ANOVA** to control Family-Wise Error Rate (FWER) across data partitions.
* **Empirical Latency Modeling ($P95, P99$ SLAs)** and **Mann-Whitney U Rank-Sum Tests** to evaluate real-time throughput and computational cost.

---

## 🏗️ 2. Three Competing AI Paradigms

| Model Paradigm | Architecture | Hyperparameters / Features | Strengths & Trade-offs |
| :--- | :--- | :--- | :--- |
| **Model A: Statistical Baseline** | **Logistic Regression** | TF-IDF N-grams $(1, 2)$, $2,500$ max features, $L_2$ ($C=1.0$), L-BFGS solver | Sub-millisecond latency, convex optimization, high interpretability ($\beta_i$ coefficients). |
| **Model B: Ensemble ML** | **Random Forest Classifier** | 100 decision trees, max depth of 20, Bootstrap aggregation | Captures non-linear feature interactions; higher latency and memory footprint. |
| **Model C: Deep Neural Network** | **Multi-Layer Perceptron (MLP)** | Dense layers $(128, 64)$, ReLU activation, Adam optimizer, Early Stopping | Non-linear latent representation; requires balanced regularization to prevent overfitting. |

---

## 📁 3. Repository Architecture

```text
Statistical_AI_Benchmarking/
│
├── data/
│   ├── raw/sentiment_benchmark.csv             # Benchmark corpus (3,000 observations)
│   └── processed/                              # Stratified partitions (Train: 2100, Val: 300, Test: 600)
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── src/                                        # Modular business logic
│   ├── __init__.py
│   ├── data_loader.py                          # Data generation, cleaning, stratified splitting
│   ├── models.py                               # Pipeline management & single-sample latency timing
│   ├── statistical_tests.py                    # Bootstrap, McNemar, ANOVA, Wilcoxon, Mann-Whitney U
│   └── visualizer.py                           # Publication-grade scientific plotting (300 DPI)
│
├── experiments/
│   └── run_benchmark.py                        # Automated master pipeline runner
│
├── results/
│   ├── metrics_summary.csv                     # Point estimates and 95% CIs
│   ├── statistical_tests.json                  # Test statistics, degrees of freedom, p-values
│   └── figures/                                # High-resolution scientific figures
│       ├── forest_plot_ci.png                  # 95% Bootstrap Confidence Intervals Forest Plot
│       ├── mcnemar_contingency.png             # McNemar's 2x2 Contingency Heatmap
│       ├── cv_scores_comparison.png            # 10-Fold CV score distributions & ANOVA
│       ├── latency_distribution.png            # Inference Latency KDE & P95/P99 SLA lines
│       └── roc_curves.png                      # Receiver Operating Characteristic curves
│
├── notebooks/
│   ├── create_notebook.py                      # Script generating the reproducible notebook
│   └── statistical_ai_benchmark.ipynb          # Fully executed Jupyter Notebook with inline outputs
│
├── requirements.txt                            # Pinned Python package dependencies
├── PROJECT_REPORT.md                           # Full formal research report (English)
├── BAO_CAO_DU_AN.md                            # Comprehensive academic report (Vietnamese)
└── README.md                                   # This technical guide
```

---

## 📊 4. Empirical Benchmark Results

### A. Point Estimates vs. 95% Bootstrap Confidence Intervals ($B = 2,000$ iterations)

| Model Architecture | Accuracy | 95% Bootstrap CI (Accuracy) | Weighted F1-Score | 95% Bootstrap CI (F1-Score) | ROC-AUC (95% CI) | Median Latency ($P50$) | Tail Latency ($P95$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.9517** | **[0.9333, 0.9683]** | **0.9517** | **[0.9333, 0.9683]** | **[0.9277, 0.9698]** | **2.10 ms** | **2.77 ms** |
| **Random Forest** | **0.9517** | **[0.9333, 0.9683]** | **0.9517** | **[0.9333, 0.9683]** | **[0.9313, 0.9712]** | **23.75 ms** | **30.03 ms** |
| **Multi-Layer Perceptron** | **0.9517** | **[0.9333, 0.9683]** | **0.9517** | **[0.9333, 0.9683]** | **[0.9288, 0.9683]** | **2.20 ms** | **2.92 ms** |

---

### B. Hypothesis Testing Summary

1. **McNemar's Paired Classification Test:**
   * Contingency matrix for all model pairs: $[n_{11}=571, n_{10}=0, n_{01}=0, n_{00}=29]$.
   * Test statistic $\chi^2 = 0.0000$, $p\text{-value} = 1.0000 > 0.05$.
   * $\rightarrow$ **Fail to reject $H_0$:** Error distributions are statistically identical on the independent test set.
2. **10-Fold Cross-Validation & Repeated Measures ANOVA:**
   * ANOVA statistic $F = 0.3013$, $p\text{-value} = 0.7423 > 0.05$.
   * Shapiro-Wilk test on differences revealed non-normality ($p = 0.0006 < 0.05$).
   * Non-parametric **Wilcoxon Signed-Rank Test**: $W = 0.0$, $p = 0.125 > 0.05$.
   * $\rightarrow$ **Fail to reject $H_0$:** The $0.24\%$ mean fold variation between LR and RF is statistically insignificant.
3. **Inference Latency & Mann-Whitney U Tests:**
   * **Logistic Regression vs. Random Forest:** $U = 0.0$, $p = 1.43 \times 10^{-197}$. **Logistic Regression is $11.30\times$ faster ($p < 0.001$)**.
   * **Multi-Layer Perceptron vs. Random Forest:** $U = 360,000.0$, $p = 1.43 \times 10^{-197}$. **MLP is $10.81\times$ faster ($p < 0.001$)**.
   * **Logistic Regression vs. MLP:** $U = 122,064.5$, $p = 4.83 \times 10^{-22}$. **Logistic Regression is $1.05\times$ faster ($p < 0.001$)**.

---

## 💡 5. Production Engineering Decision (Occam's Razor)

* **Production Winner:** **Logistic Regression** is the clear choice for production serving (Real-time APIs, microservices, stream classification). It achieves accuracy statistically indistinguishable from ensemble and deep learning methods ($p = 0.7423$) while executing **$11.3\times$ faster**, requiring minimal RAM, and offering full interpretability.
* **Core Principle:** *"Among competing architectures yielding statistically indistinguishable accuracy, the model with the minimal operational complexity and lowest tail latency should always be deployed."*

---

## ⚡ 6. Quick Start & Reproducibility

### 1. Clone & Install
```bash
git clone https://github.com/anhtran-cs/Statistical_AI_Benchmarking.git
cd Statistical_AI_Benchmarking
pip install -r requirements.txt
```

### 2. Run Full Benchmark Pipeline
```bash
python -X utf8 experiments/run_benchmark.py
```

### 3. Launch Interactive Analysis
```bash
jupyter notebook notebooks/statistical_ai_benchmark.ipynb
```

---

## 📚 7. Formal Research Reports

* 🇬🇧 **[PROJECT_REPORT.md](PROJECT_REPORT.md)** - Full English Academic Research Report with comprehensive mathematical derivations, formulas, and statistical proofs.
* 🇻🇳 **[BAO_CAO_DU_AN.md](BAO_CAO_DU_AN.md)** - Báo cáo khoa học học thuật tiếng Việt chi tiết toàn diện.
