# 📑 ACADEMIC RESEARCH PROJECT REPORT
## PROJECT TITLE: COMPUTATIONAL STATISTICS & EMPIRICAL HYPOTHESIS TESTING FOR ARTIFICIAL INTELLIGENCE & MACHINE LEARNING MODEL BENCHMARKING
### *(Statistical Hypothesis Testing & Computational Inference for AI Model Evaluation)*

* **Course:** Computational Statistics and Applications
* **Author / Researcher:** Tran Hoang Anh
* **Student ID:** 24684541
* **Completion Date:** November 07, 2026
* **Repository:** [https://github.com/anhtran-cs/Statistical_AI_Benchmarking](https://github.com/anhtran-cs/Statistical_AI_Benchmarking)
* **Local Project Directory:** `D:\Data\Project git\Statistical_AI_Benchmarking`

---

## TABLE OF CONTENTS
1. [EXECUTIVE SUMMARY & PROBLEM STATEMENT](#1-executive-summary--problem-statement)
2. [RIGOROUS STATISTICAL INFERENCE METHODOLOGY](#2-rigorous-statistical-inference-methodology)
   - 2.1. Empirical Bootstrap Resampling (Sampling Distribution & 95% CIs)
   - 2.2. McNemar's Test for Paired Classifiers
   - 2.3. 10-Fold Stratified Cross-Validation & Repeated Measures ANOVA
   - 2.4. Inference Latency Modeling & Non-parametric Mann-Whitney U Testing
3. [EXPERIMENTAL ARCHITECTURE & BENCHMARK SUITE](#3-experimental-architecture--benchmark-suite)
   - 3.1. Benchmark Corpus & Linguistic Characteristics
   - 3.2. Three Competing AI Paradigms
4. [EMPIRICAL FINDINGS & STATISTICAL ANALYSIS](#4-empirical-findings--statistical-analysis)
   - 4.1. Point Estimates vs. 95% Bootstrap Confidence Intervals
   - 4.2. McNemar's Contingency Test on Independent Test Data
   - 4.3. 10-Fold CV, Normality Assessment (Shapiro-Wilk) & ANOVA
   - 4.4. Extreme Value Latency Distribution ($P95/P99$) & Mann-Whitney U Tests
5. [ENGINEERING TRADE-OFFS & PRODUCTION RECOMMENDATIONS](#5-engineering-trade-offs--production-recommendations)
6. [REPRODUCIBILITY & SETUP GUIDE](#6-reproducibility--setup-guide)
7. [CONCLUSION](#7-conclusion)

---

## 1. EXECUTIVE SUMMARY & PROBLEM STATEMENT

In modern Artificial Intelligence and Machine Learning engineering, a critical flaw persists across industry and academia: **declaring a model superior based strictly on point estimate metrics (e.g., Accuracy, F1-score) calculated on a single, static test split**. 

From a mathematical and statistical standpoint:
1. **Sampling Variance:** Any finite evaluation dataset is merely a sample drawn from an underlying data generating distribution. A small difference in accuracy (e.g., $95.17\%$ vs. $94.80\%$) can be purely an artifact of sampling noise.
2. **Lack of Analytic Distribution:** Compound performance measures such as the harmonic mean $F_1\text{-score}$ do not possess closed-form asymptotic sampling distributions, rendering standard Wald confidence intervals ($\hat{p} \pm z\sqrt{\hat{p}(1-\hat{p})/n}$) mathematically invalid.
3. **Paired Dependency Violation:** Evaluating two competing models on the identical test set introduces strong pairwise dependency. Applying independent two-sample hypothesis tests violates fundamental statistical independence assumptions.

**Research Objective:**  
This project establishes a mathematically rigorous, reproducible benchmarking framework bridging Classical Statistical Inference and Modern AI Engineering on a Text Sentiment Analysis task:
* **Bootstrap Resampling ($B = 2,000$ iterations)** to construct empirical 95% Confidence Intervals for Accuracy, Precision, Recall, F1, and ROC-AUC.
* **McNemar's Test with Continuity Correction** to rigorously test whether classification error rates differ significantly ($p < 0.05$).
* **10-Fold Stratified Cross-Validation with Shapiro-Wilk Normality Screening, Paired t-Test / Wilcoxon Signed-Rank Test, and Repeated Measures ANOVA** to control Family-Wise Error Rate (FWER) across data partitions.
* **Extreme Value Latency Distribution Analysis ($P95, P99$ SLAs)** and **Mann-Whitney U Rank-Sum Tests** to evaluate computational efficiency and throughput.

---

## 2. RIGOROUS STATISTICAL INFERENCE METHODOLOGY

### 2.1. Empirical Bootstrap Resampling (Confidence Intervals)
Let $D_{test} = \{(x_1, y_1), \dots, (x_n, y_n)\}$ be the test dataset of size $n$. Because the true distribution of $F_1$ is analytically intractable, we apply Efron's non-parametric Bootstrap (1979):
1. Sample $n$ instances with replacement from $D_{test}$ to form a bootstrap replicate $D^{*(b)}$.
2. Compute the test metric $\theta^{*(b)} = \text{Metric}(D^{*(b)})$.
3. Repeat for $B = 2,000$ independent resamples.

The bootstrap standard error is estimated by:
$$SE_{boot} = \sqrt{\frac{1}{B-1} \sum_{b=1}^{B} \left(\theta^{*(b)} - \bar{\theta}^*\right)^2}$$
The 95% Percentile Confidence Interval with significance level $\alpha = 0.05$ is given by:
$$CI_{95\%} = \left[ \theta^*_{\alpha/2}, \; \theta^*_{1 - \alpha/2} \right] = \left[ \theta^*_{0.025}, \; \theta^*_{0.975} \right]$$

### 2.2. McNemar's Test for Paired Classifiers
McNemar's test is a non-parametric test designed specifically for comparing two binary classifiers evaluated on the identical test set. We construct a $2 \times 2$ contingency matrix:

| | Model B Correct | Model B Incorrect |
| :---: | :---: | :---: |
| **Model A Correct** | $n_{11}$ | $n_{10}$ |
| **Model A Incorrect** | $n_{01}$ | $n_{00}$ |

* $n_{10}$: Number of samples correctly classified by Model A but misclassified by Model B.
* $n_{01}$: Number of samples misclassified by Model A but correctly classified by Model B.
* $n_{11}$ and $n_{00}$ represent consensus cases and carry no discriminatory information.

**Statistical Hypotheses:**
* Null Hypothesis ($H_0$): $p_{01} = p_{10}$ (Marginal error rates of both classifiers are identical).
* Alternative Hypothesis ($H_1$): $p_{01} \neq p_{10}$ (Classifiers exhibit a statistically significant difference in classification errors).

**Test Statistic (Edwards' Continuity Correction):**
When $n_{01} + n_{10} \ge 25$, the test statistic asymptotically follows a Chi-square distribution with 1 degree of freedom:
$$\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}} \sim \chi^2(1)$$
If $n_{01} + n_{10} < 25$, the exact two-tailed binomial test with $p = 0.5$ is computed:
$$p\text{-value} = 2 \sum_{k=0}^{\min(n_{01}, n_{10})} \binom{n_{01} + n_{10}}{k} (0.5)^{n_{01} + n_{10}}$$
The Odds Ratio quantifying relative odds of disagreement is: $OR = \frac{n_{10}}{n_{01}}$.

### 2.3. 10-Fold Stratified Cross-Validation & Repeated Measures ANOVA
Across 10 stratified folds ($k = 10$), each model yields an $F_1$ score on fold $i$. The paired difference vector is:
$$d_i = s_{A,i} - s_{B,i}, \quad i \in \{1, \dots, 10\}$$

**Methodological Decision Protocol:**
1. **Normality Assessment (Shapiro-Wilk Test):**
   * If $p_{Shapiro} > 0.05$: Difference distribution is Gaussian $\rightarrow$ Run parametric **Paired Student's t-test**:
     $$t = \frac{\bar{d}}{s_d / \sqrt{k}} \sim t(k-1)$$
   * If $p_{Shapiro} \le 0.05$: Normality is violated $\rightarrow$ Run non-parametric **Wilcoxon Signed-Rank Test** based on ranked absolute differences.
2. **Repeated Measures ANOVA:**  
   When comparing $m = 3$ models simultaneously across 10 folds, ANOVA tests whether the overall main effect of model architecture is significant while controlling Family-Wise Error Rate (FWER):
   $$F = \frac{MS_{Between}}{MS_{Error}} \sim F(m-1, (m-1)(k-1))$$

### 2.4. Inference Latency Distribution & Mann-Whitney U Testing
Computational latency $T$ in computing systems exhibits heavy right skewness and extreme tail events:
1. **Log-Normal Fit & Goodness-of-fit:** We model $Y = \ln(T) \sim \mathcal{N}(\mu, \sigma^2)$ and verify fit via the **Kolmogorov-Smirnov (KS) test**.
2. **SLA Benchmarks ($P95, P99$):** The 95th and 99th percentiles capture tail latency critical for cloud service level agreements (SLAs).
3. **Mann-Whitney U Test (Rank-Sum Test):** Evaluates whether one model's latency distribution is stochastically smaller than another without assuming normality:
   $$U = n_1 n_2 + \frac{n_1(n_1 + 1)}{2} - R_1$$

---

## 3. EXPERIMENTAL ARCHITECTURE & BENCHMARK SUITE

### 3.1. Benchmark Corpus & Linguistic Characteristics
* **Size:** $N = 3,000$ authentic customer review texts.
* **Task:** Binary Sentiment Classification (Class 0: Negative, Class 1: Positive).
* **Dataset Nuance & Realism:**
  * Balanced 50:50 prior distribution.
  * 20% complex mixed reviews (e.g., dual positive/negative sentiment, conditional clauses, sarcasm).
  * 4% synthetic label noise simulating human annotation ambiguity in real crowdsourced data.
* **Stratified Data Partitioning:**
  * **Train Set:** 2,100 samples (70%)
  * **Validation Set:** 300 samples (10%)
  * **Test Set:** 600 independent samples (20%)

### 3.2. Three Competing AI Paradigms
1. **Model A (Statistical Baseline): Logistic Regression**
   * Feature Extractor: TF-IDF with N-grams $(1, 2)$, maximum $2,500$ features, $L_2$ regularization ($C = 1.0$), L-BFGS solver.
2. **Model B (Ensemble Tree-based ML): Random Forest Classifier**
   * Hyperparameters: $100$ estimators, maximum tree depth of $20$, Gini impurity criterion.
3. **Model C (Deep Learning / Dense Neural Network): Multi-Layer Perceptron (MLP)**
   * Architecture: Fully connected dense layers $(128, 64)$, ReLU activation, Adam optimizer, early stopping with 10-epoch patience.

---

## 4. EMPIRICAL FINDINGS & STATISTICAL ANALYSIS

All results were generated systematically and recorded in `results/metrics_summary.csv` and `results/statistical_tests.json`.

### 4.1. Point Estimates vs. 95% Bootstrap Confidence Intervals

Performance evaluated on the independent test split ($N_{test} = 600$) with $B = 2,000$ bootstrap iterations:

| Model Architecture | Accuracy (Point Est.) | 95% Bootstrap CI (Accuracy) | Weighted F1-Score | 95% Bootstrap CI (F1-Score) | ROC-AUC (95% CI) | Median Latency ($P50$) | 95th Percentile Latency ($P95$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.9517** | **[0.9333, 0.9683]** | **0.9517** | **[0.9333, 0.9683]** | **[0.9277, 0.9698]** | **2.10 ms** | **2.77 ms** |
| **Random Forest** | **0.9517** | **[0.9333, 0.9683]** | **0.9517** | **[0.9333, 0.9683]** | **[0.9313, 0.9712]** | **23.75 ms** | **30.03 ms** |
| **Multi-Layer Perceptron** | **0.9517** | **[0.9333, 0.9683]** | **0.9517** | **[0.9333, 0.9683]** | **[0.9288, 0.9683]** | **2.20 ms** | **2.92 ms** |

*Scientific Forest Plot available at: `results/figures/forest_plot_ci.png`.*

**Statistical Insights:**
1. All three models achieve an empirical point estimate of $0.9517$ on the test set.
2. The 95% Bootstrap confidence intervals span $[0.9333, 0.9683]$ with $SE_{boot} \approx 0.0088$. This confirms model stability above $93.3\%$ with 95% probability.
3. Random Forest achieves a marginally higher ROC-AUC point estimate ($0.9522$) compared to Logistic Regression ($0.9509$) and MLP ($0.9498$), though their confidence intervals substantially overlap.

---

### 4.2. McNemar's Contingency Test on Independent Test Data

To verify whether models commit errors on identical instances, pairwise McNemar tests were conducted:

| Model Comparison | Contingency Matrix $[n_{11}, n_{10}, n_{01}, n_{00}]$ | $\chi^2$ Statistic | $p\text{-value}$ | Statistical Conclusion ($\alpha = 0.05$) |
| :--- | :---: | :---: | :---: | :--- |
| **LR vs Random Forest** | $[571, 0, 0, 29]$ | $0.0000$ | $1.0000$ | **Fail to Reject $H_0$** (No significant error difference) |
| **LR vs MLP** | $[571, 0, 0, 29]$ | $0.0000$ | $1.0000$ | **Fail to Reject $H_0$** (No significant error difference) |
| **Random Forest vs MLP** | $[571, 0, 0, 29]$ | $0.0000$ | $1.0000$ | **Fail to Reject $H_0$** (No significant error difference) |

*Contingency heatmap available at: `results/figures/mcnemar_contingency.png`.*

**Interpretation:** All three architectures achieved identical consensus on $571$ correct classifications and $29$ error instances ($n_{10} = 0, n_{01} = 0$). With zero discordant pairs, there is no evidence of differential classification bias on the test partition.

---

### 4.3. 10-Fold Cross-Validation, Normality Assessment & ANOVA

Across 10 stratified folds of the entire corpus ($N = 3,000$):
* **Mean 10-Fold F1 Scores:**
  * Logistic Regression: $\mathbf{0.9567}$
  * Random Forest: $\mathbf{0.9543}$
  * Multi-Layer Perceptron: $\mathbf{0.9567}$
* **Repeated Measures ANOVA:**
  * $F\text{-statistic} = 0.3013$
  * $p\text{-value} = 0.7423 > 0.05$
  * $\rightarrow$ **Conclusion:** Across multiple data partitions, the main effect of model architecture is **not statistically significant**.

**Pairwise Difference Testing:**
1. **Logistic Regression vs. Random Forest:**
   * Shapiro-Wilk test on differences: $p = 0.0006 < 0.05 \rightarrow$ Non-Gaussian differences.
   * **Wilcoxon Signed-Rank Test:** $W = 0.0$, $p = 0.125 > 0.05$.
   * $\rightarrow$ The $0.24\%$ mean difference is attributable to random fold variation.
2. **Random Forest vs. MLP:**
   * Shapiro-Wilk test on differences: $p = 0.0006 < 0.05 \rightarrow$ Non-Gaussian differences.
   * **Wilcoxon Signed-Rank Test:** $W = 0.0$, $p = 0.125 > 0.05$.
   * $\rightarrow$ Fail to reject $H_0$; performance differences are not statistically significant.

*CV score distribution boxplots available at: `results/figures/cv_scores_comparison.png`.*

---

### 4.4. Extreme Value Latency Distribution ($P95/P99$) & Mann-Whitney U Tests

While classification accuracy is statistically equivalent, **computational latency displays dramatic divergence**:

#### Latency Descriptive Statistics (per single-sample inference in milliseconds):
* **Logistic Regression:**
  * Mean: $2.22 \text{ ms}$ | Median ($P50$): $\mathbf{2.10 \text{ ms}}$ | IQR: $0.35 \text{ ms}$
  * 95th Percentile ($P95$): $\mathbf{2.77 \text{ ms}}$ | 99th Percentile ($P99$): $3.22 \text{ ms}$
  * Kolmogorov-Smirnov test for Log-Normal distribution: $p = 2.97 \times 10^{-20}$ (strong right skewness).
* **Multi-Layer Perceptron:**
  * Mean: $2.31 \text{ ms}$ | Median ($P50$): $\mathbf{2.20 \text{ ms}}$ | IQR: $0.31 \text{ ms}$
  * 95th Percentile ($P95$): $\mathbf{2.92 \text{ ms}}$ | 99th Percentile ($P99$): $3.18 \text{ ms}$
* **Random Forest (100 trees):**
  * Mean: $24.60 \text{ ms}$ | Median ($P50$): $\mathbf{23.75 \text{ ms}}$ | IQR: $1.46 \text{ ms}$
  * 95th Percentile ($P95$): $\mathbf{30.03 \text{ ms}}$ | 99th Percentile ($P99$): $36.73 \text{ ms}$

#### Mann-Whitney U Rank-Sum Hypothesis Tests:
1. **Logistic Regression vs. Random Forest:**
   * $U = 0.0$, $p\text{-value} = 1.43 \times 10^{-197} < 0.001$.
   * $\rightarrow$ **Logistic Regression is $11.30\times$ faster than Random Forest with extreme statistical significance ($p < 0.001$)**.
2. **Multi-Layer Perceptron vs. Random Forest:**
   * $U = 360,000.0$, $p\text{-value} = 1.43 \times 10^{-197} < 0.001$.
   * $\rightarrow$ **MLP is $10.81\times$ faster than Random Forest with extreme statistical significance ($p < 0.001$)**.
3. **Logistic Regression vs. Multi-Layer Perceptron:**
   * $U = 122,064.5$, $p\text{-value} = 4.83 \times 10^{-22} < 0.001$.
   * $\rightarrow$ **Logistic Regression is $1.05\times$ faster than MLP ($p < 0.001$)**.

*Empirical density latency distribution plot available at: `results/figures/latency_distribution.png`.*

---

## 5. ENGINEERING TRADE-OFFS & PRODUCTION RECOMMENDATIONS

### Multi-Criteria Engineering Trade-off Matrix:

| Evaluation Dimension | Logistic Regression | Random Forest | Multi-Layer Perceptron | Statistical Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy / F1-Score** | $95.17\%$ | $95.17\%$ | $95.17\%$ | No statistically significant difference ($p = 0.7423$) |
| **Median Latency ($P50$)** | **$2.10 \text{ ms}$** | $23.75 \text{ ms}$ | $2.20 \text{ ms}$ | LR is $11.3\times$ faster than RF ($p < 10^{-190}$) |
| **Tail Latency SLA ($P95$)** | **$2.77 \text{ ms}$** | $30.03 \text{ ms}$ | $2.92 \text{ ms}$ | LR provides superior SLA guarantees |
| **Memory Footprint** | Extremely Low ($< 10\text{MB}$) | Medium | Moderate | LR minimizes cloud infrastructure costs |
| **Model Interpretability** | Extremely High ($\beta_i$ coefficients) | Moderate | Low (Black-box) | LR enables full auditability |

### Production Engineering Decision:
* **Production Winner:** **Logistic Regression** is the unambiguous choice for real-time inference environments (High-throughput APIs, stream processing). It matches the accuracy of complex neural and ensemble architectures while executing **11 times faster**, requiring minimal memory, and offering full interpretability.
* **Scientific Takeaway (Occam's Razor):** *"Among competing models with statistically indistinguishable empirical accuracy, the model with the fewest assumptions and lowest operational complexity should always be selected."*

---

## 6. REPRODUCIBILITY & SETUP GUIDE

To reproduce all experiments, figures, and statistical tables:

```bash
# 1. Clone or navigate to the repository
cd "D:\Data\Project git\Statistical_AI_Benchmarking"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Execute the full end-to-end benchmark suite
python -X utf8 experiments/run_benchmark.py

# 4. Launch interactive Jupyter Notebook
jupyter notebook notebooks/statistical_ai_benchmark.ipynb
```

---

## 7. CONCLUSION

This project successfully demonstrates the essential role of **Computational Statistics in Artificial Intelligence**:
1. It replaces naive single-metric comparisons with rigorous, reproducible inference tools: **Bootstrap Resampling, McNemar's Test, Repeated Measures ANOVA, Wilcoxon Signed-Rank Testing, and Mann-Whitney U Testing**.
2. It bridges mathematical statistical theory with applied Computer Science engineering, establishing data-driven production deployment decisions.
3. All code, datasets, figures, and interactive notebooks are fully packaged and synchronized to the GitHub repository:  
   🔗 **[https://github.com/anhtran-cs/Statistical_AI_Benchmarking](https://github.com/anhtran-cs/Statistical_AI_Benchmarking)**
