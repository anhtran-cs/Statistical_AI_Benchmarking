# 📊 Statistical AI Benchmarking: Đánh Giá Thống Kê Hiệu Năng Các Mô Hình AI/ML

> **Đồ án môn học:** Thống kê máy tính và ứng dụng (Computational Statistics & Applications)  
> **Tác giả:** Trần Hoàng Anh (MSSV: 24684541)  
> **Lĩnh vực:** Machine Learning, Natural Language Processing, Statistical Inference, AI Evaluation  

---

## 🎯 1. Đặt Vấn Đề & Mục Tiêu Nghiên Cứu
Trong kỷ nguyên phát triển bùng nổ của Trí tuệ Nhân tạo (AI) và Học máy (Machine Learning), các nhóm nghiên cứu và kỹ sư thường gặp phải một thiếu sót nghiêm trọng: **Chỉ dựa vào các con số đo lường đơn lẻ (Point Estimates) như Accuracy, F1-score trên một tập dữ liệu kiểm tra cố định để tuyên bố mô hình A vượt trội hơn mô hình B**.

Về mặt Thống kê học, sự chênh lệch này hoàn toàn có thể xuất phát từ:
- Yếu tố ngẫu nhiên trong việc phân bổ mẫu kiểm tra (Sample Variance).
- Sai số lấy mẫu (Sampling Error).
- Nhiễu ngẫu nhiên trong quá trình khởi tạo trọng số (Random Seed Effect).

**Mục tiêu của Dự án:**
Xây dựng một khung đánh giá thực nghiệm chuẩn mực thống kê (Rigorous Statistical Benchmarking Framework) kết hợp lý thuyết Thống kê suy diễn với thực nghiệm Học máy hiện đại trên bài toán Phân loại Sắc thái Văn bản (Sentiment Analysis):
1. **Ước lượng Khoảng tin cậy 95% (95% Confidence Intervals)** bằng kỹ thuật **Mô phỏng Bootstrap (Empirical Bootstrap Resampling)** với $B = 2,000$ vòng lặp.
2. **Kiểm định McNemar (McNemar's Test)** với hiệu chỉnh liên tục (Edwards Continuity Correction) và xác suất nhị thức chính xác (Exact Binomial Test) để kiểm định sự khác biệt có ý nghĩa thống kê về tỷ lệ phân loại sai giữa các cặp mô hình.
3. **10-Fold Stratified Cross-Validation kết hợp Kiểm định Cặp (Paired t-Test / Wilcoxon Signed-Rank Test)** và **Repeated Measures ANOVA** nhằm kiểm soát độ biến thiên trên nhiều phân vùng dữ liệu khác nhau.
4. **Phân tích Phân phối Độ trễ Suy luận (Inference Latency Distribution)**: Kiểm định sự phù hợp của phân phối Log-Normal và thực hiện kiểm định phi tham số **Mann-Whitney U Test** để so sánh tốc độ thực thi tại các mốc phân vị $P95, P99$.

---

## 🏗️ 2. Ba Mô Hình AI Đại Diện (Three Paradigms)

Dự án lựa chọn 3 mô hình đại diện cho 3 trường phái kiến trúc khác nhau trong Học máy:
1. **Model A (Statistical Baseline): Logistic Regression**
   - Bộ trích xuất đặc trưng TF-IDF N-grams $(1, 2)$, chuẩn hóa $L_2$, giải thuật tối ưu L-BFGS.
   - Ưu điểm: Tốc độ suy luận cực nhanh, tính giải thích cao, cơ sở toán học lồi (Convex Optimization).
2. **Model B (Ensemble Tree-based Machine Learning): Random Forest Classifier**
   - 150 cây quyết định, độ sâu tối đa 25, lấy mẫu túi (Bagging) và chọn lọc thuộc tính ngẫu nhiên.
   - Ưu điểm: Nắm bắt tương tác phi tuyến tính giữa các từ khóa, chống overfitting tốt.
3. **Model C (Deep Learning Representation): Multi-Layer Perceptron (MLP)**
   - Mạng nơ-ron truyền thẳng 2 tầng ẩn $(128, 64)$, hàm kích hoạt ReLU, cơ chế Early Stopping chống quá khớp.
   - Ưu điểm: Khả năng biểu diễn không gian đặc trưng phi tuyến tính phức tạp.

---

## 📁 3. Cấu Trúc Thư Mục Dự Án

```text
D:\Data\Project git\Statistical_AI_Benchmarking\
│
├── data\
│   ├── raw\                    # Dữ liệu gốc Benchmark Corpus (3000 quan sát)
│   │   └── sentiment_benchmark.csv
│   └── processed\              # Dữ liệu phân tầng Train (70%), Val (10%), Test (20%)
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── src\                        # Mã nguồn cốt lõi (Core Modules)
│   ├── __init__.py
│   ├── data_loader.py          # Tiền xử lý, chuẩn hóa tiếng Việt, stratified split
│   ├── models.py               # Xây dựng Pipeline 3 mô hình, đo latency từng mẫu
│   ├── statistical_tests.py    # Kiểm định thống kê (Bootstrap, McNemar, ANOVA, Wilcoxon, Mann-Whitney U)
│   └── visualizer.py           # Xuất bản biểu đồ khoa học (Forest Plot, Heatmap, KDE, ROC)
│
├── experiments\
│   └── run_benchmark.py        # Master Script thực thi toàn bộ pipeline tự động
│
├── results\
│   ├── metrics_summary.csv     # Bảng tổng hợp điểm ước lượng & 95% CI
│   ├── statistical_tests.json  # Giá trị thống kê chi tiết, p-value, bậc tự do
│   └── figures\                # Biểu đồ khoa học xuất bản chất lượng cao (300 DPI)
│       ├── forest_plot_ci.png
│       ├── mcnemar_contingency.png
│       ├── cv_scores_comparison.png
│       ├── latency_distribution.png
│       └── roc_curves.png
│
├── notebooks\
│   └── statistical_ai_benchmark.ipynb # Jupyter Notebook tương tác trực quan
│
├── requirements.txt            # Danh sách thư viện phụ thuộc
├── BAO_CAO_DU_AN.md            # Báo cáo học thuật chính thức toàn diện
└── README.md                   # Tài liệu hướng dẫn này
```

---

## ⚙️ 4. Hướng Dẫn Cài Đặt & Chạy Thực Nghiệm

### Bước 1: Kích hoạt môi trường và cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 2: Chạy toàn bộ Benchmark & Xuất kết quả
```bash
python -X utf8 experiments/run_benchmark.py
```

### Bước 3: Khám phá Notebook tương tác
```bash
jupyter notebook notebooks/statistical_ai_benchmark.ipynb
```

---

## 📈 5. Tóm Tắt Các Kỹ Thuật Thống Kê Được Áp Dụng

1. **Bootstrap Percentile Method:**
   $$SE_{boot} = \sqrt{\frac{1}{B-1}\sum_{b=1}^{B} (\theta_b^* - \bar{\theta}^*)^2}, \quad CI_{95\%} = \left[\theta^*_{\alpha/2}, \; \theta^*_{1 - \alpha/2}\right]$$
2. **Kiểm định McNemar (Continuity Correction):**
   $$\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}} \sim \chi^2(1)$$
3. **Kiểm định Cặp (Paired Differences):**
   - Kiểm tra giả định phân phối chuẩn hiệu số bằng **Shapiro-Wilk Test**:
     $$W = \frac{(\sum a_i d_{(i)})^2}{\sum (d_i - \bar{d})^2}$$
   - Nếu $p > 0.05$: Chạy **Paired t-Test**: $t = \frac{\bar{d}}{s_d / \sqrt{k}}$.
   - Nếu $p \le 0.05$: Chạy **Wilcoxon Signed-Rank Test** (Phi tham số).
4. **Repeated Measures ANOVA:**
   $$F = \frac{MS_{Between}}{MS_{Within}} \sim F(k-1, N-k)$$
5. **Mann-Whitney U Test cho Độ trễ Suy luận:**
   $$U = R_1 - \frac{n_1(n_1 + 1)}{2}$$
