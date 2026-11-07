"""
models.py
Module định nghĩa và quản lý vòng đời huấn luyện, dự đoán của 3 mô hình AI/ML:
1. Model A: Logistic Regression (Mô hình tuyến tính cơ sở - Statistical Baseline)
2. Model B: Random Forest Classifier (Mô hình cây học máy kết hợp - Ensemble Tree)
3. Model C: Multi-Layer Perceptron / Dense Neural Network (Mô hình mạng nơ-ron học sâu - Deep Learning Paradigm)
"""

import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class AIModelBenchmarkSuite:
    """
    Bộ quản lý và thực nghiệm 3 mô hình AI phục vụ kiểm định thống kê.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.vectorizer = None
        self._initialize_models()
        
    def _initialize_models(self):
        # 1. Statistical Linear Model (Logistic Regression)
        model_lr = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=2500, ngram_range=(1, 2))),
            ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=self.random_state))
        ])
        
        # 2. Non-linear Ensemble Tree (Random Forest)
        model_rf = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=2500, ngram_range=(1, 2))),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=20, random_state=self.random_state, n_jobs=1))
        ])
        
        # 3. Dense Neural Network (Multi-Layer Perceptron - Deep Learning)
        model_mlp = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=2500, ngram_range=(1, 2))),
            ('clf', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, 
                                  early_stopping=True, n_iter_no_change=10, 
                                  random_state=self.random_state))
        ])
        
        self.models = {
            "Logistic Regression": model_lr,
            "Random Forest": model_rf,
            "Multi-Layer Perceptron": model_mlp
        }

    def train_all(self, X_train: list[str], y_train: np.ndarray):
        """
        Huấn luyện toàn bộ 3 mô hình trên tập dữ liệu Train.
        """
        print("[ModelSuite] Bắt đầu huấn luyện các mô hình...")
        for name, pipeline in self.models.items():
            t0 = time.perf_counter()
            pipeline.fit(X_train, y_train)
            t_train = time.perf_counter() - t0
            print(f" -> Đã huấn luyện xong: {name:25s} (Thời gian: {t_train:.2f}s)")
            
    def predict_with_latency(self, model_name: str, X_test: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Thực hiện dự đoán và đo lường thời gian suy luận (Inference Latency tính theo ms) trên từng mẫu đơn lẻ.
        Trả về: (y_pred, y_prob, latencies_ms)
        """
        pipeline = self.models[model_name]
        y_preds = []
        y_probs = []
        latencies_ms = []
        
        # Đo lường thời gian suy luận cho từng quan sát để thu thập phân phối độ trễ (Latency Distribution)
        for text in X_test:
            t0 = time.perf_counter()
            pred = pipeline.predict([text])[0]
            prob = pipeline.predict_proba([text])[0, 1]
            t_elapsed = (time.perf_counter() - t0) * 1000.0  # ms
            
            y_preds.append(pred)
            y_probs.append(prob)
            latencies_ms.append(t_elapsed)
            
        return np.array(y_preds), np.array(y_probs), np.array(latencies_ms)

    def evaluate_all(self, X_test: list[str], y_test: np.ndarray) -> dict:
        """
        Đánh giá toàn bộ mô hình trên tập Test và thu thập các chỉ số cơ bản.
        """
        results = {}
        for name in self.models.keys():
            y_pred, y_prob, latencies = self.predict_with_latency(name, X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            auc = roc_auc_score(y_test, y_prob)
            
            results[name] = {
                "y_pred": y_pred,
                "y_prob": y_prob,
                "latencies_ms": latencies,
                "accuracy": acc,
                "f1_score": f1,
                "precision": prec,
                "recall": rec,
                "roc_auc": auc
            }
            print(f"[ModelSuite] {name:25s} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Mean Latency: {np.mean(latencies):.2f}ms")
            
        return results
