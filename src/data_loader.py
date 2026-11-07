"""
data_loader.py
Module tải, tiền xử lý và phân chia dữ liệu đánh giá thực nghiệm cho mô hình AI.
Hỗ trợ cả việc đọc từ file CSV thực tế và tự động tạo tập benchmark chuẩn hóa (Benchmark Corpus).
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """
    Tiền xử lý văn bản: chuẩn hóa chữ thường, loại bỏ ký tự lạ, làm sạch khoảng trắng.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Loại bỏ URL và email
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    # Loại bỏ ký tự đặc biệt, giữ lại chữ cái tiếng Việt/tiếng Anh và dấu cơ bản
    text = re.sub(r'[^\w\s\d.,!?]', ' ', text)
    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def generate_benchmark_sentiment_data(n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
    """
    Sinh tập dữ liệu Benchmark đánh giá sắc thái văn bản (Sentiment Analysis) thực tế
    với 2 lớp: 0 (Tiêu cực / Negative) và 1 (Tích cực / Positive).
    Tập dữ liệu mô phỏng phản hồi khách hàng sàn thương mại điện tử / dịch vụ công nghệ.
    """
    np.random.seed(random_state)
    
    positive_templates = [
        "Sản phẩm rất tốt, giao hàng nhanh chóng, đóng gói cẩn thận.",
        "Chất lượng tuyệt vời, dùng rất mượt và ổn định, rất đáng tiền.",
        "Dịch vụ hỗ trợ khách hàng nhiệt tình, giải quyết vấn đề nhanh.",
        "Thiết kế đẹp mắt, pin trâu, màn hình hiển thị sắc nét.",
        "Rất hài lòng với trải nghiệm lần này, sẽ tiếp tục ủng hộ shop.",
        "Giao diện ứng dụng trực quan, dễ sử dụng, tính năng thông minh.",
        "Đúng như mô tả, chất lượng vượt ngoài mong đợi trong tầm giá.",
        "Máy chạy êm, cấu hình mạnh mẽ, xử lý tác vụ AI rất nhanh.",
        "Đóng gói kỹ 2 lớp chống sốc, hàng chính hãng nguyên seal xịn sò.",
        "Tính năng nhận diện rất chuẩn xác, phản hồi cực kỳ nhanh nhạy.",
        "Tuyệt vời, 5 sao cho chất lượng sản phẩm và thái độ phục vụ.",
        "Đáng đồng tiền bát gạo, dùng cả tháng nay không gặp lỗi gì.",
        "Tốc độ xử lý ấn tượng, vượt trội so với phiên bản tiền nhiệm.",
        "Trải nghiệm mượt mà, âm thanh trong trẻo, kết nối ổn định.",
        "Cực kỳ ưng ý, hướng dẫn sử dụng rõ ràng, dễ cài đặt ban đầu."
    ]
    
    negative_templates = [
        "Sản phẩm quá tệ, vừa dùng được 2 ngày đã bị đơ màn hình.",
        "Giao hàng quá chậm, trễ hẹn 1 tuần mà không một lời giải thích.",
        "Chất lượng kém xa so với quảng cáo, nhựa ọp ẹp rất rẻ tiền.",
        "Dịch vụ chăm sóc khách hàng cực kỳ thất vọng, nhắn tin không trả lời.",
        "Ứng dụng liên tục bị văng crash, giật lag không thể thao tác được.",
        "Pin tụt như tụt dốc không phanh, máy rất nóng khi dùng tác vụ nhẹ.",
        "Hàng nhận được bị móp méo, trầy xước, có dấu hiệu đã qua sử dụng.",
        "Mô tả sản phẩm không đúng thực tế, kết nối bluetooth chập chờn.",
        "Hoàn toàn thất vọng, yêu cầu hoàn tiền nhưng thủ tục quá rườm rà.",
        "Lỗi vặt quá nhiều, bản cập nhật mới làm máy chạy chậm hẳn.",
        "Chất âm tệ, rè khi tăng âm lượng, micro bắt tiếng rất nhỏ.",
        "Phần mềm nhiều bug, tính năng AI nhận diện sai be bét.",
        "Quá thất vọng với số tiền bỏ ra, không bao giờ mua lại thương hiệu này.",
        "Đóng gói sơ sài, hộp rách nát, thái độ shipper cọc cằn.",
        "Máy nóng ran chỉ sau 10 phút sử dụng, sạc pin thì chập chờn."
    ]
    
    noise_modifiers = [
        "Thực sự là ", "Nhìn chung thì ", "Cá nhân mình thấy ", 
        "Đánh giá thật lòng là ", "Trải nghiệm thực tế cho thấy ", ""
    ]
    
    mixed_templates = [
        "Máy thiết kế rất đẹp, màn hình hiển thị tốt nhưng pin thì hơi đuối.",
        "Giao hàng nhanh, shipper nhiệt tình, nhưng sản phẩm dùng hơi nóng máy.",
        "Giá rẻ bất ngờ, dùng tạm ổn cho các tác vụ cơ bản, không đòi hỏi cao.",
        "Camera chụp ban ngày khá ổn, tuy nhiên ban đêm thì bị nhiễu hạt nhiều.",
        "Chất âm trong trẻo, nghe podcast rất thích nhưng âm bass hơi yếu.",
        "Dịch vụ quá 'tuyệt vời', đặt hàng từ tuần trước đến nay mới thèm phản hồi!",
        "Quảng cáo thì hoành tráng lắm, nhận hàng về thấy bình thường, tạm chấp nhận được.",
        "Dùng được 1 tuần đầu rất mượt, sang tuần thứ 2 bắt đầu thấy giật lag nhẹ.",
        "Đóng gói rất sang chảnh, mỗi tội hướng dẫn sử dụng toàn tiếng nước ngoài khó hiểu.",
        "Tính năng nhiều nhưng hơi rối mắt, người lớn tuổi chắc chắn khó làm quen."
    ]
    
    data = []
    half_samples = n_samples // 2
    
    # Tạo mẫu tích cực (Class 1)
    for i in range(half_samples):
        if np.random.rand() < 0.20:
            # Câu hỗn hợp hơi hướng tích cực
            text = np.random.choice(mixed_templates) + " Nhìn chung vẫn đáng mua trong phân khúc."
            label = 1
        else:
            base = np.random.choice(positive_templates)
            mod = np.random.choice(noise_modifiers)
            text = mod + base
            if np.random.rand() > 0.5:
                extra = np.random.choice([" Shop tư vấn có tâm.", " Khuyên mọi người nên mua nhé.", " Quá ok luôn!"])
                text += extra
            label = 1
            
        # Thêm 4% nhiễu gán nhãn thực tế (Human Annotation Noise)
        if np.random.rand() < 0.04:
            label = 0
            
        data.append({"text": text, "label": label})
        
    # Tạo mẫu tiêu cực (Class 0)
    for i in range(half_samples):
        if np.random.rand() < 0.20:
            # Câu hỗn hợp hơi hướng tiêu cực
            text = np.random.choice(mixed_templates) + " Thất vọng vì không được như kỳ vọng ban đầu."
            label = 0
        else:
            base = np.random.choice(negative_templates)
            mod = np.random.choice(noise_modifiers)
            text = mod + base
            if np.random.rand() > 0.5:
                extra = np.random.choice([" Mọi người nên né ra nhé.", " Thật sự bực mình.", " Đừng dại mà mua!"])
                text += extra
            label = 0
            
        # Thêm 4% nhiễu gán nhãn thực tế
        if np.random.rand() < 0.04:
            label = 1
            
        data.append({"text": text, "label": label})
        
    df = pd.DataFrame(data)
    # Xáo trộn ngẫu nhiên dữ liệu
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df["clean_text"] = df["text"].apply(clean_text)
    df["text_length"] = df["clean_text"].apply(lambda s: len(s.split()))
    
    return df


def load_or_create_dataset(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tải tập dữ liệu từ disk hoặc tự động sinh mới nếu chưa tồn tại.
    Phân chia thành tập Train (70%), Validation (10%), Test (20%) có phân tầng (Stratified).
    """
    raw_dir = os.path.join(data_dir, "raw")
    proc_dir = os.path.join(data_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    
    raw_path = os.path.join(raw_dir, "sentiment_benchmark.csv")
    train_path = os.path.join(proc_dir, "train.csv")
    val_path = os.path.join(proc_dir, "val.csv")
    test_path = os.path.join(proc_dir, "test.csv")
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        print(f"[DataLoader] Đã tải dữ liệu sẵn có: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df
    
    # Sinh dữ liệu nếu chưa có
    print("[DataLoader] Tạo mới tập dữ liệu Benchmark Corpus (N=3000 mẫu)...")
    full_df = generate_benchmark_sentiment_data(n_samples=3000, random_state=42)
    full_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    
    # Phân chia dữ liệu: Train 70%, Val 10%, Test 20%
    train_val_df, test_df = train_test_split(
        full_df, test_size=0.20, random_state=42, stratify=full_df["label"]
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.125, random_state=42, stratify=train_val_df["label"]
    ) # 0.8 * 0.125 = 0.10 tổng số
    
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")
    
    print(f"[DataLoader] Phân chia thành công: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df
