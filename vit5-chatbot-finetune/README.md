# vit5-chatbot-finetune

## Hướng dẫn sử dụng

1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
2. Chuẩn bị dữ liệu tại `data/chatbot_data.jsonl`
3. Tiền xử lý dữ liệu:
   ```bash
   python scripts/preprocess.py
   ```
4. Huấn luyện mô hình:
   ```bash
   python scripts/train.py
   ```
5. Suy luận với mô hình:
   ```bash
   python scripts/inference.py
   ```

## Cấu trúc thư mục
- data/: Dữ liệu hội thoại
- notebooks/: Notebook khám phá dữ liệu/model
- scripts/: Các script xử lý, huấn luyện, suy luận
- models/: Lưu checkpoint mô hình
