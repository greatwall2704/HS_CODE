from google import genai
from pydantic import BaseModel
import json
import asyncio
from tqdm.asyncio import tqdm
from collections import defaultdict
import pandas as pd
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_GENAI_API_KEY")

def group_by_hs_prefix(df):
    grouped = defaultdict(list)
    for _, row in df.iterrows():
        mahs_value = row['mahs']
        prefix = str(mahs_value)[:4]  # Chuyển sang chuỗi và lấy 4 ký tự đầu
        grouped[prefix].append(row.to_dict())  # Chuyển hàng thành từ điển
    return grouped

class QAItem(BaseModel):
    question: str
    answer: str

class QAList(BaseModel):
    items: list[QAItem]

def build_prompt_for_group(prefix: str, items: List[Dict]) -> str:
    items_list = "\n".join(
        f"- {item['mahs']}: {item['mo_ta']}" for item in items
    )
    num_questions = 5 * len(items)  # tính số câu hỏi theo yêu cầu

    prompt = f"""Bạn là chuyên gia phân loại mã HS Code. Dưới đây là danh sách các mã HS thuộc nhóm {prefix} cùng mô tả chi tiết:

{items_list}

---
**Mục tiêu:** Tạo ra **{num_questions} cặp câu hỏi và câu trả lời tự nhiên**, tương đương **5 cặp cho mỗi mã HS** trong nhóm.

---

✅ **Yêu cầu cụ thể**:
1. **Ngôn ngữ đời thường**, tự nhiên, thân thiện như trong hội thoại hằng ngày.
2. Câu hỏi mang tính **thực tiễn**, ví dụ:
   - “Tôi đang chuẩn bị nhập khẩu con X, thì dùng mã nào?”
   - “Hai mã này khác nhau ở chỗ nào khi làm thủ tục?”
   - “Loại nào áp dụng cho hàng giống và loại nào cho hàng không giống?”
3. Tập trung vào **so sánh, phân biệt, hướng dẫn chọn mã** giữa các mã HS trong cùng nhóm.
4. Nội dung phải **ngắn gọn nhưng dễ hiểu**, phù hợp với người không chuyên về hải quan hoặc HS code.
5. Tránh thuật ngữ chuyên ngành khó hiểu; thay vào đó, dùng ví dụ cụ thể, tình huống nhập hàng, kinh doanh, kê khai thực tế.

---

📦 **Đầu ra mong muốn**: Trả về đúng định dạng JSON sau:

{
  "items": [
    {
      "question": "Sự khác biệt giữa mã 01013010 và 01013090 là gì?",
      "answer": "01013010 dùng cho lừa thuần chủng để nhân giống, còn 01013090 là các loại lừa khác không dùng nhân giống."
    },
    ...
  ]
}

Lưu ý: Trả về **chỉ JSON**, không có giải thích hoặc mô tả nào thêm.
"""
    return prompt

async def fetch_hscode_qa_from_csv(grouped_data: dict, model="gemini-2.0-flash-001"):
    all_qas = []  # List gom tất cả câu hỏi-đáp từ các prefix
    client = genai.Client(api_key=api_key)
    for prefix, items in grouped_data.items():
        content_data = build_prompt_for_group(prefix, items)
        response = client.models.generate_content(
            model=model,
            contents=content_data,
            config={
                "response_mime_type": "application/json",
                "response_schema": QAList
            },
        )
        # Parse response.text JSON ra dict rồi lấy phần "items"
        data = json.loads(response.text)
        if "items" in data:
            all_qas.extend(data["items"])  # Thêm vào list chung
    return all_qas

def main():
    data_path = "data/description data/data_with_new_mota.csv"
    df = pd.read_csv(data_path, dtype=str)
    grouped = group_by_hs_prefix(df)
    result = asyncio.run(fetch_hscode_qa_from_csv(grouped, "gemini-2.0-flash-001"))
    print(result)
    # Lưu JSON
    output_path = "data/QA data/QAdataset.json"
    with open(output_path, "w", encoding="utf-8") as f_json:
        json.dump(result, f_json, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
