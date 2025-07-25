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
    mahs: List[str] # danh sách mã HS liên qua tới QA
    prompt: str
    response: str

class QAList(BaseModel):
    items: list[QAItem]

def build_prompt_for_group(prefix: str, items: List[Dict], length_mode: str = "medium") -> str:
    items_list = "\n".join(
        f"- {item['mahs']}: {item['mo_ta']}" for item in items
    )

    num_questions = 3 * len(items)

    length_note = {
        "short": "- Câu hỏi tối đa 20 từ, câu trả lời tối đa 40 từ.\n",
        "medium": "- Câu hỏi tối đa 30 từ, câu trả lời tối đa 60 từ.\n",
        "verbose": "- Không giới hạn độ dài, nhưng rõ ràng, có ví dụ.\n"
    }.get(length_mode, "- Câu hỏi tối đa 30 từ, câu trả lời tối đa 60 từ.\n")

    prompt = f"""Bây giờ bạn là một chuyên gia xây dựng bộ dữ liệu chatbot trong lĩnh vực hs code, hải quan.
    Dưới đây là danh sách các mã HS thuộc nhóm {prefix}, mỗi mã đi kèm mô tả chi tiết:

{items_list}

---

**Bối cảnh hội thoại:**
- Người hỏi: doanh nhân nhỏ chuẩn bị khai báo nhập khẩu, không rõ về mã HS
- Người trả lời: chuyên viên tư vấn hải quan có kinh nghiệm

---

**Nhiệm vụ:**
Tạo **{num_questions} cặp câu hỏi – trả lời**, mỗi mã HS phải có đúng **3 cặp câu hỏi liên quan tới các mã khác trong nhóm bao gồm nhiều kiểu câu hỏi đa dạng khác nhau.**.
- Câu hỏi phải xuất phát từ thực tế kinh doanh"
- Câu trả lời phải rõ ràng, đúng mã, có thể so sánh nếu phù hợp

---

**Yêu cầu nội dung:**
1. Ngôn ngữ đời thường, dễ hiểu với người không chuyên về hải quan.
2. Ưu tiên các tình huống thực tế: nhập khẩu, chọn sai mã, so sánh, phân biệt, kiểm tra nguồn gốc, thủ tục...
3. Tránh lặp nội dung. Câu hỏi phải có **đa dạng góc nhìn**: mục đích sử dụng, giống/không giống, phân biệt,so sánh, gợi ý mã...
4. Nếu có mã dễ gây nhầm lẫn, **bắt buộc sinh câu hỏi so sánh** chúng.
5. Tránh bias: không ưu tiên mã nào đặc biệt, không bỏ sót mã nào.

---

**Chiến lược đa dạng câu hỏi cho mỗi mã tương ứng với 3 cặp câu hỏi:**
- ✔ câu hỏi chỉ liên quan **mã HS đang xét** (dựa trên mô tả cụ thể)
- ✔ Một số câu hỏi so sánh **mã HS đang xét với 1 hoặc nhiều mã khác** trong nhóm
- ✔ Một số câu hỏi về **tình huống lựa chọn mã đúng/sai**

---

**Đa dạng & logic:**
- Câu hỏi phải **đa dạng góc nhìn**, không lặp lại giữa các mã
- Có thể hỏi về: mục đích hàng, nguồn gốc, phân biệt mã, giống/không giống
- Nếu câu hỏi liên quan nhiều mã HS (so sánh), hãy liệt kê tất cả mã trong `"mahs"`

---

**Ràng buộc nghiêm ngặt:**
- Không được tạo mã HS mới ngoài danh sách
- Không tự suy đoán mô tả sản phẩm
- Mỗi mã HS trong danh sách phải có đủ 3 cặp hỏi–đáp
- Mỗi mục phải gán đúng 1 hoặc nhiều mã trong trường `"mahs"`
- Đảm bảo tính đa dạng về cách hỏi và trả lời
- Không tạo câu hỏi và câu trả lời mà không liên quan đến mô tả của mã HS trong danh sách

{length_note}

---

📤 **Đầu ra bắt buộc:**
Trả về **một đoạn JSON hợp lệ duy nhất**, không có văn bản ngoài JSON.

**Định dạng JSON như sau:**

```json
{{
  "items": [
    {{
      "mahs": ["01013010"],
      "prompt": "Tôi đang nhập lừa giống, dùng mã nào?",
      "response": "Bạn nên dùng mã 01013010 vì áp dụng cho lừa giống thuần chủng."
    }},
    {{
      "mahs": ["01013010", "01013090"],
      "prompt": "Sự khác biệt giữa mã 01013010 và 01013090 là gì?",
      "response": "01013010 áp dụng cho lừa giống để nhân giống, còn 01013090 là lừa không dùng để nhân giống."
    }}
  ]
}}
*** Trên đây chỉ là ví dụ minh hoạ, không được nhìn vào để sao chép nội dung, chỉ mang tích chất tham khảo văn phong cũng như cấu trúc trình bày JSON.
---

**Lưu ý:**
- Trường "mahs" là danh sách chứa các mã HS liên quan đến nội dung được sử dụng cho mỗi câu hỏi–trả lời.
- Không tạo mã HS ngoài danh sách.
- Không được thiếu hoặc thừa Q&A cho mỗi mã HS
- Không dùng markdown (` ```json `), không có chú thích hoặc phần giải thích
- JSON phải được parse thành công với `json.loads(...)`
"""
    return prompt





def split_large_groups(items, max_items=8):
    """Chia nhóm lớn thành các nhóm nhỏ hơn để tránh vượt quá giới hạn token"""
    if len(items) <= max_items:
        return [items]
    
    chunks = []
    for i in range(0, len(items), max_items):
        chunks.append(items[i:i + max_items])
    return chunks

async def fetch_hscode_qa_from_csv(grouped_data: dict, model="gemini-2.0-flash-001"):
    all_qas = []  # List gom tất cả câu hỏi-đáp từ các prefix
    client = genai.Client(api_key=api_key)
    prefixes = list(grouped_data.keys())
    for prefix in tqdm(prefixes, desc="Đang sinh Q&A cho từng nhóm HS", ascii=True):
        items = grouped_data[prefix]
        
        # Chia nhóm lớn thành các nhóm nhỏ hơn
        item_chunks = split_large_groups(items, max_items=8)
        
        for chunk_idx, chunk in enumerate(item_chunks):
            content_data = build_prompt_for_group(f"{prefix}_{chunk_idx}", chunk)
            response = client.models.generate_content(
                model=model,
                contents=content_data,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": QAList,
                    "max_output_tokens": 8192
                },
            )
            # Parse response.text JSON ra dict rồi lấy phần "items"
            try:
                data = json.loads(response.text)
                if "items" in data:
                    all_qas.extend(data["items"])  # Thêm vào list chung
            except json.JSONDecodeError as e:
                print(f"JSON parsing error for prefix {prefix}_chunk_{chunk_idx}: {e}")
                print(f"Response text length: {len(response.text)}")
                print(f"First 500 chars: {response.text[:500]}")
                print(f"Last 500 chars: {response.text[-500:]}")
                print("---")
                # Có thể thử parse lại hoặc skip prefix này
                continue
    return all_qas

def main():
    data_path = "data/description data/description_data.csv"
    df = pd.read_csv(data_path, dtype=str)
    df = df[2000:] 
    grouped = group_by_hs_prefix(df)
    result = asyncio.run(fetch_hscode_qa_from_csv(grouped, "gemini-2.0-flash-001"))
    # Lưu JSON
    output_path = "data/QA data/QAdataset_8000.csv"
    df_result = pd.DataFrame(result)
    df_result.to_csv(output_path, index=False, encoding="utf-8")

if __name__ == "__main__":
    main()
