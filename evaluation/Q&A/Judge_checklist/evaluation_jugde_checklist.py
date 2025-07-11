
from google import genai
from pydantic import BaseModel
import json
import asyncio
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from typing import List, Dict
from dotenv import load_dotenv
import os
import csv
from pydantic import RootModel

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_GENAI_API_KEY")
if not api_key:
    raise ValueError("API key không được tìm thấy. Vui lòng đặt GOOGLE_GENAI_API_KEY trong .env.")

# Load data from CSV
def load_data(data_path, nrows=None):
    df = pd.read_csv(data_path, dtype=str)
    if nrows is not None:
        data = df.head(nrows)
    else:
        data = df
    data = data[['mahs', 'mo_ta', 'mo_ta_moi']]
    return data

# Group by HS prefix (4 first digits)
def group_by_hs_prefix(df):
    grouped = defaultdict(list)
    for _, row in df.iterrows():
        prefix = str(row['mahs'])[:4]
        grouped[prefix].append(row.to_dict())
    return grouped

# Pydantic models for validating LLM response
from pydantic import BaseModel

class Evaluation(BaseModel):
    mahs: str  # Mã HS
    giu_dung_noi_dung: bool  # Mô tả mới giữ đúng nội dung và ý chính của mô tả gốc
    dung_chuyen_nganh: bool  # Mô tả đúng lĩnh vực/hàng hóa gốc, không sai chuyên ngành
    phan_biet_ma_hs: bool  # Có khả năng phân biệt với các mã HS khác trong cùng nhóm
    ngon_ngu_tu_nhien: bool  # Diễn đạt tự nhiên, không hành chính/máy móc
    dinh_dang_ro_rang: bool  # Trình bày rõ ràng, đúng cấu trúc
    khong_loi_logic: bool  # Không có lỗi logic, ngữ pháp hoặc mâu thuẫn
    giai_thich_loi: str = None  # Giải thích lý do nếu bất kỳ tiêu chí nào là false
    goi_y_sua: str = None  # Gợi ý sửa nếu mô tả chưa đạt yêu cầu

class EvaluationGroup(BaseModel):
    items: List[Evaluation]


# Prompt builder

from typing import List, Dict

def build_prompt_for_evaluation(prefix: str, items: List[Dict]) -> str:
    item_list = "".join(
        f"- {item['mahs']}:\n"
        f"  + Mô tả gốc: {item['mo_ta']}\n"
        f"  + Phiên bản viết lại: {item['mo_ta_moi']}\n"
        for item in items
    )

    prompt = f"""Bạn là chuyên gia hàng đầu trong việc đánh giá mô tả hàng hóa theo mã HS. Dưới đây là các cặp mô tả gốc và phiên bản viết lại thuộc nhóm `{prefix}`:

{item_list}

---

## Nhiệm vụ:
Đánh giá mỗi mô tả viết lại qua 2 tầng:
- So với mô tả gốc: Giữ đúng ý, không thêm thông tin không có trong mô tả gốc.
- So với các mô tả mới: Rõ ràng, khác biệt, không trùng lặp.

---

## Checklist (`true` nếu chắc chắn, `false` nếu không):
1. `giu_dung_noi_dung`: Giữ đúng nội dung gốc.
2. `dung_chuyen_nganh`: Đúng lĩnh vực, loại hàng.
3. `phan_biet_ma_hs`: Có khả năng phân biệt tốt với mã khác trong nhóm.
4. `ngon_ngu_tu_nhien`: Diễn đạt dễ hiểu với người dùng.
5. `dinh_dang_ro_rang`: Cấu trúc trình bày rõ ràng.
6. `khong_loi_logic`: Không mâu thuẫn, sai ngữ pháp.

---

## 🛠 Nếu bất kỳ tiêu chí nào bị đánh giá là `false`:
- Hãy thêm 2 trường sau:
  - `giai_thich_loi`: Giải thích lý do vì sao mô tả chưa đạt yêu cầu.
  - `goi_y_sua`: Đề xuất cách sửa lại mô tả để đáp ứng yêu cầu.

---
## Ví dụ:

### Tốt:
- `mahs`: 01041010  
  + Gốc: Cừu, dê sống/- Cừu:/- - Loại thuần chủng để nhân giống  
  + Viết lại: Cừu sống loại thuần chủng dùng để nhân giống  

```json
{{
  "mahs": "01041010",
  "giu_dung_noi_dung": true,
  "dung_chuyen_nganh": true,
  "phan_biet_ma_hs": true,
  "ngon_ngu_tu_nhien": true,
  "dinh_dang_ro_rang": true,
  "khong_loi_logic": true,
  "giai_thich_loi": null,
  "goi_y_sua": null
}}
```

### Chưa tốt:
- `mahs`: 01042090  
  + Gốc: Cừu, dê sống/- Dê:/- - Loại khác  
  + Viết lại: Dê giống cao sản nhập khẩu từ Châu Âu  

```json
{{
  "mahs": "01042090",
  "giu_dung_noi_dung": false,
  "dung_chuyen_nganh": false,
  "phan_biet_ma_hs": true,
  "ngon_ngu_tu_nhien": true,
  "dinh_dang_ro_rang": true,
  "khong_loi_logic": false,
  "giai_thich_loi": "Mô tả mới đã thêm thông tin 'cao sản nhập khẩu từ Châu Âu' mà không có trong mô tả gốc.",
  "goi_y_sua": "Chỉ nêu rõ 'Dê loại khác', không đề cập xuất xứ hoặc chất lượng không có trong bản gốc."
}}
```

---

## Định dạng JSON:

```json
{{
  "{prefix}": [
    {{
      "mahs": "01041010",
      "giu_dung_noi_dung": true,
      "dung_chuyen_nganh": true,
      "phan_biet_ma_hs": true,
      "ngon_ngu_tu_nhien": true,
      "dinh_dang_ro_rang": true,
      "khong_loi_logic": true,
      "giai_thich_loi": null,
      "goi_y_sua": null
    }},
    ...
  ]
}}
```
*** Chú ý: Các ví dụ trên chỉ mang tính chất minh họa, mang tính tham khảo văn phong và cách trình bày. Bạn cần đánh giá dựa trên mô tả thực tế trong dữ liệu bạn đang xử lý.

## Lưu ý quan trọng (Guidelines for Objective Evaluation):

- Chỉ chọn `true` nếu bạn hoàn toàn chắc chắn tiêu chí được đáp ứng.
- Nếu có nghi ngờ, mô tả không rõ hoặc không đủ cơ sở → chọn `false`.
- Với bất kỳ tiêu chí nào bị false, luôn bổ sung giai_thich_loi và goi_y_sua.
- Không đánh giá theo cảm tính hoặc dựa vào ví dụ trước. Mỗi mô tả phải được đánh giá độc lập.
- Không thêm nhận xét, mở rộng hoặc viết lại mô tả. Chỉ trả về JSON đúng định dạng.
- Duy trì tư duy khách quan, trung lập, với vai trò của một chuyên gia hiểu ngôn ngữ và ngữ nghĩa trong phân loại HS.
"""
    return prompt




async def fetch_and_save_hscode_evaluation(grouped_data: dict, output_file, model):
    """Fetch evaluation results from LLM and save to CSV"""
    all_rows = []  # All evaluations across groups
    client = genai.Client(api_key=api_key)

    # Thêm tqdm để theo dõi tiến trình
    for prefix, items in tqdm(grouped_data.items(), desc="Đánh giá nhóm", unit="nhóm"):
        prompt = build_prompt_for_evaluation(prefix, items)

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": EvaluationGroup
            },
        )

        raw_text = response.text

        try:
            parsed = EvaluationGroup.parse_raw(raw_text)
            all_rows.extend([item.dict() for item in parsed.items])
        except Exception as e:
            print(f"❌ Lỗi tại prefix {prefix}: {e}\nResponse: {raw_text}")

    # Write to CSV
    with open(output_file, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(Evaluation.__fields__.keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"✅ Đã lưu kết quả đánh giá vào file: {output_file}")
    return all_rows

# Main async logic
async def main():
    print("🚀 Bắt đầu đánh giá mô tả mã HS...")
    data_path = "data/description data/mota_motamoi.csv"
    output_path = "evaluation/Judge_checklist/results/description/evaluation_results.csv"
    model = "gemini-2.5-pro"

    print("📊 Đang tải dữ liệu từ:", data_path)
    data = load_data(data_path)
    print(f"✅ Đã tải {len(data)} dòng dữ liệu.")

    grouped_data = group_by_hs_prefix(data)
    print(f"🔍 Đã chia thành {len(grouped_data)} nhóm theo prefix mã HS.")

    print(f"🤖 Đang đánh giá mô tả bằng mô hình: {model}")
    await fetch_and_save_hscode_evaluation(grouped_data, output_path, model)
    print(f"🏁 Đánh giá hoàn tất. Kết quả lưu tại: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
