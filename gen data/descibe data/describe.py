#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HS Code Description Generator
Converts notebook code to Python script for generating HS code descriptions
"""

# Import required libraries
from google import genai
from pydantic import BaseModel
import json
import asyncio
from tqdm.asyncio import tqdm
from collections import defaultdict
import pandas as pd
from typing import List, Dict
from dotenv import load_dotenv
import os
import csv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_GENAI_API_KEY")

# Data path configuration

def load_data(data_path, nrows=None):
    df = pd.read_csv(data_path, dtype=str)
    if nrows is not None:
        data = df.head(nrows)
    else:
        data = df
    data = data[['mahs', 'mo_ta']]
    return data

def group_by_hs_prefix(df):
    """Group HS codes by their 4-digit prefix"""
    grouped = defaultdict(list)
    for _, row in df.iterrows():
        mahs_value = row['mahs']
        prefix = str(mahs_value)[:4]  # Convert to string and take first 4 characters
        grouped[prefix].append(row.to_dict())  # Convert row to dictionary
    return grouped

# Pydantic models for data validation
class SimpleHSCodeItem(BaseModel):
    mahs: str
    mo_ta: str

class SimpleHSCodeGroup(BaseModel):
    items: List[SimpleHSCodeItem]

def build_prompt_for_description(prefix: str, items: List[Dict], length_mode: str = "medium") -> str:
    """Sinh prompt mô tả mã HS, có kiểm soát độ dài và hướng dẫn nội dòng"""
    
    # Hướng dẫn độ dài mô tả
    length_note = {
        "short": "- Ưu tiên diễn đạt ngắn gọn, súc tích nhất có thể (1–2 câu), vẫn giữ đủ ý phân biệt.",
        "medium": "- Mô tả rõ ràng, tự nhiên, từ 2–3 câu, đủ để phân biệt giữa các mã.",
        "verbose": "- Diễn đạt chi tiết hơn khi cần thiết, đặc biệt khi các mã dễ gây nhầm lẫn."
    }.get(length_mode, "")

    # Dữ liệu đầu vào định dạng liệt kê
    item_list = "".join(f"- {item['mahs']}: {item['mo_ta']}\n" for item in items)

    prompt = f"""Bạn là chuyên gia phân loại mã HS code. Dưới đây là danh sách các mã HS thuộc nhóm {prefix}:

{item_list}

---

## Nhiệm vụ:

Tái cấu trúc lại phần mô tả (`mo_ta`) cho từng mã HS sao cho:

- Mô tả phải được viết **bằng tiếng Việt tự nhiên**, như cách một chuyên gia giải thích cho người dùng phổ thông.
- **Giữ đúng ngôn ngữ đầu vào là tiếng Việt**, **được phép giữ lại các thuật ngữ chuyên ngành có trong dữ liệu gốc**.
- **Tuyệt đối không được thêm bất kỳ thông tin nào không có trong `mo_ta` gốc**.
- **Không được suy diễn, bổ sung kiến thức từ bên ngoài** hoặc "phỏng đoán".
- Làm nổi bật đặc trưng riêng biệt của từng mã: công dụng, đối tượng, trọng lượng, mục đích sử dụng, trạng thái, thành phần, v.v.
- Nếu nhiều mã mô tả các sản phẩm gần giống nhau, **phải chỉ rõ điểm giống và khác biệt giữa chúng**.
- Mỗi mô tả **bắt đầu bằng: “Mã {{mahs}} thuộc nhóm {prefix}, mô tả…”**
{length_note}
- **Không sử dụng các thuật ngữ kỹ thuật như “level”, “tầng”, “phân cấp”**

---

## Phản hồi nội dòng (gợi ý chiến lược mô tả):

Trước khi mô tả lại, bạn có thể:
1. **Tìm điểm giống nhau giữa các mã trong nhóm**
2. **Sau đó nêu rõ điểm khác biệt của từng mã**

Việc này giúp mô tả rõ ràng và dễ hiểu hơn, đặc biệt với các mã có nội dung gần giống nhau.

---

## Trung lập ngành hàng – tránh thiên vị:

Ví dụ minh họa dưới đây chỉ mang tính chất hướng dẫn **cách trình bày mô tả phân biệt rõ ràng giữa các mã HS**.  
**Chúng không đại diện cho bất kỳ ngành hàng cụ thể nào** và không nên ảnh hưởng đến cách diễn đạt với các ngành như: thiết bị điện tử, dược phẩm, hóa chất, máy móc, sản phẩm cơ khí...

---

## Định dạng JSON đầu ra:

{{
  "{prefix}": [
    {{
      "mahs": "mã HS",
      "mo_ta": "mô tả mới"
    }},
    ...
  ]
}}

---

## Ví dụ minh họa (trích ngành hàng - chỉ để hiểu cách trình bày):

### Đầu vào:
- 01051110: Gia cầm sống gồm gà Gallus domesticus, vịt, ngan, ngỗng, gà tây... /- Khối lượng ≤ 185g:/- - Gà:/- - - Để nhân giống
- 01051190: .../- - - Loại khác
- 01051210: .../- - Gà tây:/- - - Để nhân giống
- 01051290: .../- - - Loại khác

### Đầu ra:
{{
  "0105": [
    {{
      "mahs": "01051110",
      "mo_ta": "Mã 01051110 thuộc nhóm 0105 (Gia cầm sống), mô tả gà giống thuộc loài Gallus domesticus có khối lượng không quá 185g, được dùng để nhân giống."
    }},
    {{
      "mahs": "01051190",
      "mo_ta": "Mã 01051190 thuộc nhóm 0105 (Gia cầm sống), mô tả gà Gallus domesticus không quá 185g, không dùng để nhân giống mà thuộc loại khác."
    }},
    {{
      "mahs": "01051210",
      "mo_ta": "Mã 01051210 mô tả gà tây có khối lượng không quá 185g, dùng để nhân giống."
    }},
    {{
      "mahs": "01051290",
      "mo_ta": "Mã 01051290 thuộc nhóm 0105 (Gia cầm sống), mô tả gà tây không quá 185g, thuộc loại không dùng để nhân giống."
    }}
  ]
}}

**Lưu ý:** Không sao chép cấu trúc hoặc nội dung ví dụ trên cho các ngành hàng khác. Chỉ dùng để tham khảo **văn phong và cách làm nổi bật điểm phân biệt**.
"""
    return prompt




async def fetch_and_save_hscode_csv(grouped_data, output_file="data/description data/new_mota.csv", model="gemini-2.0-flash-001"):
    """Fetch HS code descriptions from AI and save to CSV"""
    all_rows = []  # Contains all data rows from all prefixes
    client = genai.Client(api_key=api_key)

    for prefix, items in tqdm(grouped_data.items(), desc="Processing HS Prefixes"):
        prompt = build_prompt_for_description(prefix, items)

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": SimpleHSCodeGroup
            },
        )

        raw_text = response.text

        try:
            data = json.loads(raw_text)
            if "items" in data and isinstance(data["items"], list):
                all_rows.extend(data["items"])
            else:
                print(f"⚠️ Không tìm thấy 'items' hoặc không phải list tại prefix {prefix}")
        except json.JSONDecodeError:
            print(f"❌ JSON lỗi ở prefix {prefix}: {raw_text}")

    # Write data to CSV file
    with open(output_file, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mahs", "mo_ta"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"✅ Đã lưu kết quả vào file: {output_file}")
    return all_rows

async def main():
    """Main function to run the HS code description generation process"""
    print("🚀 Starting HS Code description generation...")
    
    # Load and prepare data
    data_path = "data/original data/hs_code.csv"
    nrows = 19  # Adjust as needed, or set to None to load all data
    print("📊 Loading data...")
    data = load_data(data_path)
    print(f"Loaded {len(data)} records")
    print(data.head())
    
    # Group data by prefix
    print("🔄 Grouping data by HS prefix...")
    grouped_data = group_by_hs_prefix(data)
    print(f"Found {len(grouped_data)} groups:")
    for prefix, items in grouped_data.items():
        print(f"  - {prefix}: {len(items)} items")
    
    # Generate descriptions and save to CSV
    print("🤖 Generating new descriptions...")
    output_path = "data/description data/description_data.csv"
    model = "gemini-2.0-flash-001"
    
    result = await fetch_and_save_hscode_csv(grouped_data, output_path, model)
    print(f"✅ Process completed. Generated {len(result)} descriptions.")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())

