import pandas as pd
import google.generativeai as genai

# Đọc dữ liệu từ file CSV
file_input = './data/working_data.csv'  # Đường dẫn đến file CSV của bạn
file_output = './data/output_data.csv'

genai.configure(api_key="AIzaSyDPtwg1S_3nnH8zwOGjMrN-dagulLQH-0Q")

# Chọn model Gemini 2.0 Flash
model = genai.GenerativeModel('gemini-2.0-flash')

# Đọc dữ liệu từ CSV vào DataFrame
df = pd.read_csv(file_input)

# Hàm tóm tắt mô tả bằng API Gemini
def summarize_description(text_to_summarize):
    try:
        # Gọi API để tóm tắt mô tả
        response = model.generate_content(
            f'''Tôi có mô tả 1 mặt hàng được phân lớp từ chung tới riêng sau:{text_to_summarize}
                        Hãy tóm tắt lại mô tả đó trong vòng 1 câu, đảm bảo các yêu cầu sau: 
                        - Giữ cấu trúc câu tập trung và súc tích.
                        - Giữ lại đầy đủ các từ chính.
                        - Tránh sự lặp lại các từ nhiều lần không có đóng góp lớn cho câu.
                        - Đảm bảo logic cho câu.
                        - Lấy kết quả cuối cùng, không cần giải thích.
    ''',
            generation_config = {
                "temperature": 0,  # Điều chỉnh giá trị temperature theo ý muốn
            }  # Đóng ngoặc đúng chỗ
        )
        # Trả về kết quả tóm tắt
        return response.text
    except Exception as e:
        print(f"Error occurred: {e}")
        return text_to_summarize  # Trả về mô tả gốc nếu có lỗi

# Duyệt qua từng dòng và tóm tắt mô tả
# df['mo_ta_tom_tat'] = df['mo_ta'].apply(summarize_description)

# # Lưu kết quả vào file CSV mới với encoding utf-8
# df.to_csv(file_output, index=False, encoding='utf-8')

# print(f"Đã lưu kết quả tóm tắt vào {file_output}")

text="Ngựa, lừa, la sống/- Ngựa:/- - Loại thuần chủng để nhân giống"
print(summarize_description(text))
