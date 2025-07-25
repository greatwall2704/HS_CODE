import json
import os
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def split_dataset():
    """
    Chia dataset thành train/validation/test với tỷ lệ 7:1.5:1.5
    """
    # Đường dẫn tới file dữ liệu gốc
    data_path = "HS_CODE/data/QA data/QAdataset.csv"
    output_dir = "vit5-chatbot-finetune/data"
    
    # Load dữ liệu gốc
    print("Đang tải dữ liệu...")
    raw_dataset = load_dataset("json", data_files=data_path)["train"]
    raw_data = [ex for ex in raw_dataset]
    
    # Đổi tên trường từ question/answer thành prompt/response
    print("Đang đổi tên các trường...")
    processed_data = []
    for example in raw_data:
        processed_example = {
            "prompt": example["question"],
            "response": example["answer"]
        }
        processed_data.append(processed_example)
    
    print(f"Tổng số mẫu dữ liệu: {len(processed_data)}")
    
    # Chia train/val/test (7/1.5/1.5)
    print("Đang chia dữ liệu...")
    train_data, temp_data = train_test_split(processed_data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    print(f"Train set: {len(train_data)} mẫu ({len(train_data)/len(processed_data)*100:.1f}%)")
    print(f"Validation set: {len(val_data)} mẫu ({len(val_data)/len(processed_data)*100:.1f}%)")
    print(f"Test set: {len(test_data)} mẫu ({len(test_data)/len(processed_data)*100:.1f}%)")
    
    # Lưu các tập dữ liệu
    train_path = os.path.join(output_dir, "train_dataset.json")
    val_path = os.path.join(output_dir, "val_dataset.json")
    test_path = os.path.join(output_dir, "test_dataset.json")
    
    print("Đang lưu dữ liệu...")
    
    # Lưu train set
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu train set tại: {train_path}")
    
    # Lưu validation set
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu validation set tại: {val_path}")
    
    # Lưu test set
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu test set tại: {test_path}")
    
    print("Hoàn thành chia dataset!")
    
    return {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }

if __name__ == "__main__":
    split_dataset()