import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from preprocess import preprocess

def load_and_process_datasets(
    train_path="data/train_dataset_full.json",
    val_path="data/val_dataset_full.json",
    test_path="data/test_dataset_full.json",
    tokenizer_name="VietAI/vit5-base",
    max_length=256
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # Load các tập đã chia

    dataset = load_dataset(
        "json",
        data_files={
            "train": train_path,
            "val": val_path,
            "test": test_path
        }
    )

    train_dataset = dataset["train"]
    val_dataset = dataset["val"]
    test_dataset = dataset["test"]


    def process_split(split_dataset):
        processed = [preprocess(ex, tokenizer, max_length) for ex in split_dataset]
        input_ids = torch.stack([item["input_ids"] for item in processed])
        attention_mask = torch.stack([item["attention_mask"] for item in processed])
        labels = torch.stack([item["labels"] for item in processed])
        return input_ids, attention_mask, labels

    train_tensors = process_split(train_dataset)
    val_tensors = process_split(val_dataset)
    test_tensors = process_split(test_dataset)

    return {
        "train": train_tensors,
        "validation": val_tensors,
        "test": test_tensors
    }

class ChatbotDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
        }
    
