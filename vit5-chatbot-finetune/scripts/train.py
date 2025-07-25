from pyexpat import model
import torch
from transformers import EarlyStoppingCallback,AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import evaluate
from torch.optim import AdamW
import json
import os
import wandb
import warnings
from dataset import load_and_process_datasets, ChatbotDataset

# Thiết lập môi trường
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Giảm TensorFlow logs

def get_datasets_for_trainer(batch_size=32, max_length=256, tokenizer_name="VietAI/vit5-base"):
    tensors = load_and_process_datasets(max_length=max_length, tokenizer_name=tokenizer_name)
    
    # Unpack 4 tensors từ process_split
    train_input_ids, train_attention_mask, train_labels = tensors["train"]
    val_input_ids, val_attention_mask, val_labels = tensors["validation"]
    
    # Tạo dataset với đủ 4 arguments
    train_dataset = ChatbotDataset(
        train_input_ids, 
        train_attention_mask, 
        train_labels, 
    )
    val_dataset = ChatbotDataset(
        val_input_ids, 
        val_attention_mask, 
        val_labels, 
    )
    
    return train_dataset, val_dataset



def train(
    model_name="VietAI/vit5-base",
    num_epochs=25,
    learning_rate=5e-5,
    batch_size=32,
    output_dir="models/vit5-base-bs64-ml128",
    early_stop_patience=3,
    save_total_limit=2,
    logging_dir="logs",
    report_to="none",
    max_length=128,
):

    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # for param in model.encoder.parameters():
    #     param.requires_grad = False

    train_dataset, val_dataset = get_datasets_for_trainer(batch_size=batch_size, max_length=max_length, tokenizer_name=model_name)

        # Tối ưu hóa TrainingArguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="steps",  # Đánh giá thường xuyên hơn
        eval_steps=100,         # Mỗi 100 steps
        save_strategy="steps",
        save_steps=500,         # Save mỗi 500 steps
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,      # Giảm overfitting
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=1,     # Giữ 1 checkpoints tốt nhất
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        lr_scheduler_type="linear",  # Linear decay
        warmup_steps=300,       # Warmup
        greater_is_better=False,
        report_to="wandb",
        run_name="vit5-chatbot-bs64-lr5e-5-maxlen128-no-freeze-base",
        dataloader_num_workers=12,
        # Tối ưu memory và tốc độ
        gradient_accumulation_steps=2,  # Nếu batch_size nhỏ
        fp16=True,              # Mixed precision training
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        # Label smoothing cho seq2seq
        # label_smoothing_factor=0.1,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Model and checkpoints saved to {output_dir}")

    with open("log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    return trainer


if __name__ == "__main__":
    train()