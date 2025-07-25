from transformers import AutoTokenizer

def preprocess(example, tokenizer, max_length=128):
    """
    Tiền xử lý một mẫu dữ liệu chatbot: mã hóa prompt và response với attention_mask.
    - prompt: đầu vào cho encoder
    - response: đầu ra cho decoder
    """
    # Mã hóa prompt
    input_enc = tokenizer(
        example['prompt'],
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = input_enc.input_ids.squeeze()
    attention_mask = input_enc.attention_mask.squeeze()


    # Mã hóa response
    label_enc = tokenizer(
        example['response'],
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    labels = label_enc.input_ids.squeeze()
    # Padding labels đổi về -100 để không tính vào loss
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }