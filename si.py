import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel

def tokenize_pair(comment, context, tokenizer, do_truncation=True, max_length=128):
    """
    Tokenizes a comment and its context as a pair, similar to the training procedure.
    """
    return tokenizer(
        comment,
        context,
        add_special_tokens=True,
        padding="max_length",
        truncation=do_truncation,
        max_length=max_length,
        return_tensors="pt"
    )

class SarcasmModel(nn.Module):
    def __init__(self):
        super(SarcasmModel, self).__init__()
        # Load pre-trained RoBERTa
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        # Add dropout and layer normalization for stability
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(768)
        # Multihead attention on the CLS token embedding
        self.multihead_attn = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        # Final classification head (binary classification: Not Sarcastic vs. Sarcastic)
        self.classifier = nn.Linear(768, 2)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [B, seq_len, 768]
        # Extract the [CLS] token representation (first token)
        cls_token = last_hidden_state[:, 0, :].unsqueeze(1)  # [B, 1, 768]
        # Apply multihead attention using CLS token as query, key, and value
        attn_output, _ = self.multihead_attn(cls_token, cls_token, cls_token)
        attn_output = self.dropout(attn_output)
        attn_output = self.layer_norm(attn_output)
        # Squeeze and pass through classifier
        logits = self.classifier(attn_output.squeeze(1))
        return logits

def load_sarcasm_model(model_path, device):
    """
    Loads the sarcasm detection model and the associated tokenizer.
    """
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = SarcasmModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer

def predict_sarcasm(comment, context, model, tokenizer, device):
    """
    Tokenizes the input pair (comment and context) and predicts sarcasm.
    Returns the predicted label and confidence.
    """
    encoding = tokenize_pair(comment, context, tokenizer, do_truncation=True, max_length=128)
    # Move tensors to the specified device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = F.softmax(logits, dim=1)
    
    predicted_idx = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0, predicted_idx].item()
    
    label_map = {0: "Not Sarcastic", 1: "Sarcastic"}
    return label_map[predicted_idx], confidence

def run_sarcasm_inference(input_csv, output_csv, model_path):
    """
    Reads paired data from a CSV, performs sarcasm detection, and saves the results.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_sarcasm_model(model_path, device)
    
    df = pd.read_csv(input_csv)
    required_cols = ["Comment", "Most Relevant Transcript Chunk"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError("Input CSV must contain 'Comment' and 'Most Relevant Transcript Chunk' columns.")
    
    results = []
    for _, row in df.iterrows():
        comment = row["Comment"]
        context = row["Most Relevant Transcript Chunk"]
        label, confidence = predict_sarcasm(comment, context, model, tokenizer, device)
        results.append({
            "Comment": comment,
            "Most Relevant Transcript Chunk": context,
            "Sarcasm Label": label,
            "Confidence": confidence
        })
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"Sarcasm inference results saved to {output_csv}")
    return output_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run sarcasm detection on paired comments and transcript chunks."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file with paired data")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file for sarcasm inference results")
    parser.add_argument("--model", type=str, required=True, help="Path to the sarcasm model weights file")
    args = parser.parse_args()
    run_sarcasm_inference(args.input, args.output, args.model)
