import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer

#############################################
# Utility Functions & Model Definitions
#############################################

def get_device() -> str:
    """
    Determines the best available device for computation.
    
    Returns:
        str: The device to use ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():  # For Apple Silicon
        return 'mps'
    else:
        return 'cpu'
    

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class CoAttentionSarcasmModel(nn.Module):
    def __init__(self, num_classes=2, share_encoder=True):
        super().__init__()
        # 1) Shared or separate encoders
        if share_encoder:
            self.comment_encoder = self.context_encoder = RobertaModel.from_pretrained("distilroberta-base")
        else:
            self.comment_encoder = RobertaModel.from_pretrained("distilroberta-base")
            self.context_encoder = RobertaModel.from_pretrained("distilroberta-base")
        
        # 2) Co‑attention layers
        self.comment_to_context_attn = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.context_to_comment_attn = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        
        # 3) Pooling & fusion
        self.attn_pool = nn.Linear(768, 1)
        self.layer_norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(0.3)
        
        # 4) Classification head
        self.mish = Mish()
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            self.mish,
            nn.Dropout(0.3),
            nn.Linear(768, num_classes)
        )
    
    def forward(self, input_ids_comment, attention_mask_comment, input_ids_context, attention_mask_context):
        # Encode both sequences
        comment_out = self.comment_encoder(
            input_ids=input_ids_comment,
            attention_mask=attention_mask_comment
        ).last_hidden_state            # (B, Lc, 768)
        
        context_out = self.context_encoder(
            input_ids=input_ids_context,
            attention_mask=attention_mask_context
        ).last_hidden_state            # (B, Lx, 768)
        
        # Prepare masks for MultiheadAttention (True == ignore)
        ctx_key_padding = (attention_mask_context == 0)
        cmt_key_padding = (attention_mask_comment  == 0)
        
        # 1) Comment → Context
        c2x, _ = self.comment_to_context_attn(
            query=comment_out,
            key=context_out,
            value=context_out,
            key_padding_mask=ctx_key_padding
        )
        c2x = self.dropout(c2x)
        c2x = self.layer_norm(comment_out + c2x)  # skip connection
        
        # 2) Context → Comment
        x2c, _ = self.context_to_comment_attn(
            query=context_out,
            key=comment_out,
            value=comment_out,
            key_padding_mask=cmt_key_padding
        )
        x2c = self.dropout(x2c)
        x2c = self.layer_norm(context_out + x2c)  # skip connection
        
        # 3) Attention pooling on both updated sequences
        # Comment pooled
        scores_c = self.attn_pool(c2x).squeeze(-1)                    # (B, Lc)
        scores_c = scores_c.masked_fill(attention_mask_comment==0, -1e9)
        weights_c = F.softmax(scores_c, dim=1).unsqueeze(-1)          # (B, Lc,1)
        pooled_c = torch.sum(c2x * weights_c, dim=1)                  # (B, 768)
        
        # Context pooled
        scores_x = self.attn_pool(x2c).squeeze(-1)                    # (B, Lx)
        scores_x = scores_x.masked_fill(attention_mask_context==0, -1e9)
        weights_x = F.softmax(scores_x, dim=1).unsqueeze(-1)          # (B, Lx,1)
        pooled_x = torch.sum(x2c * weights_x, dim=1)                  # (B, 768)
        
        # 4) Fuse and classify
        fused = self.layer_norm(pooled_c + pooled_x)                  # (B, 768)
        logits = self.classifier(fused)                               # (B, num_classes)
        return logits

def load_model(model_path, device):
    model = CoAttentionSarcasmModel(num_classes=2, share_encoder=True)
    state_dict = torch.load(model_path, map_location='cpu')  # force load on CPU first
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    return model, tokenizer


'''
# Define the Mish activation function
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class SarcasmDetectionModel(nn.Module):
    def __init__(self, num_classes=2, share_encoder=True):
        super(SarcasmDetectionModel, self).__init__()
        
        # Initialize encoders: option to share weights or use separate ones
        if share_encoder:
            self.encoder = RobertaModel.from_pretrained("roberta-base")
            self.comment_encoder = self.encoder
            self.context_encoder = self.encoder
        else:
            self.comment_encoder = RobertaModel.from_pretrained("roberta-base")
            self.context_encoder = RobertaModel.from_pretrained("roberta-base")
        
        # Cross-attention: comment tokens attend to context tokens
        self.cross_attn = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        
        # Attention pooling over the updated comment tokens
        self.attention_pooling = nn.Linear(768, 1)
        
        # Layer normalization and dropout used in multiple places
        self.layer_norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(0.3)
        
        # Mish activation
        self.mish = Mish()
        
        # Classification head with additional non-linearity and dropout
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            self.mish,
            nn.Dropout(0.2),
            nn.Linear(768, num_classes)
        )
    
    def forward(self, input_ids_comment, attention_mask_comment, input_ids_context, attention_mask_context):
        # Encode comment and context separately
        comment_outputs = self.comment_encoder(input_ids=input_ids_comment, attention_mask=attention_mask_comment)
        comment_hidden = comment_outputs.last_hidden_state  # shape: (batch, seq_len, 768)
        
        context_outputs = self.context_encoder(input_ids=input_ids_context, attention_mask=attention_mask_context)
        context_hidden = context_outputs.last_hidden_state  # shape: (batch, seq_len, 768)
        
        # Cross-attention: comment tokens (query) attend to context tokens (key, value)
        key_padding_mask = (attention_mask_context == 0)  # True for positions to ignore
        cross_attn_output, _ = self.cross_attn(
            query=comment_hidden,
            key=context_hidden,
            value=context_hidden,
            key_padding_mask=key_padding_mask
        )
        
        # Apply dropout and layer normalization to the cross-attended output
        cross_attn_output = self.dropout(cross_attn_output)
        cross_attn_output = self.layer_norm(cross_attn_output)
        
        # Fuse the original comment representation with the cross-attended output (skip connection)
        updated_comment = comment_hidden + cross_attn_output
        
        # Attention pooling: compute attention scores for each token in the comment
        attn_scores = self.attention_pooling(updated_comment).squeeze(-1)  # shape: (batch, seq_len)
        attn_scores = attn_scores.masked_fill(attention_mask_comment == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)  # shape: (batch, seq_len)
        
        # Weighted average of the token embeddings
        pooled_output = torch.sum(updated_comment * attn_weights.unsqueeze(-1), dim=1)  # shape: (batch, 768)
        
        # Final normalization before classification
        pooled_output = self.layer_norm(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def load_model(model_path, device):
    model = SarcasmDetectionModel(num_classes=2, share_encoder=True)
    state_dict = torch.load(model_path, map_location='cpu')  # force load on CPU first
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    return model, tokenizer
'''


'''def load_model(model_path, device):
    """
    Load the trained sarcasm detection model along with the tokenizer.
    """
    model = SarcasmDetectionModel(num_classes=2, share_encoder=True)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    return model, tokenizer'''

def predict_instance(comment, context, model, tokenizer, device, max_length=64):
    """
    Tokenize a single comment and its context, then predict sarcasm.
    
    Returns:
        label (str): "Sarcastic" or "Non-sarcastic"
        sarcastic_prob (float): Probability for the sarcastic class.
    """
    # Tokenize the comment
    comment_enc = tokenizer(
        comment,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors="pt"
    )
    # Tokenize the context
    context_enc = tokenizer(
        context,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors="pt"
    )
    # Move tensors to the device
    comment_input_ids = comment_enc["input_ids"].to(device)
    comment_attention_mask = comment_enc["attention_mask"].to(device)
    context_input_ids = context_enc["input_ids"].to(device)
    context_attention_mask = context_enc["attention_mask"].to(device)
    
    with torch.no_grad():
        logits = model(comment_input_ids, comment_attention_mask, context_input_ids, context_attention_mask)
        probs = F.softmax(logits, dim=1)
        # Assuming index 1 corresponds to "sarcastic"
        sarcastic_prob = probs[0, 1].item()
    
    label = "Sarcastic" if sarcastic_prob >= 0.5 else "Non-sarcastic"
    return label, sarcastic_prob

#############################################
# Inference Function
#############################################

def run_sarcasm_inference(input_csv, output_csv, model_path):
    """
    Runs sarcasm inference on input CSV data and writes predictions to output CSV.
    
    Expects the input CSV to have at least:
        - 'comment'
        - 'context'
    """
    device = get_device()
    model, tokenizer = load_model(model_path, device)
    
    df = pd.read_csv(input_csv)
    if "Comment" not in df.columns or "Most Relevant Transcript Chunk" not in df.columns:
        raise ValueError("Input CSV must contain 'comment' and 'context' columns")
    
    sarcasm_labels = []
    sarcasm_probs = []
    
    # Loop over each row and predict sarcasm
    for idx, row in df.iterrows():
        comment = row["Comment"]
        context = row["Most Relevant Transcript Chunk"]
        label, prob = predict_instance(comment, context, model, tokenizer, device)
        sarcasm_labels.append(label)
        sarcasm_probs.append(prob)
    
    # Add predictions to the DataFrame
    df["sarcasm_label"] = sarcasm_labels
    df["sarcasm_prob"] = sarcasm_probs
    df.to_csv(output_csv, index=False)
    print(f"Inference complete. Predictions saved to {output_csv}")

#############################################
# Main Block
#############################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sarcasm inference using the SarcasmDetectionModel")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file with 'comment' and 'context' columns")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file to save predictions")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained sarcasm model weights (.pth file)")
    args = parser.parse_args()
    run_sarcasm_inference(args.input, args.output, args.model)
