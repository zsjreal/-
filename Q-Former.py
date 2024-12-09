import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear transformations and reshape
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Reshape and apply final linear transformation
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)

        return output, attention_weights


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class QFormerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, visual_features, mask=None):
        # Self attention
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Cross attention with visual features
        cross_attn_output, _ = self.cross_attention(x, visual_features, visual_features)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class QFormer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # Visual feature processing
        self.visual_proj = nn.Linear(2048, d_model)  # Assuming input visual features are 2048-dim

        # Query embedding
        self.query_embed = nn.Parameter(torch.randn(1, 32, d_model))  # 32 learnable query embeddings

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Stack of Q-Former layers
        self.layers = nn.ModuleList([
            QFormerLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_features, mask=None):
        batch_size = visual_features.size(0)

        # Project visual features to model dimension
        visual_features = self.visual_proj(visual_features)

        # Expand query embeddings for batch size
        queries = self.query_embed.expand(batch_size, -1, -1)

        # Add positional encoding
        queries = self.pos_encoder(queries)

        # Apply Q-Former layers
        x = queries
        for layer in self.layers:
            x = layer(x, visual_features, mask)

        return x


# Example usage
def main():
    # Create model
    model = QFormer()

    # Create sample input (batch_size=2, seq_len=50, feature_dim=2048)
    visual_features = torch.randn(2, 50, 2048)

    # Forward pass
    output = model(visual_features)

    print(f"Output shape: {output.shape}")  # Should be [2, 32, 512]


if __name__ == "__main__":
    main()