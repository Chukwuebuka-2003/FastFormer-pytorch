class FastformerForSentimentAnalysis(nn.Module):
    def __init__(self, vocab_size, d_model, d_k, num_classes, dropout=0.1):
        """
        d_model: Dimension for the embeddings (and first layer input).
        d_k: The dimension used within Fastformer layers.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fastformer = nn.Sequential(
            FastformerLayer(d_model, d_k),  # first layer: input dim is d_model (e.g., 512) -> outputs d_k (e.g., 64)
            FastformerLayer(d_k, d_k)       # subsequent layer: input and output dims are d_k
            # You can add more layers here with (d_k, d_k) if needed.
        )
        self.dropout = nn.Dropout(dropout)
        self.pooling = nn.AdaptiveAvgPool1d(1)  # pool along the sequence dimension
        self.linear = nn.Linear(d_k, num_classes)  # classifier now expects d_k features

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, d_model]
        encoded = self.fastformer(embedded)  # first layer converts to [batch_size, seq_len, d_k], and so on.
        # Pool over the sequence dimension.
        pooled = self.pooling(encoded.transpose(1, 2)).squeeze(-1)  # [batch_size, d_k]
        pooled = self.dropout(pooled)
        logits = self.linear(pooled)  # [batch_size, num_classes]
        return logits
