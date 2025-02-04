# --- Start of the Model Definition ---
class InputProjection(nn.Module):
    def __init__(self, input_dim, d_k):
        super().__init__()
        self.input_dim = input_dim
        self.d_k = d_k  # dimension of key/query/value
        self.wq = nn.Linear(input_dim, d_k)
        self.wk = nn.Linear(input_dim, d_k)
        self.wv = nn.Linear(input_dim, d_k)

    def forward(self, x):
        q = self.wq(x)  # [batch_size, seq_len, d_k]
        k = self.wk(x)  
        v = self.wv(x)  
        return q, k, v

class AdditiveAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        

    def forward(self, x):  # input x is Q [batch_size, seq_len, d_k]
        # Calculate global query vector by taking the mean of all query vectors along the sequence length dimension
        global_query = torch.mean(x, dim=1)  
        return global_query  

class GlobalContextKey(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.wk = nn.Linear(d_k, 1)  # parameter vector w_k
        self.additive_attention = AdditiveAttention(d_k)

    def forward(self, key_matrix, global_query):
        # key_matrix: [batch_size, seq_len, d_k]
        # global_query: [batch_size, d_k]
        # Element-wise multiplication with global query (broadcasted)
        context_aware_key_matrix = key_matrix * global_query.unsqueeze(1)  # [batch_size, seq_len, d_k]
        # Reduce along sequence dimension via additive attention (i.e. averaging)
        global_key_vector = self.additive_attention(context_aware_key_matrix)  # [batch_size, d_k]
        return global_key_vector, context_aware_key_matrix

class GlobalContextValue(nn.Module):
    def __init__(self, d_k):
        super().__init__()

    def forward(self, value_matrix, global_key):
        # value_matrix: [batch_size, seq_len, d_k]
        # global_key: [batch_size, d_k]
        global_context_aware_value = value_matrix * global_key.unsqueeze(1)  # [batch_size, seq_len, d_k]
        return global_context_aware_value

class OutputTransformation(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.linear_transform = nn.Linear(d_k, d_k)

    def forward(self, global_context_aware_value, query_matrix):
        transformed_value = self.linear_transform(global_context_aware_value)
        # Add the original Q with the transformed value.
        output = transformed_value + query_matrix
        return output

class FastformerLayer(nn.Module):
    def __init__(self, input_dim, d_k):
        """
        input_dim: Dimension of the input (e.g., 512 for the first layer and d_k for subsequent layers)
        d_k: The dimension for the keys/queries/values.
        """
        super().__init__()
        self.input_projection = InputProjection(input_dim, d_k)
        self.additive_attention = AdditiveAttention(d_k)
        self.global_context_key = GlobalContextKey(d_k)
        self.global_context_value = GlobalContextValue(d_k)
        self.output_transformation = OutputTransformation(d_k)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        q, k, v = self.input_projection(x)  # q, k, v: [batch_size, seq_len, d_k]
        global_query = self.additive_attention(q)  # [batch_size, d_k]
        global_key_vector, _ = self.global_context_key(k, global_query)  # [batch_size, d_k]
        global_context_aware_value = self.global_context_value(v, global_key_vector)  # [batch_size, seq_len, d_k]
        output = self.output_transformation(global_context_aware_value, q)  # [batch_size, seq_len, d_k]
        return output

