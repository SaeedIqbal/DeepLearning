import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import time

# ==================== POSITIONAL ENCODING ====================
class PositionalEncoding(nn.Module):
    """Implement the positional encoding (PE) function."""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer (not a trainable parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# ==================== MULTI-HEAD ATTENTION ====================
class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Query tensor of shape (batch_size, num_heads, seq_len, d_k)
            K: Key tensor
            V: Value tensor
            mask: Optional mask tensor
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).
        """
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        """
        Combine the head dimensions.
        """
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        """
        Forward pass for multi-head attention.
        
        Args:
            Q: Query tensor of shape (batch_size, seq_len, d_model)
            K: Key tensor
            V: Value tensor
            mask: Optional mask tensor
        """
        batch_size = Q.size(0)
        
        # Linear projections and split into heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply final linear projection
        output = self.W_o(self.combine_heads(attention_output))
        
        return output, attention_weights

# ==================== FEED-FORWARD NETWORK ====================
class PositionWiseFeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

# ==================== ENCODER LAYER ====================
class EncoderLayer(nn.Module):
    """Single encoder layer."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask for attention
        """
        # Self-attention sublayer with residual connection
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed-forward sublayer with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x

# ==================== DECODER LAYER ====================
class DecoderLayer(nn.Module):
    """Single decoder layer."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Decoder input tensor
            encoder_output: Output from encoder
            src_mask: Source mask
            tgt_mask: Target mask
        """
        # Self-attention sublayer
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Cross-attention sublayer
        attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)
        
        # Feed-forward sublayer
        ff_output = self.feed_forward(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)
        
        return x

# ==================== TRANSFORMER ENCODER ====================
class TransformerEncoder(nn.Module):
    """Transformer Encoder with multiple layers."""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            mask: Optional attention mask
        """
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)

# ==================== TRANSFORMER DECODER ====================
class TransformerDecoder(nn.Module):
    """Transformer Decoder with multiple layers."""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Target tensor of shape (batch_size, seq_len)
            encoder_output: Output from encoder
            src_mask: Source mask
            tgt_mask: Target mask
        """
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        x = self.norm(x)
        
        # Project to vocabulary size
        return self.output_projection(x)

# ==================== COMPLETE TRANSFORMER ====================
class Transformer(nn.Module):
    """Complete Transformer model for sequence-to-sequence tasks."""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, 
                 max_seq_len=100, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, num_heads, num_encoder_layers, 
            d_ff, max_seq_len, dropout
        )
        
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, num_heads, num_decoder_layers,
            d_ff, max_seq_len, dropout
        )
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_mask(self, src, tgt, pad_idx=0):
        """
        Create masks for source and target sequences.
        
        Args:
            src: Source tensor (batch_size, src_seq_len)
            tgt: Target tensor (batch_size, tgt_seq_len)
            pad_idx: Padding index
        
        Returns:
            src_mask: Source mask
            tgt_mask: Target mask
        """
        # Source mask (padding mask)
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        
        # Target mask (padding mask + look-ahead mask)
        tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        tgt_sub_mask = torch.tril(torch.ones(tgt_len, tgt_len)).type(torch.bool)
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        
        return src_mask, tgt_mask
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass of the Transformer.
        
        Args:
            src: Source sequence (batch_size, src_seq_len)
            tgt: Target sequence (batch_size, tgt_seq_len)
            src_mask: Source mask
            tgt_mask: Target mask
        
        Returns:
            Output logits
        """
        if src_mask is None or tgt_mask is None:
            src_mask, tgt_mask = self.create_mask(src, tgt)
        
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        return decoder_output
    
    def generate(self, src, max_len=50, temperature=1.0, pad_idx=0, start_token=1):
        """
        Generate sequence using greedy decoding.
        
        Args:
            src: Source sequence
            max_len: Maximum generation length
            temperature: Sampling temperature
            pad_idx: Padding index
            start_token: Start token index
        """
        self.eval()
        with torch.no_grad():
            src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
            encoder_output = self.encoder(src, src_mask)
            
            # Initialize with start token
            generated = torch.ones(src.size(0), 1).fill_(start_token).type_as(src)
            
            for _ in range(max_len - 1):
                tgt_mask = self.create_mask(src, generated)[1]
                output = self.decoder(generated, encoder_output, src_mask, tgt_mask)
                
                # Get the last token
                next_token_logits = output[:, -1, :] / temperature
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if all sequences generated padding
                if (next_token == pad_idx).all():
                    break
            
            return generated

# ==================== DATASET PREPARATION ====================
def prepare_wikitext2_data(batch_size=32, seq_len=64):
    """Prepare WikiText-2 dataset for language modeling."""
    print("Loading WikiText-2 dataset...")
    
    # Download and load dataset
    train_iter, val_iter, test_iter = WikiText2(split=('train', 'valid', 'test'))
    
    # Tokenizer
    tokenizer = get_tokenizer('basic_english')
    
    # Build vocabulary from training data
    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(text)
    
    vocab = build_vocab_from_iterator(
        yield_tokens(train_iter),
        specials=['<pad>', '<unk>', '<bos>', '<eos>']
    )
    vocab.set_default_index(vocab['<unk>'])
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Data processing function
    def data_process(raw_text_iter):
        data = []
        for text in raw_text_iter:
            tokens = tokenizer(text)
            tokens = ['<bos>'] + tokens + ['<eos>']
            data.append(torch.tensor([vocab[token] for token in tokens], dtype=torch.long))
        return torch.cat(data)
    
    # Process datasets
    train_data = data_process(train_iter)
    val_data = data_process(val_iter)
    test_data = data_process(test_iter)
    
    # Create batches
    def batchify(data, batch_size):
        """Divide data into batches."""
        nbatch = data.size(0) // batch_size
        data = data[:nbatch * batch_size]
        data = data.view(batch_size, -1).t().contiguous()
        return data
    
    train_data = batchify(train_data, batch_size)
    val_data = batchify(val_data, batch_size)
    test_data = batchify(test_data, batch_size)
    
    # Create sequence pairs for language modeling
    def get_batch(source, i, seq_len):
        """Get batch of source and target sequences."""
        seq_len = min(seq_len, len(source) - 1 - i)
        src = source[i:i+seq_len]
        tgt = source[i+1:i+1+seq_len]
        return src, tgt
    
    return vocab, train_data, val_data, test_data, get_batch

# ==================== TRAINING LOOP ====================
class TransformerTrainer:
    """Trainer class for the Transformer model."""
    
    def __init__(self, model, lr=0.0001, betas=(0.9, 0.98), eps=1e-9):
        self.model = model
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=betas, eps=eps
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)
        
    def train_epoch(self, train_data, get_batch, seq_len, clip=1.0):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for i in range(0, train_data.size(0) - 1, seq_len):
            # Get batch
            src, tgt = get_batch(train_data, i, seq_len)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(src, tgt[:, :-1])
            
            # Calculate loss
            loss = self.criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt[:, 1:].contiguous().view(-1)
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(self, val_data, get_batch, seq_len):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, val_data.size(0) - 1, seq_len):
                src, tgt = get_batch(val_data, i, seq_len)
                
                output = self.model(src, tgt[:, :-1])
                
                loss = self.criterion(
                    output.contiguous().view(-1, output.size(-1)),
                    tgt[:, 1:].contiguous().view(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_data, val_data, get_batch, epochs=5, seq_len=64):
        """Full training loop."""
        print("Starting training...")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_data, get_batch, seq_len)
            
            # Evaluate
            val_loss = self.evaluate(val_data, get_batch, seq_len)
            
            # Update learning rate
            self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.6f}")

# ==================== MAIN EXECUTION ====================
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    BATCH_SIZE = 32
    SEQ_LEN = 64
    D_MODEL = 128  # Reduced for faster training
    NUM_HEADS = 8
    NUM_ENCODER_LAYERS = 3  # Reduced for demo
    NUM_DECODER_LAYERS = 3  # Reduced for demo
    D_FF = 512
    MAX_SEQ_LEN = 100
    DROPOUT = 0.1
    EPOCHS = 3  # Reduced for demo
    
    # Prepare dataset
    vocab, train_data, val_data, test_data, get_batch = prepare_wikitext2_data(
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN
    )
    
    vocab_size = len(vocab)
    print(f"\nModel Configuration:")
    print(f"  Vocabulary Size: {vocab_size}")
    print(f"  d_model: {D_MODEL}")
    print(f"  Num Heads: {NUM_HEADS}")
    print(f"  Num Encoder Layers: {NUM_ENCODER_LAYERS}")
    print(f"  Num Decoder Layers: {NUM_DECODER_LAYERS}")
    
    # Create model
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT
    ).to(device)
    
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Move data to device
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    
    # Create trainer and train
    trainer = TransformerTrainer(model, lr=0.0005)
    trainer.train(train_data, val_data, get_batch, epochs=EPOCHS, seq_len=SEQ_LEN)
    
    # Test the model
    print("\nTesting the model...")
    test_loss = trainer.evaluate(test_data, get_batch, SEQ_LEN)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Generate some text
    print("\nGenerating sample text...")
    model.eval()
    
    # Create a sample input
    sample_text = "The history of artificial intelligence"
    tokenizer = get_tokenizer('basic_english')
    tokens = tokenizer(sample_text.lower())
    
    # Convert to indices
    indices = [vocab[token] for token in tokens]
    src = torch.tensor([indices], device=device)
    
    # Generate
    generated = model.generate(src, max_len=30, start_token=vocab['<bos>'])
    
    # Convert back to text
    generated_tokens = [vocab.get_itos()[idx] for idx in generated[0].cpu().numpy()]
    generated_text = ' '.join(generated_tokens)
    print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main()