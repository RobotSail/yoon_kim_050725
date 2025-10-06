from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from fla.models.gru.modeling_gru import GRUForCausalLM
from fla.models.gru.configuration_gru import GRUConfig

from data_gen import DataConfig, get_dataloaders
from hack_utils import non_shifting_loss, compute_accuracy

import random
import numpy as np

MANUAL_SEED = 37
def set_seed(seed: int):
    """set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# GRU with Self-Attention and Weight Projection
class GRUWithSelfAttention(nn.Module):
    """
    GRU augmented with self-attention and weight projection for better long-range memory.
    Architecture: GRU(1024) → Self-Attention(1024) → Project(1024→256) → Output(256→vocab)
    """
    def __init__(self, config, num_attention_heads=8, projection_dim=256):
        super().__init__()
        self.gru_model = GRUForCausalLM(config)
        self.hidden_size = config.hidden_size  # Internal size (1024)
        self.projection_dim = projection_dim  # Output size (256)
        self.num_heads = num_attention_heads
        
        # Multi-head self-attention on full hidden size
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=num_attention_heads,
            dropout=0.0,
            batch_first=True
        )
        
        # Layer norm after attention
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        
        # Weight projection: hidden_size (1024) → projection_dim (256)
        self.projection = nn.Linear(config.hidden_size, projection_dim)
        self.proj_norm = nn.LayerNorm(projection_dim, eps=config.norm_eps)
        
        # New output head for projected dimension
        self.lm_head = nn.Linear(projection_dim, config.vocab_size, bias=False)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Get base GRU output (hidden states)
        base_output = self.gru_model.model(input_ids, attention_mask=attention_mask, **kwargs)
        gru_hidden = base_output.last_hidden_state  # (batch, seq_len, 1024)
        
        # Apply self-attention on full hidden size
        # MultiheadAttention expects (batch, seq, features) with batch_first=True
        attn_output, _ = self.attention(gru_hidden, gru_hidden, gru_hidden)
        
        # Residual connection + layer norm (still 1024-dim)
        hidden_states = self.attn_norm(gru_hidden + attn_output)
        
        # Project down: 1024 → 256
        projected = self.projection(hidden_states)
        projected = self.proj_norm(projected)
        
        # Pass through language model head: 256 → vocab_size
        logits = self.lm_head(projected)
        
        # Return in same format as original
        class Output:
            def __init__(self, logits):
                self.logits = logits
        
        return Output(logits)
    
    def num_parameters(self):
        """Get number of parameters"""
        return sum(p.numel() for p in self.parameters())


def train(
    model: nn.Module,  # Accept any model (GRU or GRU+Attention)
    train_dl: DataLoader,
    test_dl: DataLoader,  # Added for per-epoch evaluation
    optimizer: optim.Optimizer,
    max_epochs: int,
    scheduler=None,
):
    model.train()
    device = next(model.parameters()).device  # Get device from model parameters
    grad_norm_ema = 0.0  # Track gradient norm EMA
    beta = 0.98  # EMA decay factor
    
    for epoch in range(max_epochs):
        model.train()
        for inputs, targets in tqdm(train_dl, desc=f"Train Epoch {epoch}/{max_epochs}"):
            inputs, targets = inputs.to(device).long(), targets.to(device).long()
            optimizer.zero_grad()
            output = model(inputs, attention_mask=torch.ones_like(inputs))
            logits = output.logits
            loss = non_shifting_loss(logits, targets)
            
            # L1 regularization for weight sparsity (commented out for now)
            # l1_penalty = 0.0000001
            # l1_penalty = l1_penalty * sum(p.abs().sum() for p in model.parameters())
            # loss = loss + l1_penalty
            
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_norm_ema = beta * grad_norm_ema + (1 - beta) * grad_norm
            optimizer.step()
        
        # Evaluate accuracy after each epoch
        accuracy = eval(model, test_dl)
        
        # Step scheduler after each epoch
        if scheduler is not None:
            scheduler.step()
            print(f"Epoch {epoch}, Loss: {loss.item():.5f}, LR: {scheduler.get_last_lr()[0]:.6f}, Grad Norm EMA: {grad_norm_ema:.5f}, Accuracy: {accuracy:.5f}")
        else:
            print(f"Epoch {epoch}, Loss: {loss.item():.5f}, Grad Norm EMA: {grad_norm_ema:.5f}, Accuracy: {accuracy:.5f}")
    return model

def eval(model: nn.Module, test_dl: DataLoader):
    model.eval()
    device = next(model.parameters()).device  # Get device from model parameters
    all_logits = []
    all_targets = []
    with torch.no_grad():  # No gradients needed for evaluation
        for inputs, targets in test_dl:
            inputs, targets = inputs.to(device).long(), targets.to(device).long()
            output = model(inputs)
            logits = output.logits
            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return compute_accuracy(all_logits, all_targets)

if __name__ == "__main__":
    set_seed(MANUAL_SEED)
    
    data_cfg = DataConfig(
        num_train_examples=100_000,
        num_test_examples=3_000,
        input_seq_len=512,  # 4 * 128 (longer sequence for 128 KV pairs)
        vocab_size=513,     # 4 * 128 + 1 (larger vocab for 128 KV pairs)
        batch_size=256,
        num_kv_pairs=128,    # Increased from 16 to 128 (8x harder task!)
        train_power_a=0.01,
        test_power_a=0.01,
        random_non_queries=False,
        seed=MANUAL_SEED,
    )
    model_cfg = GRUConfig(
        # core architecture
        vocab_size=513,     # Must match data vocab_size (4*128+1)
        hidden_size=1024,   # Internal dimension for GRU + attention
        num_hidden_layers=3,  # 3 layers for deep memory
        dropout=0.0,
        bidirectional=False,

        # MLP (SwiGLU) - GRU also supports these
        hidden_ratio=4,
        intermediate_size=4096,  # 1024 * 4 = 4096 for internal processing
        hidden_act="swish",

        # norms and numerics
        norm_eps=1e-6,
        elementwise_affine=True,
        fuse_norm=False,
        fuse_cross_entropy=False,
        fuse_linear_cross_entropy=False,
        use_l2warp=False,

        # runtime semantics
        use_cache=False,
        tie_word_embeddings=False,
    )

    train_dl, test_dl = get_dataloaders(data_cfg)

    # model_cfg = gru_cfg
    # we will use float32 for ours
    # GRU with Self-Attention and Weight Projection: 1024 internal → 256 output
    model = GRUWithSelfAttention(model_cfg, num_attention_heads=8, projection_dim=256).to(torch.device("cuda:0")).to(torch.float32)
    print(f"Number of parameters: {model.num_parameters():,}")
    model = torch.compile(model)    

    optimizer = optim.AdamW(model.parameters(), 
                            lr=5e-5,  # Very conservative LR as requested (1e-4)
                            weight_decay=0.01,  # Small weight decay for regularization
                            betas=(0.9, 0.95))  # Custom betas for better optimization
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)  # Enable scheduler for smooth convergence

    # from ipdb import set_trace; set_trace()
    model = train(model, train_dl, test_dl, optimizer, 60, scheduler)  # Evaluate accuracy after each epoch
    print("\n✅ Training completed!")
    # from ipdb import set_trace; set_trace()
