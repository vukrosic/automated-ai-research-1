import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import time
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings
import os
import pickle
from torchtune.modules import RotaryPositionalEmbeddings
warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Set all seeds to {seed}")

@dataclass
class MoEModelConfig:
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    batch_size: int = 24
    max_steps: int = 3000

    # Training parameters
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.01

    # Data parameters
    max_seq_len: int = 512
    num_documents: int = 2000
    max_tokens: int = 500000

    # Evaluation
    eval_every: int = 500
    eval_steps: int = 100

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None
    log_milestones: Tuple[int, ...] = (2000, 5000, 10000)

    # MoE specific parameters
    num_experts: int = 8
    num_zero_experts: int = 4  # Number of zero-computation experts
    expert_top_k: int = 2
    expected_ffn_experts: float = 1.5  # Average FFN experts to activate (Ke in paper)
    bias_adaptation_rate: float = 0.01  # μ in the paper
    load_balancing_weight: float = 0.01

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X

class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
	
def load_and_cache_data(config: MoEModelConfig, cache_dir: str = "data_cache"):
    """Load and cache tokenized data to avoid reprocessing"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenized_data_{config.num_documents}_{config.max_tokens}.pkl"

    # Check if cached data exists
    if os.path.exists(cache_file):
        print(f"📦 Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        texts = cached_data['texts']
        tokenizer = cached_data['tokenizer']
        tokens = cached_data['tokens']
        config.vocab_size = tokenizer.vocab_size

        print(f"✅ Loaded {len(texts)} documents, {len(tokens):,} tokens from cache")
        return texts, tokenizer, tokens

    print(f"🔄 Processing new data (will cache for future use)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True, token=False)

    texts = []
    for i, item in enumerate(dataset):
        if i >= config.num_documents:
            break
        texts.append(item["text"][:3000])

    print(f"Loaded {len(texts)} documents")

    # Tokenize
    print("Tokenizing texts...")
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

    tokens = all_tokens[:config.max_tokens]
    print(f"Using {len(tokens):,} tokens")
    config.vocab_size = tokenizer.vocab_size

    # Cache the processed data
    cached_data = {'texts': texts, 'tokenizer': tokenizer, 'tokens': tokens}
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)

    print(f"💾 Cached data to {cache_file}")
    return texts, tokenizer, tokens

class TextTokenDataset(Dataset):
    def __init__(self, tokens: List[int], seq_len: int = 512):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.rope = RotaryPositionalEmbeddings(dim=dim, max_seq_len=max_seq_len, base=10000)

    def forward(self, x_BTHD: torch.Tensor):
        # x_BTHD shape: [B, T, H, D] - need to convert to [B, T, H, D] for torchtune
        # torchtune expects [batch, seq_len, num_heads, head_dim]
        # Our input is already [B, T, H, D] which matches torchtune's expectation
        return self.rope(x_BTHD)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        # B, T = x.size(0), x.size(1)
        # qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        # Q, K, V = qkv[0], qkv[1], qkv[2]  # [B, H, T, D]

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2] # [B, H, T, D]

        # Q = self.rotary(Q)
        # K = self.rotary(K)
        # Apply RoPE on [B, T, H, D]
        Q = self.rotary(Q.transpose(1, 2)).transpose(1, 2)
        K = self.rotary(K.transpose(1, 2)).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        # attn_output = attn_output.transpose(1, 2).reshape(B, T, self.d_model)
        return self.w_o(attn_output)



class Expert(nn.Module):
    """Single expert network (essentially a FeedForward layer)"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.silu(self.linear1(x))))

class ZeroComputationExpert(nn.Module):
    """Zero-computation expert that returns input unchanged"""
    def __init__(self):
        super().__init__()
        # No parameters needed - this is just an identity function

    def forward(self, x):
        return x  # Simply return input unchanged

class TopKRouter(nn.Module):
    """Router that selects top-k experts with adaptive bias for zero-computation experts"""
    def __init__(self, d_model: int, num_experts: int, num_zero_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.num_zero_experts = num_zero_experts
        self.total_experts = num_experts + num_zero_experts
        self.top_k = top_k

        # Gate projects to all experts (regular + zero-computation)
        self.gate = nn.Linear(d_model, self.total_experts, bias=False)
        self.noise_std = 0.1

        # Initialize expert bias (learnable but updated via PID controller)
        # Only FFN experts have bias, zero-computation experts have 0 bias
        self.register_buffer('expert_bias', torch.zeros(self.total_experts))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape

        # Compute router logits
        router_logits = self.gate(x)  # [batch_size, seq_len, total_experts]

        # Add expert bias (only for FFN experts, zero for zero-computation experts)
        router_logits = router_logits + self.expert_bias.unsqueeze(0).unsqueeze(0)

        # Add noise during training
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise

        # Get full probability distribution
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        return top_k_weights, top_k_indices, router_probs

class MixtureOfExperts(nn.Module):
    """Mixture of Experts with zero-computation experts"""
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        num_zero_experts: int = 4,
        top_k: int = 2,
        expected_ffn_experts: float = 1.5,
        bias_adaptation_rate: float = 0.01,
        dropout: float = 0.1,
        load_balancing_weight: float = 0.01
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_zero_experts = num_zero_experts
        self.total_experts = num_experts + num_zero_experts
        self.top_k = top_k
        self.expected_ffn_experts = expected_ffn_experts
        self.bias_adaptation_rate = bias_adaptation_rate
        self.load_balancing_weight = load_balancing_weight

        # Create FFN experts
        self.ffn_experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])

        # Create zero-computation experts (just placeholders, no parameters)
        self.zero_experts = nn.ModuleList([
            ZeroComputationExpert() for _ in range(num_zero_experts)
        ])

        # Combined expert list for easier indexing
        self.all_experts = self.ffn_experts + self.zero_experts

        # Create router with bias support
        self.router = TopKRouter(d_model, num_experts, num_zero_experts, top_k)

        # Track expert usage for bias updates
        self.register_buffer('expert_usage', torch.zeros(self.total_experts))
        self.register_buffer('total_tokens', torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, d_model = x.shape

        # Get routing decisions
        router_weights, expert_indices, router_probs = self.router(x)

        # Track expert usage for bias updates
        if self.training:
            with torch.no_grad():
                # Count tokens routed to each expert
                expert_mask = F.one_hot(expert_indices, num_classes=self.total_experts).float()
                tokens_per_expert = expert_mask.sum(dim=[0, 1])  # Sum over batch and seq_len
                self.expert_usage += tokens_per_expert.sum(dim=0)  # Sum over top_k
                self.total_tokens += batch_size * seq_len

        # Initialize output tensor
        output = torch.zeros_like(x)

        # Process each expert
        for expert_idx in range(self.total_experts):
            # Find tokens routed to this expert
            expert_mask = (expert_indices == expert_idx).any(dim=-1)

            if expert_mask.any():
                # Get tokens for this expert
                expert_input = x[expert_mask]

                # Apply appropriate expert (FFN or zero-computation)
                if expert_idx < self.num_experts:
                    # Regular FFN expert
                    expert_output = self.ffn_experts[expert_idx](expert_input)
                else:
                    # Zero-computation expert (identity)
                    expert_output = self.zero_experts[expert_idx - self.num_experts](expert_input)

                # Get weights for this expert
                mask_for_expert = (expert_indices == expert_idx)
                positions = mask_for_expert[expert_mask].float().argmax(dim=-1)
                expert_weights = router_weights[expert_mask].gather(
                    -1, positions.unsqueeze(-1)
                ).squeeze(-1)

                # Add weighted expert output
                output[expert_mask] += expert_weights.unsqueeze(-1) * expert_output

        # Update expert bias based on usage (PID controller)
        if self.training and self.total_tokens > 0:
            self._update_expert_bias()

        # Compute auxiliary loss
        aux_loss = None
        if self.training:
            aux_loss = self._compute_load_balancing_loss(router_probs, expert_indices)

        return output, aux_loss

    def _update_expert_bias(self):
        """Update expert bias using PID controller from Equation 2"""
        with torch.no_grad():
            # Calculate actual usage proportions
            usage_proportions = self.expert_usage / (self.total_tokens * self.top_k + 1e-6)

            # Expected proportion for each FFN expert
            expected_ffn_proportion = self.expected_ffn_experts / (self.top_k * self.num_experts)

            # Update bias only for FFN experts (not zero-computation experts)
            for i in range(self.num_experts):
                # Calculate bias increment using PID controller
                actual_proportion = usage_proportions[i]
                target_proportion = expected_ffn_proportion

                # Equation 2 from paper
                delta_bi = self.bias_adaptation_rate * (target_proportion - actual_proportion)
                self.router.expert_bias[i] += delta_bi

            # Zero-computation experts don't get bias updates (remain at 0)
            # This is handled by initialization

            # Reset usage tracking periodically
            if self.total_tokens > 10000:  # Reset every 10k tokens
                self.expert_usage.zero_()
                self.total_tokens.zero_()

    def _compute_load_balancing_loss(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """Load balancing loss for all experts (including zero-computation)"""
        expert_mask = F.one_hot(expert_indices, num_classes=self.total_experts).float()
        tokens_per_expert = expert_mask.sum(dim=[0, 1, 2]) / expert_mask.sum()
        router_prob_mean = router_probs.mean(dim=[0, 1])
        aux_loss = torch.sum(tokens_per_expert * router_prob_mean) * self.total_experts
        return aux_loss * self.load_balancing_weight

class MoETransformerBlock(nn.Module):
    """Transformer block with MoE"""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        num_experts: int = 8,
        num_zero_experts: int = 4,
        top_k: int = 2,
        expected_ffn_experts: float = 1.5,
        bias_adaptation_rate: float = 0.01,
        dropout: float = 0.1
    ):
        super().__init__()

        # Attention layer
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)

        # MoE layer with zero-computation experts
        self.feed_forward = MixtureOfExperts(
            d_model, d_ff, num_experts, num_zero_experts, top_k,
            expected_ffn_experts, bias_adaptation_rate, dropout
        )

        # Normalization layers
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)

        # MoE feed-forward
        ff_out, aux_loss = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x, aux_loss


class MoEMinimalLLM(nn.Module):
    """Minimal LLM with Mixture of Experts"""
    def __init__(self, config: MoEModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        # Transformer blocks with MoE
        self.transformer_blocks = nn.ModuleList([
            MoETransformerBlock(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.max_seq_len,
                config.num_experts,
                config.num_zero_experts,
                config.expert_top_k,
                config.expected_ffn_experts,
                config.bias_adaptation_rate,
                config.dropout
            )
            for i in range(config.n_layers)
        ])

        # Output layers
        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

        # Language modeling head (tied with embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, return_aux_loss=True):
        # Token embeddings
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        # Collect auxiliary losses from MoE layers
        aux_losses = []

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x, aux_loss = block(x)
            if aux_loss is not None and return_aux_loss:
                aux_losses.append(aux_loss)

        # Output projection
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)

        # Combine auxiliary losses
        total_aux_loss = sum(aux_losses) if aux_losses else None

        if return_aux_loss:
            return logits, total_aux_loss
        return logits

def evaluate_model(model: nn.Module, val_loader: DataLoader, config: MoEModelConfig):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= config.eval_steps:
                break
            x, y = x.to(device), y.to(device)

            with autocast(enabled=config.use_amp):
                # MoE model evaluation
                logits = model(x, return_aux_loss=False)  # Don't return aux loss during eval
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y).sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))

    model.train()
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}

def setup_muon_optimizer(model: nn.Module, config: MoEModelConfig):
    """Setup Muon optimizer with hybrid approach"""
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
            'norm' not in name and 
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    muon_optimizer = Muon(muon_params, lr=config.muon_lr, momentum=0.95)
    adamw_optimizer = torch.optim.AdamW(adamw_params, lr=config.muon_lr*0.1, weight_decay=config.weight_decay)

    return [muon_optimizer, adamw_optimizer]


def train_moe_model(config: MoEModelConfig, train_loader: DataLoader, val_loader: DataLoader):
    """Train the MoE model"""
    print(f"\n🚀 Training MoE model with {config.num_experts} FFN + {config.num_zero_experts} zero experts (top-{config.expert_top_k})")

    # Initialize model
    set_seed(42)
    model = MoEMinimalLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    ffn_expert_params = sum(p.numel() for n, p in model.named_parameters()
                           if 'ffn_experts' in n)
    active_params = total_params - ffn_expert_params

    print(f"  📊 Total parameters: {total_params:,}")
    print(f"  📊 Active parameters: {active_params:,} ({active_params/total_params:.1f}% active)")
    print(f"  📊 FFN expert parameters: {ffn_expert_params:,}")
    print(f"  📊 Zero experts: {config.num_zero_experts} (0 parameters)")
    print(f"  📊 Parameter efficiency: {active_params/total_params:.1%} active per forward pass")

    # Setup optimizers
    optimizers = setup_muon_optimizer(model, config)

    # Learning rate schedule
    schedulers = []
    for optimizer in optimizers:
        warmup_steps = config.max_steps // 20
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)

    scaler = GradScaler() if config.use_amp else None

    # Training loop
    model.train()
    step = 0
    pbar = tqdm(total=config.max_steps, desc="Training MoE")

    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break

            x, y = x.to(device), y.to(device)

            # Forward pass
            if config.use_amp:
                with autocast():
                    logits, aux_loss = model(x, return_aux_loss=True)
                    ce_loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

                    # Combine main loss and auxiliary loss
                    total_loss = ce_loss
                    if aux_loss is not None:
                        total_loss = total_loss + aux_loss

                    loss = total_loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits, aux_loss = model(x, return_aux_loss=True)
                ce_loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

                total_loss = ce_loss
                if aux_loss is not None:
                    total_loss = total_loss + aux_loss

                loss = total_loss / config.gradient_accumulation_steps
                loss.backward()

            # Optimizer step
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                    for optimizer in optimizers:
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

            # Logging
            if step % 100 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = ce_loss.item()
                    perplexity = math.exp(min(current_loss, 20))

                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'aux': f'{aux_loss.item() if aux_loss is not None else 0:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}'
                })

            # Evaluation
            if step % config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, config)
                print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                      f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                      f"Val PPL: {eval_metrics['val_perplexity']:.2f}")

            # Milestone evaluations
            if step in getattr(config, 'log_milestones', ()):    
                eval_metrics = evaluate_model(model, val_loader, config)
                print(f"\n🧪 Milestone {step}: Val Loss: {eval_metrics['val_loss']:.4f}")

            step += 1
            if step % 100 == 0:
                pbar.update(100)

    pbar.close()

    # Final evaluation
    final_eval = evaluate_model(model, val_loader, config)
    print(f"\n📊 Final Results:")
    print(f"   Val Loss: {final_eval['val_loss']:.4f}")
    print(f"   Val Accuracy: {final_eval['val_accuracy']:.4f}")
    print(f"   Val Perplexity: {final_eval['val_perplexity']:.2f}")

    return model, final_eval

if __name__ == "__main__":
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Set seed
    set_seed(42)

    # Load data first to get vocab_size
    temp_config = MoEModelConfig()  # Use MoE config for data loading
    texts, tokenizer, tokens = load_and_cache_data(temp_config)
    vocab_size = temp_config.vocab_size

    # Test MoE model
    config = MoEModelConfig(
        # Base model parameters
        d_model=384,
        n_heads=8,
        n_layers=6,
        d_ff=1536,
        batch_size=24,
        max_steps=2000,
        eval_every=10000000,
        vocab_size=vocab_size,

        # MoE specific
        num_experts=6,  # Reduce FFN experts since we're adding zero experts
        num_zero_experts=4,  # Add 4 zero-computation experts
        expert_top_k=2,  # Select top-2 experts per token
        expected_ffn_experts=1.2,  # On average, use 1.2 FFN experts (0.8 zero experts)
        bias_adaptation_rate=0.01,  # Learning rate for bias updates
        load_balancing_weight=0.01
    )

    dataset = TextTokenDataset(tokens, temp_config.max_seq_len)

    # Train/val split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=temp_config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=temp_config.batch_size, shuffle=False, num_workers=2)

    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")

    # Train MoE model
    print(f"\n{'='*60}")
    print(f"🧪 TRAINING: Mixture of Experts Model")
    print(f"{'='*60}")

    print(f"\n📋 MoE Model Configuration:")
    print(f"   Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H, {config.d_ff}ff")
    print(f"   MoE: {config.num_experts} FFN + {config.num_zero_experts} zero experts, top-{config.expert_top_k} routing")
    print(f"   Training: {config.max_steps} steps, batch size {config.batch_size}")
    print(f"   Data: {config.max_tokens:,} tokens, seq_len {config.max_seq_len}")

    # Train model
    start_time = time.time()
    model, final_metrics = train_moe_model(config, train_loader, val_loader)
    total_time = time.time() - start_time

    print(f"\n🎯 MoE Model Results:")
    print(f"⏱️ Training time: {total_time/60:.1f} minutes")
    print(f"🏆 Final Results:")
    print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
    print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")
    print(f"{'='*60}")