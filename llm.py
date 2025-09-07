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
    expert_top_k: int = 2
    load_balancing_weight: float = 0.01
    noise_std: float = 0.1

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

class TopKRouter(nn.Module):
    """Router that selects top-k experts for each token"""
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2, noise_std: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.noise_std = noise_std  # Standard deviation for noise during training

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            - router_weights: Softmax weights for selected experts [batch_size, seq_len, top_k]
            - expert_indices: Indices of selected experts [batch_size, seq_len, top_k]
            - router_probs: Full probability distribution over experts (for load balancing loss)
        """
        batch_size, seq_len, d_model = x.shape

        # Compute router logits
        router_logits = self.gate(x)  # [batch_size, seq_len, num_experts]

        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise

        # Get full probability distribution (for load balancing loss)
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        return top_k_weights, top_k_indices, router_probs

class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer with top-k routing"""
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        load_balancing_weight: float = 0.01,
        noise_std: float = 0.1
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balancing_weight = load_balancing_weight

        # Create experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])

        # Create router
        self.router = TopKRouter(d_model, num_experts, top_k, noise_std)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            - output: MoE output [batch_size, seq_len, d_model]
            - aux_loss: Load balancing auxiliary loss (only during training)
        """
        batch_size, seq_len, d_model = x.shape

        # Get routing decisions
        router_weights, expert_indices, router_probs = self.router(x)

        # Initialize output tensor
        output = torch.zeros_like(x)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (expert_indices == expert_idx).any(dim=-1)  # [batch_size, seq_len]

            if expert_mask.any():
                # Get tokens for this expert
                expert_input = x[expert_mask]  # [num_tokens, d_model]

                # Apply expert
                expert_output = self.experts[expert_idx](expert_input)

                # Get weights for this expert - CORRECTED APPROACH
                # First get the mask for this expert's positions
                mask_for_expert = (expert_indices == expert_idx)  # [batch, seq, top_k]
                # Find which position (0 or 1) this expert appears in for relevant tokens
                positions = mask_for_expert[expert_mask].float().argmax(dim=-1)
                # Gather weights only for relevant tokens
                expert_weights = router_weights[expert_mask].gather(
                    -1, positions.unsqueeze(-1)
                ).squeeze(-1)

                # Add weighted expert output to result
                output[expert_mask] += expert_weights.unsqueeze(-1) * expert_output

        # Compute load balancing loss during training
        aux_loss = None
        if self.training:
            aux_loss = self._compute_load_balancing_loss(router_probs, expert_indices)

        return output, aux_loss

    def _compute_load_balancing_loss(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary loss to ensure balanced expert usage.
        This encourages the router to distribute tokens evenly across experts.
        """
        # Compute the fraction of tokens routed to each expert
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts).float()
        tokens_per_expert = expert_mask.sum(dim=[0, 1, 2]) / expert_mask.sum()

        # Compute the average probability of routing to each expert
        router_prob_mean = router_probs.mean(dim=[0, 1])

        # Load balancing loss encourages uniform distribution
        aux_loss = torch.sum(tokens_per_expert * router_prob_mean) * self.num_experts

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
        top_k: int = 2,
        dropout: float = 0.1,
        load_balancing_weight: float = 0.01,
        noise_std: float = 0.1
    ):
        super().__init__()

        # Attention layer
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)

        # MoE layer
        self.feed_forward = MixtureOfExperts(
            d_model, d_ff, num_experts, top_k, dropout, load_balancing_weight, noise_std
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
                config.expert_top_k,
                config.dropout,
                config.load_balancing_weight,
                config.noise_std
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

def compute_expert_usage_cv(model: nn.Module, val_loader: DataLoader, config: MoEModelConfig):
    """Compute Coefficient of Variation of expert usage across entire validation set"""
    model.eval()
    device = next(model.parameters()).device

    expert_usage_counts = torch.zeros(config.num_experts, device=device)

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            # Forward pass to get routing decisions
            with autocast(enabled=config.use_amp):
                # We need to manually route through the MoE layers to collect statistics
                x_embed = model.token_embedding(x) * math.sqrt(config.d_model)
                x_embed = model.position_dropout(x_embed)

                # Pass through transformer blocks and collect routing info
                for block in model.transformer_blocks:
                    # Get routing decisions from the MoE layer
                    batch_size, seq_len, d_model = x_embed.shape
                    router_weights, expert_indices, _ = block.feed_forward.router(x_embed)

                    # Count expert usage (each token contributes to top-k experts)
                    for expert_idx in range(config.num_experts):
                        expert_mask = (expert_indices == expert_idx).any(dim=-1)  # [batch_size, seq_len]
                        expert_usage_counts[expert_idx] += expert_mask.sum()

                    # Continue with normal forward pass
                    x_embed, _ = block(x_embed)

    # Move to CPU for numpy operations
    expert_usage_counts = expert_usage_counts.cpu().numpy()

    # Compute Coefficient of Variation
    mean_usage = np.mean(expert_usage_counts)
    std_usage = np.std(expert_usage_counts)

    if mean_usage > 0:
        cv = std_usage / mean_usage
    else:
        cv = 0.0

    model.train()
    return cv

def evaluate_model(model: nn.Module, val_loader: DataLoader, config: MoEModelConfig, compute_expert_cv: bool = False):
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

    # Compute expert usage CV if requested
    expert_cv = None
    if compute_expert_cv:
        expert_cv = compute_expert_usage_cv(model, val_loader, config)

    model.train()

    result = {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}
    if expert_cv is not None:
        result['expert_usage_cv'] = expert_cv

    return result

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
    print(f"\n🚀 Training MoE model with {config.num_experts} experts (top-{config.expert_top_k})")

    # Initialize model
    model = MoEMinimalLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Track best validation perplexity
    best_val_perplexity = float('inf')

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    active_params = sum(p.numel() for n, p in model.named_parameters()
                       if 'expert' not in n)
    expert_params = total_params - active_params

    print(f"  📊 Total parameters: {total_params:,}")
    print(f"  📊 Active parameters: {active_params:,}")
    print(f"  📊 Expert parameters: {expert_params:,}")
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

                # Track best validation perplexity
                if eval_metrics['val_perplexity'] < best_val_perplexity:
                    best_val_perplexity = eval_metrics['val_perplexity']

            # Milestone evaluations
            if step in getattr(config, 'log_milestones', ()):
                eval_metrics = evaluate_model(model, val_loader, config)
                print(f"\n🧪 Milestone {step}: Val Loss: {eval_metrics['val_loss']:.4f}")

                # Track best validation perplexity
                if eval_metrics['val_perplexity'] < best_val_perplexity:
                    best_val_perplexity = eval_metrics['val_perplexity']

            step += 1
            if step % 100 == 0:
                pbar.update(100)

    pbar.close()

    # Final evaluation (compute expert CV for final evaluation)
    final_eval = evaluate_model(model, val_loader, config, compute_expert_cv=True)

    # Update best perplexity with final evaluation if better
    if final_eval['val_perplexity'] < best_val_perplexity:
        best_val_perplexity = final_eval['val_perplexity']

    # Add best perplexity to final metrics
    final_eval['best_val_perplexity'] = best_val_perplexity

    print(f"\n📊 Final Results:")
    print(f"   Val Loss: {final_eval['val_loss']:.4f}")
    print(f"   Val Accuracy: {final_eval['val_accuracy']:.4f}")
    print(f"   Val Perplexity: {final_eval['val_perplexity']:.2f}")
    print(f"   Best Val Perplexity: {best_val_perplexity:.2f}")
    if 'expert_usage_cv' in final_eval:
        print(f"   Expert Usage CV: {final_eval['expert_usage_cv']:.4f}")

    return model, final_eval

def run_single_experiment(config: MoEModelConfig, seed: int, train_loader: DataLoader, val_loader: DataLoader):
    """Run a single MoE training experiment"""
    print(f"\n🔬 Running experiment with seed={seed}, noise_std={config.noise_std}, load_balancing_weight={config.load_balancing_weight}")

    # Set seed for reproducibility
    set_seed(seed)

    # Reset data loaders with new seed for consistent splits
    temp_config = MoEModelConfig()
    texts, tokenizer, tokens = load_and_cache_data(temp_config)
    dataset = TextTokenDataset(tokens, temp_config.max_seq_len)

    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    # Track memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    model, final_metrics = train_moe_model(config, train_loader, val_loader)
    total_time = time.time() - start_time

    # Get memory usage
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
    else:
        peak_memory = 0.0

    results = {
        'seed': seed,
        'noise_std': config.noise_std,
        'load_balancing_weight': config.load_balancing_weight,
        'val_loss': final_metrics['val_loss'],
        'val_accuracy': final_metrics['val_accuracy'],
        'val_perplexity': final_metrics['val_perplexity'],
        'training_time_minutes': total_time / 60,
        'peak_memory_gb': peak_memory,
        'best_val_perplexity': final_metrics.get('best_val_perplexity', final_metrics['val_perplexity']),
        'expert_usage_cv': final_metrics.get('expert_usage_cv', None)
    }

    print(f"✅ Experiment completed in {total_time/60:.1f} minutes")
    return results

if __name__ == "__main__":
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data first to get vocab_size
    temp_config = MoEModelConfig()
    texts, tokenizer, tokens = load_and_cache_data(temp_config)
    vocab_size = temp_config.vocab_size

    # Experimental conditions
    noise_levels = [0.0, 0.1, 0.5]
    load_balancing_weights = [0.01, 0.0]
    seeds = [42, 43, 44]

    print(f"\n🧪 EXPERIMENTAL DESIGN: Routing Noise in Small-Scale MoEs")
    print(f"{'='*70}")
    print(f"Conditions: {len(noise_levels)} noise levels × {len(load_balancing_weights)} weight levels × {len(seeds)} seeds = {len(noise_levels) * len(load_balancing_weights) * len(seeds)} total runs")

    # Base configuration
    base_config = MoEModelConfig(
        # Base model parameters
        d_model=384,
        n_heads=8,
        n_layers=6,
        d_ff=1536,
        batch_size=24,
        max_steps=3000,
        eval_every=500,  # Every 500 steps as per research plan
        vocab_size=vocab_size,

        # MoE specific (will be overridden per experiment)
        num_experts=8,
        expert_top_k=2,
        load_balancing_weight=0.01,  # Will be overridden
        noise_std=0.1  # Will be overridden
    )

    # Store all results
    all_results = []

    # Run experiments
    for noise_std in noise_levels:
        for lb_weight in load_balancing_weights:
            for seed in seeds:
                # Create config for this experiment
                # Only copy the actual initialization parameters, not computed properties
                init_params = [
                    'd_model', 'n_heads', 'n_layers', 'd_ff', 'batch_size', 'max_steps',
                    'gradient_accumulation_steps', 'muon_lr', 'max_seq_len', 'num_documents',
                    'max_tokens', 'eval_every', 'eval_steps', 'weight_decay', 'dropout',
                    'grad_clip', 'use_amp', 'vocab_size', 'log_milestones', 'num_experts',
                    'expert_top_k', 'load_balancing_weight', 'noise_std'
                ]
                config_dict = {param: getattr(base_config, param) for param in init_params}
                config_dict['noise_std'] = noise_std
                config_dict['load_balancing_weight'] = lb_weight
                config = MoEModelConfig(**config_dict)

                # Run experiment
                result = run_single_experiment(config, seed, None, None)  # Data loaders created inside
                all_results.append(result)

    # Save results to CSV
    import csv
    results_file = "moe_routing_noise_experiment_results.csv"
    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['seed', 'noise_std', 'load_balancing_weight', 'val_loss', 'val_accuracy',
                     'val_perplexity', 'training_time_minutes', 'peak_memory_gb', 'best_val_perplexity', 'expert_usage_cv']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n💾 Results saved to {results_file}")

    # Summary statistics
    print(f"\n📊 EXPERIMENT SUMMARY")
    print(f"{'='*70}")

    for noise_std in noise_levels:
        for lb_weight in load_balancing_weights:
            condition_results = [r for r in all_results if r['noise_std'] == noise_std and r['load_balancing_weight'] == lb_weight]

            if condition_results:
                mean_ppl = sum(r['best_val_perplexity'] for r in condition_results) / len(condition_results)
                std_ppl = (sum((r['best_val_perplexity'] - mean_ppl)**2 for r in condition_results) / len(condition_results))**0.5
                mean_time = sum(r['training_time_minutes'] for r in condition_results) / len(condition_results)

                # Compute mean expert CV if available
                expert_cvs = [r['expert_usage_cv'] for r in condition_results if r['expert_usage_cv'] is not None]
                mean_cv = sum(expert_cvs) / len(expert_cvs) if expert_cvs else None

                print(f"Noise: {noise_std}, LB Weight: {lb_weight}")
                print(f"  Best Val PPL: {mean_ppl:.2f} ± {std_ppl:.2f}")
                print(f"  Avg Time: {mean_time:.1f} minutes")
                if mean_cv is not None:
                    print(f"  Expert Usage CV: {mean_cv:.4f}")
                print()

    print(f"✅ All {len(all_results)} experiments completed!")