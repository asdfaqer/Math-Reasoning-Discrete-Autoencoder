import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Vector Quantizer Bottleneck layer with Cosine Quantization and Vector Resetting (Dead Code Revival).
    Quantizes continuous latent embeddings into discrete codebook indices (dict size up to 16384).
    Uses Straight-Through Estimator (STE) for backpropagation.
    """
    def __init__(
        self,
        num_embeddings=16384,
        embedding_dim=72,
        commitment_cost=0.25,
        reset_threshold=1.0,
        reset_interval=20,
        use_cosine=True
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.reset_threshold = reset_threshold
        self.reset_interval = reset_interval
        self.use_cosine = use_cosine

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        self.register_buffer("cluster_usage", torch.zeros(num_embeddings))
        self.register_buffer("step_count", torch.zeros(1, dtype=torch.long))

    def reset_dead_codes(self, flat_inputs):
        dead_mask = self.cluster_usage < self.reset_threshold
        num_dead = dead_mask.sum().item()
        
        if num_dead > 0 and flat_inputs.size(0) > 0:
            dead_indices = torch.where(dead_mask)[0]
            rand_batch_idx = torch.randint(0, flat_inputs.size(0), (num_dead,), device=flat_inputs.device)
            sampled_vectors = flat_inputs[rand_batch_idx].detach()
            jitter = torch.randn_like(sampled_vectors) * 0.01
            new_vectors = F.normalize(sampled_vectors + jitter, p=2, dim=1) if self.use_cosine else (sampled_vectors + jitter)
            
            with torch.no_grad():
                self.embedding.weight.data[dead_indices] = new_vectors
                self.cluster_usage[dead_indices] = self.reset_threshold
                
        self.cluster_usage.mul_(0.9)

    def forward(self, inputs, jitter_scale=0.0, reset_dead_codes=True):
        flat_inputs = inputs.reshape(-1, self.embedding_dim)

        if self.use_cosine:
            flat_inputs_norm = F.normalize(flat_inputs, p=2, dim=1)
            codebook_weights = F.normalize(self.embedding.weight, p=2, dim=1)
        else:
            flat_inputs_norm = flat_inputs
            codebook_weights = self.embedding.weight

        if self.training and jitter_scale > 0.0:
            jitter = torch.randn_like(flat_inputs_norm) * jitter_scale
            search_inputs = F.normalize(flat_inputs_norm + jitter, p=2, dim=1)
        else:
            search_inputs = flat_inputs_norm

        distances = (
            torch.sum(search_inputs ** 2, dim=1, keepdim=True)
            + torch.sum(codebook_weights ** 2, dim=1)
            - 2 * torch.matmul(search_inputs, codebook_weights.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices).view(inputs.shape)

        if self.training and reset_dead_codes:
            batch_counts = torch.bincount(encoding_indices, minlength=self.num_embeddings).float()
            self.cluster_usage.add_(batch_counts)
            self.step_count.add_(1)
            
            if self.step_count.item() % self.reset_interval == 0:
                self.reset_dead_codes(flat_inputs_norm)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        encoding_indices = encoding_indices.view(inputs.shape[0], inputs.shape[1])
        return quantized, loss, encoding_indices


class SoftHardVectorQuantizer(nn.Module):
    """
    Continuous-Discrete Probability Blended Vector Quantizer with Dead Code Revival.
    Generates a token probability distribution vector p_cont alongside the
    hard one-hot token vector y_onehot, and computes a convex blend before passing to decoder.
    """
    def __init__(
        self,
        num_embeddings=16384,
        embedding_dim=72,
        commitment_cost=0.25,
        reset_threshold=1.0,
        reset_interval=20,
        temperature=1.0,
        init_tau=0.5,
        tau_l1_coeff=0.01,
        use_cosine=True,
        learnable_tau=False
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.reset_threshold = reset_threshold
        self.reset_interval = reset_interval
        self.temperature = temperature
        self.tau_l1_coeff = tau_l1_coeff
        self.use_cosine = use_cosine
        self.learnable_tau = learnable_tau

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        self.register_buffer("cluster_usage", torch.zeros(num_embeddings))
        self.register_buffer("step_count", torch.zeros(1, dtype=torch.long))

        if self.learnable_tau:
            init_logit = torch.logit(torch.tensor(float(init_tau), dtype=torch.float32)).item()
            self.raw_tau = nn.Parameter(torch.tensor([init_logit], dtype=torch.float32))
        else:
            self.register_buffer("tau_val", torch.tensor([float(init_tau)], dtype=torch.float32))

    @property
    def tau(self):
        if self.learnable_tau:
            return torch.sigmoid(self.raw_tau)
        return self.tau_val

    def set_tau(self, val):
        val_float = float(val)
        if self.learnable_tau:
            val_clamped = max(1e-5, min(1.0 - 1e-5, val_float))
            init_logit = torch.logit(torch.tensor(val_clamped, dtype=torch.float32)).item()
            with torch.no_grad():
                self.raw_tau.fill_(init_logit)
        else:
            self.tau_val.fill_(val_float)

    def reset_dead_codes(self, flat_inputs):
        dead_mask = self.cluster_usage < self.reset_threshold
        num_dead = dead_mask.sum().item()
        
        if num_dead > 0 and flat_inputs.size(0) > 0:
            dead_indices = torch.where(dead_mask)[0]
            rand_batch_idx = torch.randint(0, flat_inputs.size(0), (num_dead,), device=flat_inputs.device)
            sampled_vectors = flat_inputs[rand_batch_idx].detach()
            jitter = torch.randn_like(sampled_vectors) * 0.01
            new_vectors = F.normalize(sampled_vectors + jitter, p=2, dim=1) if self.use_cosine else (sampled_vectors + jitter)
            
            with torch.no_grad():
                self.embedding.weight.data[dead_indices] = new_vectors
                self.cluster_usage[dead_indices] = self.reset_threshold
                
        self.cluster_usage.mul_(0.9)

    def forward(self, inputs, jitter_scale=0.0, reset_dead_codes=True, tau=None):
        B, L, D = inputs.shape
        flat_inputs = inputs.reshape(-1, D)

        if self.use_cosine:
            flat_inputs_norm = F.normalize(flat_inputs, p=2, dim=1)
            codebook_weights = F.normalize(self.embedding.weight, p=2, dim=1)
            logits = torch.matmul(flat_inputs_norm, codebook_weights.t()) / self.temperature
        else:
            dists = (
                torch.sum(flat_inputs ** 2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight ** 2, dim=1)
                - 2 * torch.matmul(flat_inputs, self.embedding.weight.t())
            )
            logits = -dists / self.temperature

        encoding_indices = torch.argmax(logits, dim=1)
        z_hard = self.embedding(encoding_indices).view(B, L, D)
        z_hard_ste = inputs + (z_hard - inputs).detach()

        if tau is not None:
            current_tau = torch.tensor(float(tau), dtype=inputs.dtype, device=inputs.device)
        else:
            current_tau = self.tau.to(inputs.device).to(inputs.dtype)

        if current_tau > 0.0:
            p_cont = F.softmax(logits, dim=-1)
            z_soft = torch.matmul(p_cont, self.embedding.weight).view(B, L, D)
            quantized = current_tau * z_soft + (1.0 - current_tau) * z_hard_ste
        else:
            quantized = z_hard_ste

        if self.training and reset_dead_codes:
            batch_counts = torch.bincount(encoding_indices, minlength=self.num_embeddings).float()
            self.cluster_usage.add_(batch_counts)
            self.step_count.add_(1)
            
            if self.step_count.item() % self.reset_interval == 0:
                self.reset_dead_codes(flat_inputs_norm if self.use_cosine else flat_inputs)

        e_latent_loss = F.mse_loss(z_hard.detach(), inputs)
        q_latent_loss = F.mse_loss(z_hard, inputs.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        if self.learnable_tau:
            tau_loss = self.tau_l1_coeff * current_tau.squeeze()
        else:
            tau_loss = torch.tensor(0.0, device=inputs.device)

        total_loss = vq_loss + tau_loss
        encoding_indices = encoding_indices.view(B, L)
        return quantized, total_loss, encoding_indices


class AccuracyProbe(nn.Module):
    """
    Lightweight MLP Accuracy Probe.
    Takes 64 codebook token indices, embeds them with the codebook embedding layer,
    zero-masks the dropped (64 - M) tail tokens (avoiding noisy learned embeddings),
    and predicts the expected sequence reconstruction accuracy in [0, 1].
    """
    def __init__(self, codebook_size=16384, d_model=72, max_length=64, hidden_dim=128, codebook_embedding=None):
        super().__init__()
        self.codebook_size = codebook_size
        self.d_model = d_model
        self.max_length = max_length
        self.hidden_dim = hidden_dim

        if codebook_embedding is not None:
            self.codebook_embedding = codebook_embedding
        else:
            self.codebook_embedding = nn.Embedding(codebook_size, d_model)

        self.pos_encoder = nn.Parameter(torch.randn(1, max_length, d_model) * 0.02)

        in_dim = max_length * d_model
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def embed_codes(self, code_indices, prefix_len=None):
        B, L = code_indices.shape
        device = code_indices.device
        z_emb = self.codebook_embedding(code_indices)

        if prefix_len is not None:
            if isinstance(prefix_len, int):
                if prefix_len < L:
                    zeros_tail = torch.zeros(B, L - prefix_len, self.d_model, device=device, dtype=z_emb.dtype)
                    z_emb = torch.cat([z_emb[:, :prefix_len, :], zeros_tail], dim=1)
            elif isinstance(prefix_len, torch.Tensor):
                seq_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
                prefix_exp = prefix_len.view(B, 1).expand(B, L)
                keep_mask = (seq_idx < prefix_exp).unsqueeze(-1)
                z_emb = z_emb * keep_mask.to(z_emb.dtype)

        z_emb = z_emb + self.pos_encoder[:, :L, :]
        return z_emb.reshape(B, L * self.d_model)

    def forward(self, code_indices, prefix_len=None):
        flat_emb = self.embed_codes(code_indices, prefix_len)
        pred_acc = self.mlp(flat_emb).squeeze(-1)
        return pred_acc

    def predict_full_curve(self, code_indices):
        B, L = code_indices.shape
        device = code_indices.device
        candidate_m = torch.arange(1, L + 1, device=device).repeat(B)
        expanded_codes = code_indices.repeat_interleave(L, dim=0)
        flat_emb = self.embed_codes(expanded_codes, candidate_m)
        preds = self.mlp(flat_emb).view(B, L)
        return preds




class VectorizedPositionPerceptron(nn.Module):
    """
    Single-layer perceptron per position with non-linear activation (GELU).
    Takes scalar normalized budget m_norm = M / max_length in (0, 1].
    Computes delta representations in a single vectorized kernel without autograd scattering:
        delta = GELU(W[:M] * m_norm + b[:M])
    """
    def __init__(self, max_length=64, d_model=72):
        super().__init__()
        self.max_length = max_length
        self.d_model = d_model
        # Parameterized as [max_length, d_model] weights and biases
        self.weight = nn.Parameter(torch.randn(max_length, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(max_length, d_model))
        self.act = nn.GELU()

    def forward(self, m_norm, length=None):
        if length is None:
            length = self.max_length
        w = self.weight[:length] # [length, d_model]
        b = self.bias[:length]   # [length, d_model]
        delta = self.act(w * m_norm + b) # [length, d_model]
        return delta.unsqueeze(0) # [1, length, d_model]

    def __len__(self):
        return self.max_length

    def __getitem__(self, idx):
        class _SliceProxy(nn.Module):
            def __init__(self, w, b, act):
                super().__init__()
                self.weight = w
                self.bias = b
                self.act = act
            def forward(self, x):
                return self.act(self.weight.unsqueeze(0) * x + self.bias.unsqueeze(0))
        return _SliceProxy(self.weight[idx], self.bias[idx], self.act)


class TopMAttentionIndexer(nn.Module):
    """
    Attention-Guided Indexer:
    Uses the budget embedding c_M as an attention query to score candidate latent tokens.
    Selects the top M most relevant tokens dynamically and applies straight-through gradient gating.
    Computes an auxiliary load-balancing loss across the batch to ensure all 64 slots receive gradients.
    """
    def __init__(self, d_model=128, noise_std=0.05, balance_loss_weight=0.1):
        super().__init__()
        self.d_model = d_model
        self.noise_std = noise_std
        self.balance_loss_weight = balance_loss_weight
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, c_m, h_latents, k=32, noise_std=None):
        """
        c_m: [B, 1, d_model]
        h_latents: [B, N, d_model] (e.g. N = 64)
        k: int, number of top tokens to select
        """
        B, N, D = h_latents.shape
        sigma = noise_std if noise_std is not None else (self.noise_std if self.training else 0.0)

        # 1. Attention Scoring
        q = self.q_proj(c_m) # [B, 1, D]
        k_vecs = self.k_proj(h_latents) # [B, N, D]
        scores = torch.bmm(q, k_vecs.transpose(1, 2)).squeeze(1) * self.scale # [B, N]

        # Gumbel exploration during training to avoid local minima
        if self.training and sigma > 0.0:
            u = torch.rand_like(scores).clamp(min=1e-6, max=1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(u))
            scores_perturbed = scores + gumbel * sigma
        else:
            scores_perturbed = scores

        # 2. Top-K Selection
        eff_k = min(k, N)
        topk_indices = torch.topk(scores_perturbed, k=eff_k, dim=-1).indices # [B, eff_k]
        topk_indices, _ = torch.sort(topk_indices, dim=-1) # [B, eff_k]

        # 3. Straight-Through Gradient Routing
        probs = F.softmax(scores, dim=-1) # [B, N]
        probs_topk = torch.gather(probs, 1, topk_indices) # [B, eff_k]
        gate = probs_topk / probs_topk.detach().clamp(min=1e-8) # [B, eff_k]

        idx_exp = topk_indices.unsqueeze(-1).expand(-1, -1, D) # [B, eff_k, D]
        selected_latents = torch.gather(h_latents, 1, idx_exp) * gate.unsqueeze(-1) # [B, eff_k, D]

        # 4. Auxiliary Load Balancing Loss
        if self.balance_loss_weight > 0.0 and self.training and B > 1:
            mask = torch.zeros_like(scores).scatter_(-1, topk_indices, 1.0) # [B, N]
            slot_usage = mask.mean(dim=0) # [N]
            target_usage = float(eff_k) / float(N)
            balance_loss = F.mse_loss(slot_usage, torch.full_like(slot_usage, target_usage)) * self.balance_loss_weight
        else:
            balance_loss = torch.tensor(0.0, device=h_latents.device)

        return selected_latents, topk_indices, balance_loss, scores


class CascadingMemoryEncoder(nn.Module):
    """
    Symmetric Cross-Attention Transformer Encoder with Raw Text as Memory.
    - Raw text embeddings [B, 64, d_model] serve as the cross-attention 'memory' (K, V).
    - Learned latent queries [1, 64, d_model] serve as the 'tgt' sequence (Q).
    - Causal self-attention mask on queries ensures token i only attends to tokens 0 ... i.
      This forces a strictly CASCADING level of detail across the 64 latent tokens,
      guaranteeing that prefix tokens 0 ... M-1 are mathematically invariant to tail pruning.
    - Quantizes directly through the discrete VQ codebook (NO conv, NO MLP).
    """
    def __init__(
        self,
        vocab_size=50265,
        d_model=72,
        nhead=4,
        num_layers=3,
        codebook_size=4096,
        max_length=64,
        use_cosine=True,
        reset_dead_codes=False,
        pos_dropout=0.0,
        use_tau_skip=False,
        init_tau=0.5,
        tau_l1_coeff=0.01,
        temperature=1.0,
        learnable_tau=False,
        norm_first=True,
        text_encoder_layers=3,
        causal_cross_attn=True,
        causal_encoder_queries=True,
        normalize_prefix_pos=False,
        adaptive_prefix_queries=False,
        use_compression_embeddings=False,
        append_m_query=True,
        mlp_prefix_pos=False,
        mlp_pos_hidden_dim=64,
        bottleneck_dim=None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.max_length = max_length
        self.use_cosine = use_cosine
        self.reset_dead_codes = reset_dead_codes
        self.use_tau_skip = use_tau_skip
        self.learnable_tau = learnable_tau
        self.norm_first = norm_first
        self.text_encoder_layers = text_encoder_layers
        self.causal_cross_attn = causal_cross_attn
        self.causal_encoder_queries = causal_encoder_queries
        self.normalize_prefix_pos = normalize_prefix_pos
        self.adaptive_prefix_queries = adaptive_prefix_queries
        self.use_compression_embeddings = use_compression_embeddings
        self.append_m_query = append_m_query
        self.mlp_prefix_pos = mlp_prefix_pos
        self.mlp_pos_hidden_dim = mlp_pos_hidden_dim

        # Per-position learned single-layer perceptron with GELU activation
        if mlp_prefix_pos:
            self.mlp_pos = VectorizedPositionPerceptron(max_length=max_length, d_model=d_model)
        else:
            self.mlp_pos = None

        # Option A: Learned compression-level vector for each budget M in [1, max_length]
        if use_compression_embeddings:
            self.compression_embeddings = nn.Embedding(max_length + 1, d_model)
            nn.init.normal_(self.compression_embeddings.weight, mean=0.0, std=0.02)

        # Raw text embedding and positional encoding
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.text_embedding.weight, mean=0.0, std=0.02)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_length + 64, d_model) * 0.02)
        self.pos_dropout = nn.Dropout(p=pos_dropout) if pos_dropout > 0.0 else nn.Identity()

        # Bidirectional TransformerEncoder on raw text tokens (full contextualization + direct residual highway)
        if text_encoder_layers > 0:
            text_enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                batch_first=True,
                norm_first=norm_first
            )
            self.text_encoder = nn.TransformerEncoder(text_enc_layer, num_layers=text_encoder_layers)
            self.text_norm = nn.LayerNorm(d_model)
        else:
            self.text_encoder = None
            self.text_norm = None

        # Learned latent query tokens for the 64 positions
        self.latent_queries = nn.Parameter(torch.randn(1, max_length, d_model) * 0.02)

        # Transformer cross-attention stack (TransformerDecoderLayer: self-attn on tgt + cross-attn to memory)
        encoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=norm_first
        )
        self.transformer = nn.TransformerDecoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Optional dimension downsampling bottleneck before codebook
        if bottleneck_dim is not None and bottleneck_dim < d_model:
            self.down_proj = nn.Linear(d_model, bottleneck_dim, bias=False)
            self.up_proj = nn.Linear(bottleneck_dim, d_model, bias=False)
            self.up_norm = nn.LayerNorm(d_model)
            vq_dim = bottleneck_dim
        else:
            self.down_proj = None
            self.up_proj = None
            self.up_norm = None
            vq_dim = d_model

        # Discrete VQ Codebook (Direct quantization without conv or mlp)
        if use_tau_skip:
            self.vq = SoftHardVectorQuantizer(
                num_embeddings=codebook_size,
                embedding_dim=vq_dim,
                temperature=temperature,
                init_tau=init_tau,
                tau_l1_coeff=tau_l1_coeff,
                use_cosine=use_cosine,
                learnable_tau=learnable_tau
            )
        else:
            self.vq = VectorQuantizer(
                num_embeddings=codebook_size,
                embedding_dim=vq_dim,
                use_cosine=use_cosine
            )

    @property
    def pos_proj(self):
        return self.mlp_pos

    def set_tau(self, val):
        if hasattr(self.vq, "set_tau"):
            self.vq.set_tau(val)

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_causal_memory_mask(self, tgt_sz, src_sz, device):
        """
        Ensures Query position i can ONLY attend to Text Token positions j <= i.
        Shape: [tgt_sz, src_sz]
        """
        i = torch.arange(tgt_sz, device=device).unsqueeze(1) # [tgt_sz, 1]
        j = torch.arange(src_sz, device=device).unsqueeze(0) # [1, src_sz]
        mask = (j <= i).float()
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, input_ids, attention_mask=None, prefix_len=None, jitter_scale=0.0, tau=None, return_continuous=False):
        B, L = input_ids.shape
        device = input_ids.device

        # 1. Prepare raw text memory: [B, L, d_model]
        pos_emb = self.pos_dropout(self.pos_encoder[:, :L, :])
        raw_text_memory = self.text_embedding(input_ids) + pos_emb

        # Memory key padding mask (if provided)
        mem_key_padding_mask = None
        if attention_mask is not None:
            # HuggingFace attention_mask is 1 for valid, 0 for pad. PyTorch requires True for pad.
            mem_key_padding_mask = (attention_mask == 0)

        # Contextualize raw text with bidirectional TransformerEncoder stack
        if self.text_encoder is not None:
            raw_text_memory = self.text_encoder(raw_text_memory, src_key_padding_mask=mem_key_padding_mask)
            raw_text_memory = self.text_norm(raw_text_memory)

        # 2. Determine target query count M and generate queries
        M = prefix_len if prefix_len is not None else self.max_length
        M = max(1, min(M, self.max_length))

        # Slot positional encoding
        if getattr(self, "mlp_prefix_pos", False) and self.mlp_pos is not None:
            m_norm = torch.tensor(float(M) / float(self.max_length), device=device, dtype=self.pos_encoder.dtype)
            dyn_queries = self.mlp_pos(m_norm, M) # [1, M, d_model]
            query_pos = dyn_queries
            indiv_queries = self.latent_queries[:, :M, :] + dyn_queries
        elif getattr(self, "normalize_prefix_pos", False) and M > 1:
            pos_table = self.pos_encoder[:, :self.max_length, :].transpose(1, 2)
            query_pos = F.interpolate(pos_table, size=M, mode='linear', align_corners=True).transpose(1, 2)
            indiv_queries = self.latent_queries[:, :M, :] + query_pos
        else:
            query_pos = self.pos_encoder[:, :M, :]
            indiv_queries = self.latent_queries[:, :M, :] + query_pos

        if getattr(self, "use_compression_embeddings", False) and getattr(self, "compression_embeddings", None) is not None:
            m_idx = torch.tensor(M, device=device)
            c_m = self.compression_embeddings(m_idx).view(1, 1, -1) # [1, 1, d_model]
            if getattr(self, "append_m_query", True):
                # Append learned M embedding as another separate query: [1, M + 1, d_model]
                tgt_queries = torch.cat([indiv_queries, c_m], dim=1).expand(B, -1, -1)
            else:
                # Elementwise add
                tgt_queries = (indiv_queries + c_m).expand(B, -1, -1)
        elif not getattr(self, "mlp_prefix_pos", False) and getattr(self, "adaptive_prefix_queries", False) and M < self.max_length:
            query_table = self.latent_queries[:, :self.max_length, :].transpose(1, 2)
            interp_queries = F.adaptive_avg_pool1d(query_table, M).transpose(1, 2)
            tgt_queries = (interp_queries + query_pos).expand(B, -1, -1)
        else:
            tgt_queries = indiv_queries.expand(B, -1, -1)

        M_tgt = tgt_queries.size(1)

        # 3. Causal mask on queries to enforce cascading level of detail (optional for non-causal encoder)
        tgt_mask = self.generate_square_subsequent_mask(M_tgt, device) if self.causal_encoder_queries else None

        # 4. Causal mask on memory cross-attention (Query i only attends to Text 0..i)
        if self.causal_cross_attn:
            memory_mask = self.generate_causal_memory_mask(M_tgt, L, device)
            if M_tgt > M:
                # Appended budget query at index M must NOT leak future text tokens beyond horizon M;
                # it only attends causally to text tokens 0..M-1 (strictly respecting budget horizon M)
                memory_mask[M:, :] = float('-inf')
                memory_mask[M:, :M] = 0.0
        else:
            memory_mask = None

        # 5. Cross-attention: queries attend to themselves and to raw text memory
        h_latents = self.transformer(
            tgt=tgt_queries,
            memory=raw_text_memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            memory_key_padding_mask=mem_key_padding_mask
        )
        h_latents = self.norm(h_latents) # [B, M_tgt, d_model]

        # 5. Direct Vector Quantization (with optional bottleneck)
        if self.down_proj is not None:
            # Project and L2-normalize to unit sphere so bottleneck vectors match codebook radius (norm=1.0)
            latents_for_vq = F.normalize(self.down_proj(h_latents), p=2, dim=-1)
            h_continuous = self.up_norm(self.up_proj(latents_for_vq)) if self.up_norm is not None else self.up_proj(latents_for_vq)
        else:
            latents_for_vq = h_latents
            h_continuous = h_latents

        if hasattr(self.vq, "forward") and self.use_tau_skip:
            quantized_bottleneck, vq_loss, code_indices = self.vq(
                latents_for_vq,
                jitter_scale=jitter_scale,
                reset_dead_codes=self.reset_dead_codes,
                tau=tau
            )
        else:
            quantized_bottleneck, vq_loss, code_indices = self.vq(
                latents_for_vq,
                jitter_scale=jitter_scale,
                reset_dead_codes=self.reset_dead_codes
            )

        if self.up_proj is not None:
            # Ensure quantized bottleneck vectors are on the unit sphere before up-projection
            normed_quantized_bottleneck = F.normalize(quantized_bottleneck, p=2, dim=-1)
            quantized = self.up_norm(self.up_proj(normed_quantized_bottleneck)) if self.up_norm is not None else self.up_proj(normed_quantized_bottleneck)
        else:
            quantized = quantized_bottleneck

        if return_continuous:
            return quantized, vq_loss, code_indices, h_continuous
        return quantized, vq_loss, code_indices


class CascadingMemoryDecoder(nn.Module):
    """
    Symmetric Cross-Attention Transformer Decoder with Latent Tokens as Memory.
    - Retained prefix latents [B, M, d_model] serve as the cross-attention 'memory' (K, V).
    - Latents first undergo bidirectional self-attention to coordinate prefix context.
    - Target text tokens [B, L, d_model] serve as 'tgt' (Q) with causal self-attention.
    - Input and output embeddings are tied to unify representation geometry.
    - Decodes full text autoregressively or teacher-forced during training.
    """
    def __init__(
        self,
        vocab_size=50265,
        d_model=72,
        nhead=4,
        num_layers=3,
        max_length=64,
        pos_dropout=0.0,
        norm_first=True,
        latent_self_attn_layers=1,
        shared_embedding=None,
        tie_weights=True,
        normalize_prefix_pos=True,
        prefix_pos_type="interpolated",
        append_m_query=False,
        mlp_prefix_pos=False,
        mlp_pos_hidden_dim=64
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_length = max_length
        self.norm_first = norm_first
        self.latent_self_attn_layers = latent_self_attn_layers
        self.normalize_prefix_pos = normalize_prefix_pos
        self.prefix_pos_type = prefix_pos_type
        self.append_m_query = append_m_query
        self.mlp_prefix_pos = mlp_prefix_pos
        self.mlp_pos_hidden_dim = mlp_pos_hidden_dim

        # Per-position learned single-layer perceptron with GELU activation
        if mlp_prefix_pos:
            self.mlp_pos = VectorizedPositionPerceptron(max_length=max_length, d_model=d_model)
        else:
            self.mlp_pos = None

        if shared_embedding is not None:
            self.text_embedding = shared_embedding
        else:
            self.text_embedding = nn.Embedding(vocab_size, d_model)
            nn.init.normal_(self.text_embedding.weight, mean=0.0, std=0.02)

        self.pos_encoder = nn.Parameter(torch.randn(1, max_length + 64, d_model) * 0.02)
        self.pos_dropout = nn.Dropout(p=pos_dropout) if pos_dropout > 0.0 else nn.Identity()

        # Bidirectional self-attention on retained prefix latents to coordinate prefix memory
        if latent_self_attn_layers > 0:
            latent_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                batch_first=True,
                norm_first=norm_first
            )
            self.latent_self_attn = nn.TransformerEncoder(latent_layer, num_layers=latent_self_attn_layers)
            self.latent_norm = nn.LayerNorm(d_model)
        else:
            self.latent_self_attn = None
            self.latent_norm = None

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=norm_first
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size, bias=False if tie_weights else True)
        if tie_weights:
            self.fc_out.weight = self.text_embedding.weight

    @property
    def pos_proj(self):
        return self.mlp_pos

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(
        self,
        tgt_input_ids,
        memory_latents,
        memory_key_padding_mask=None,
        k_samples=1,
        noise_std=0.0,
        exploration_mode="latent",
        selected_indices=None
    ):
        """
        tgt_input_ids: [B, L] or [B * K, L] shifted text input
        memory_latents: [B, M, d_model] retained prefix latents (M <= 64)
        k_samples: number of exploration candidates (K >= 1)
        noise_std: exploration noise standard deviation
        exploration_mode: 'latent', 'gumbel', 'both'
        selected_indices: [B, M] or [B, M-1] optional slot indices gathered by indexer
        """
        B, M, D = memory_latents.shape
        L = tgt_input_ids.shape[-1]
        device = memory_latents.device

        if k_samples > 1 and (self.training or noise_std > 0.0):
            # Expand memory latents: [B * K, M, D]
            mem_exp = memory_latents.unsqueeze(1).repeat(1, k_samples, 1, 1).view(B * k_samples, M, D)
            if noise_std > 0.0 and exploration_mode in ["latent", "both"]:
                mem_exp = mem_exp + torch.randn_like(mem_exp) * noise_std

            # Expand target text if not already expanded
            if tgt_input_ids.shape[0] == B:
                tgt_exp = tgt_input_ids.unsqueeze(1).repeat(1, k_samples, 1).view(B * k_samples, L)
            else:
                tgt_exp = tgt_input_ids
        else:
            k_samples = 1
            mem_exp = memory_latents
            tgt_exp = tgt_input_ids

        # Target text embedding + pos encoding
        pos_emb = self.pos_dropout(self.pos_encoder[:, :L, :])
        tgt_emb = self.text_embedding(tgt_exp) + pos_emb

        # Latent memory positional encoding (normalized based on M if enabled)
        if selected_indices is not None:
            pos_m_len = selected_indices.shape[1]
            has_appended_m = getattr(self, "append_m_query", False) and (M > pos_m_len)
        else:
            has_appended_m = getattr(self, "append_m_query", False) and M > 1
            pos_m_len = M - 1 if has_appended_m else M

        cur_b = mem_exp.shape[0]
        if selected_indices is not None:
            # Gather slot positional encodings matching the dynamically selected tokens
            if k_samples > 1 and selected_indices.shape[0] != cur_b:
                sel_exp = selected_indices.unsqueeze(1).repeat(1, k_samples, 1).view(cur_b, pos_m_len)
            else:
                sel_exp = selected_indices
            pos_table = self.pos_encoder[:, :self.max_length, :].expand(cur_b, -1, -1)
            pos_m = torch.gather(pos_table, 1, sel_exp.unsqueeze(-1).expand(-1, -1, D))
        elif getattr(self, "mlp_prefix_pos", False) and self.mlp_pos is not None:
            m_norm = torch.tensor(float(pos_m_len) / float(self.max_length), device=device, dtype=self.pos_encoder.dtype)
            dyn_pos = self.mlp_pos(m_norm, pos_m_len) # [1, pos_m_len, d_model]
            pos_m = self.pos_encoder[:, :pos_m_len, :] + dyn_pos
        elif self.normalize_prefix_pos and pos_m_len > 1:
            if self.prefix_pos_type == "interpolated":
                pos_table = self.pos_encoder[:, :self.max_length, :].transpose(1, 2)
                pos_m = F.interpolate(pos_table, size=pos_m_len, mode='linear', align_corners=True).transpose(1, 2)
            elif self.prefix_pos_type == "sinusoidal":
                pos_norm = torch.linspace(0, 1, steps=pos_m_len, device=device).unsqueeze(0).unsqueeze(-1)
                half_d = D // 2
                freqs = torch.exp(torch.arange(half_d, device=device).float() * (-math.log(10000.0) / max(1, half_d - 1)))
                args_sin = pos_norm * self.max_length * freqs.unsqueeze(0).unsqueeze(0)
                pos_m = torch.cat([torch.sin(args_sin), torch.cos(args_sin)], dim=-1)
                if pos_m.shape[-1] < D:
                    pos_m = F.pad(pos_m, (0, D - pos_m.shape[-1]))
            else:
                pos_m = self.pos_encoder[:, :pos_m_len, :]
        else:
            pos_m = self.pos_encoder[:, :pos_m_len, :]

        if has_appended_m:
            # Dedicated positional embedding for the appended M budget token
            m_token_pos = self.pos_encoder[:, self.max_length:self.max_length+1, :]
            mem_pos = torch.cat([pos_m, m_token_pos], dim=1)
        else:
            mem_pos = pos_m

        memory_with_pos = mem_exp + mem_pos

        # Latents self-attend among themselves (cooperative prefix memory)
        if self.latent_self_attn is not None:
            memory_with_pos = self.latent_self_attn(memory_with_pos)
            memory_with_pos = self.latent_norm(memory_with_pos)

        # Causal mask for autoregressive teacher forcing
        tgt_mask = self.generate_square_subsequent_mask(L, device)

        decoded = self.transformer(
            tgt=tgt_emb,
            memory=memory_with_pos,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        decoded = self.norm(decoded)
        logits = self.fc_out(decoded) # [B * K, L, vocab_size]

        # Gumbel exploration if requested
        if k_samples > 1 and exploration_mode in ["gumbel", "both"] and self.training:
            u = torch.rand_like(logits).clamp(min=1e-6, max=1.0 - 1e-6)
            gumbel_noise = -torch.log(-torch.log(u))
            logits = logits + gumbel_noise * max(noise_std, 0.01)

        if k_samples > 1:
            return logits.view(B, k_samples, L, self.vocab_size)
        else:
            return logits

    @torch.no_grad()
    def generate(self, memory_latents, start_token_id=0, max_length=64, selected_indices=None):
        """
        Autoregressive generation from retained prefix latents.
        memory_latents: [B, M, d_model]
        Returns: [B, max_length] generated token IDs (excluding initial start_token_id).
        """
        device = memory_latents.device
        B = memory_latents.shape[0]
        generated_ids = torch.full((B, 1), start_token_id, dtype=torch.long, device=device)

        for _ in range(max_length):
            logits = self.forward(generated_ids, memory_latents, k_samples=1, selected_indices=selected_indices)
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        # Drop initial start prompt token so output has exactly max_length tokens
        return generated_ids[:, 1:]


class CascadingMemoryAutoencoder(nn.Module):
    """
    End-to-End Cascading Cross-Attention Autoencoder with Exploration Trick:
      Encoder: Raw Text as Memory -> 64 Cascading Latents -> VQ Codebook
      Dropout: Retain prefix M <= 64 latents
      Decoder: Retained Latents as Memory -> Reconstructed Text
      Exploration: Generates K candidate hypotheses, optimizes strictly closest candidate
                   (Winner-Takes-All) to prevent predicting the mean.
    """
    def __init__(
        self,
        vocab_size=50265,
        d_model=72,
        nhead=4,
        num_layers=3,
        decoder_num_layers=3,
        text_encoder_layers=3,
        codebook_size=4096,
        max_length=64,
        k_samples=4,
        noise_std=0.1,
        exploration_mode="latent",
        selection_metric="loss",
        probe_hidden_dim=128,
        probe_loss_weight=1.0,
        use_cosine=True,
        reset_dead_codes=False,
        pos_dropout=0.0,
        use_tau_skip=False,
        init_tau=0.5,
        tau_l1_coeff=0.01,
        temperature=1.0,
        learnable_tau=False,
        fixed_prefix_len=32,
        prefix_distribution="fixed",
        min_prefix_len=8,
        max_prefix_len=64,
        word_dropout=0.1,
        bos_token_id=0,
        pad_token_id=1,
        skip_bottleneck=False,
        vq_loss_weight=1.0,
        norm_first=True,
        causal_cross_attn=True,
        causal_encoder_queries=True,
        normalize_prefix_pos=True,
        prefix_pos_type="interpolated",
        latent_grad_scale_mode="none",
        latent_grad_scale_max=20.0,
        adaptive_prefix_queries=False,
        use_compression_embeddings=False,
        append_m_query=True,
        diversity_loss_weight=0.5,
        diversity_threshold=0.2,
        diversity_loss_type="squared",
        latent_self_attn_layers=1,
        share_embeddings=True,
        tie_weights=True,
        mlp_prefix_pos=False,
        mlp_pos_hidden_dim=64,
        bottleneck_dim=None,
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.max_length = max_length
        self.codebook_size = codebook_size
        self.k_samples = k_samples
        self.noise_std = noise_std
        self.exploration_mode = exploration_mode
        self.selection_metric = selection_metric
        self.probe_loss_weight = probe_loss_weight
        self.fixed_prefix_len = fixed_prefix_len
        self.prefix_distribution = prefix_distribution
        self.min_prefix_len = min_prefix_len
        self.max_prefix_len = max_prefix_len
        self.word_dropout = word_dropout
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.skip_bottleneck = skip_bottleneck
        self.vq_loss_weight = vq_loss_weight
        self.norm_first = norm_first
        self.causal_cross_attn = causal_cross_attn
        self.causal_encoder_queries = causal_encoder_queries
        self.normalize_prefix_pos = normalize_prefix_pos
        self.prefix_pos_type = prefix_pos_type
        self.latent_grad_scale_mode = latent_grad_scale_mode
        self.latent_grad_scale_max = latent_grad_scale_max
        self.adaptive_prefix_queries = adaptive_prefix_queries
        self.use_compression_embeddings = use_compression_embeddings
        self.append_m_query = append_m_query
        self.diversity_loss_weight = diversity_loss_weight
        self.diversity_threshold = diversity_threshold
        self.diversity_loss_type = diversity_loss_type
        self.text_encoder_layers = text_encoder_layers
        self.latent_self_attn_layers = latent_self_attn_layers
        self.share_embeddings = share_embeddings
        self.tie_weights = tie_weights
        self.mlp_prefix_pos = mlp_prefix_pos
        self.mlp_pos_hidden_dim = mlp_pos_hidden_dim

        # 1. Cascading Encoder (Raw text as memory, learned queries as tgt)
        self.encoder = CascadingMemoryEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            codebook_size=codebook_size,
            max_length=max_length,
            use_cosine=use_cosine,
            reset_dead_codes=reset_dead_codes,
            pos_dropout=pos_dropout,
            use_tau_skip=use_tau_skip,
            init_tau=init_tau,
            tau_l1_coeff=tau_l1_coeff,
            temperature=temperature,
            learnable_tau=learnable_tau,
            norm_first=norm_first,
            text_encoder_layers=text_encoder_layers,
            causal_cross_attn=causal_cross_attn,
            causal_encoder_queries=causal_encoder_queries,
            normalize_prefix_pos=normalize_prefix_pos,
            adaptive_prefix_queries=adaptive_prefix_queries,
            use_compression_embeddings=use_compression_embeddings,
            append_m_query=append_m_query,
            mlp_prefix_pos=mlp_prefix_pos,
            mlp_pos_hidden_dim=mlp_pos_hidden_dim,
            bottleneck_dim=bottleneck_dim
        )

        # 2. Cascading Decoder (Retained latents as memory, text as tgt)
        self.decoder = CascadingMemoryDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=decoder_num_layers,
            max_length=max_length,
            pos_dropout=pos_dropout,
            norm_first=norm_first,
            latent_self_attn_layers=latent_self_attn_layers,
            shared_embedding=self.encoder.text_embedding if share_embeddings else None,
            tie_weights=tie_weights,
            normalize_prefix_pos=normalize_prefix_pos,
            prefix_pos_type=prefix_pos_type,
            append_m_query=append_m_query,
            mlp_prefix_pos=mlp_prefix_pos,
            mlp_pos_hidden_dim=mlp_pos_hidden_dim
        )

        # 3. Accuracy Probe (predicts accuracy curve from discrete codes)
        probe_dim = bottleneck_dim if (bottleneck_dim is not None and bottleneck_dim < d_model) else d_model
        self.probe = AccuracyProbe(
            codebook_size=codebook_size,
            d_model=probe_dim,
            max_length=max_length,
            hidden_dim=probe_hidden_dim,
            codebook_embedding=self.encoder.vq.embedding
        )

        # 4. Attention-Guided Indexer (Top-M selection using budget query c_M)
        self.indexer_mode = kwargs.get("indexer_mode", "prefix")
        self.balance_loss_weight = kwargs.get("balance_loss_weight", 0.1)
        self.indexer_noise_std = kwargs.get("indexer_noise_std", 0.05)

        if self.indexer_mode == "attention_topk":
            self.indexer = TopMAttentionIndexer(
                d_model=d_model,
                noise_std=self.indexer_noise_std,
                balance_loss_weight=self.balance_loss_weight
            )
        else:
            self.indexer = None

    def set_tau(self, val):
        self.encoder.set_tau(val)

    def compute_token_accuracy(self, logits, target_ids):
        """
        logits: [B, L, V] or [B, K, L, V]
        target_ids: [B, L]
        """
        non_pad_mask = (target_ids != self.pad_token_id) & (target_ids != self.bos_token_id)
        if non_pad_mask.sum() == 0:
            non_pad_mask = (target_ids != self.pad_token_id)

        if logits.dim() == 4:
            B, K, L, V = logits.shape
            pred_ids = torch.argmax(logits, dim=-1) # [B, K, L]
            target_expanded = target_ids.unsqueeze(1).expand(B, K, L)
            mask_expanded = non_pad_mask.unsqueeze(1).expand(B, K, L)
            correct = (pred_ids == target_expanded) & mask_expanded
            acc = correct.sum(dim=-1).float() / mask_expanded.sum(dim=-1).clamp(min=1).float() # [B, K]
            return acc
        else:
            pred_ids = torch.argmax(logits, dim=-1)
            correct = (pred_ids == target_ids) & non_pad_mask
            acc = correct.sum(dim=-1).float() / non_pad_mask.sum(dim=-1).clamp(min=1).float()
            return acc

    def forward(
        self,
        input_ids,
        attention_mask=None,
        prefix_len=None,
        jitter_scale=0.0,
        tau=None,
        train_probe=True,
        word_dropout=None,
        k_samples=None,
        noise_std=None,
        exploration_mode=None,
        selection_metric=None,
        **kwargs
    ):
        B, L = input_ids.shape
        device = input_ids.device

        K = k_samples if k_samples is not None else self.k_samples
        sigma = noise_std if noise_std is not None else self.noise_std
        mode = exploration_mode if exploration_mode is not None else self.exploration_mode
        metric = selection_metric if selection_metric is not None else self.selection_metric

        # 1. Determine prefix length M
        if prefix_len is not None:
            M = prefix_len if isinstance(prefix_len, int) else int(prefix_len[0].item())
        elif self.training and getattr(self, "prefix_distribution", "fixed") == "uniform":
            min_m = getattr(self, "min_prefix_len", 8)
            max_m = getattr(self, "max_prefix_len", self.max_length)
            M = torch.randint(min_m, max_m + 1, (1,)).item()
        elif self.fixed_prefix_len is not None and self.fixed_prefix_len > 0:
            M = self.fixed_prefix_len
        else:
            M = 32

        M = max(1, min(M, self.max_length))

        # 2. Encode raw text into compression-adaptive latents
        is_attention_topk = getattr(self, "indexer_mode", "prefix") == "attention_topk" and getattr(self, "indexer", None) is not None
        encoder_prefix = self.max_length if is_attention_topk else M

        quantized, vq_loss, code_indices, h_latents = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            prefix_len=encoder_prefix,
            jitter_scale=jitter_scale,
            tau=tau,
            return_continuous=True
        )

        # 3. Retain latents as memory (skip bottleneck uses continuous latents)
        skip = kwargs.get("skip_bottleneck", self.skip_bottleneck)
        latents_source = h_latents if skip else quantized
        has_appended = getattr(self, "append_m_query", False) and getattr(self, "use_compression_embeddings", False)

        topk_indices = None
        balance_loss = torch.tensor(0.0, device=device)

        if is_attention_topk:
            # Budget embedding c_m representing budget M
            if getattr(self, "use_compression_embeddings", False) and getattr(self.encoder, "compression_embeddings", None) is not None:
                m_idx = torch.tensor(M, device=device)
                c_m = self.encoder.compression_embeddings(m_idx).view(1, 1, -1).expand(B, 1, -1)
            else:
                c_m = latents_source[:, -1:, :] if latents_source.shape[1] > self.max_length else latents_source[:, :1, :]

            cand_latents = latents_source[:, :self.max_length, :]
            selected_latents, topk_indices, balance_loss, attn_scores = self.indexer(c_m, cand_latents, k=M)

            if has_appended:
                memory_latents = torch.cat([selected_latents, c_m], dim=1)
            else:
                memory_latents = selected_latents
        else:
            expected_len = M + 1 if has_appended else M
            memory_latents = latents_source if latents_source.shape[1] == expected_len else latents_source[:, :expected_len, :]

            # Gradient normalization across variable M prefix tokens to avoid later-token starvation
            grad_mode = getattr(self, "latent_grad_scale_mode", "none")
            if self.training and grad_mode != "none":
                cur_mem_len = memory_latents.size(1)
                k_idx = torch.arange(cur_mem_len, device=device).float()
                max_scale = getattr(self, "latent_grad_scale_max", 20.0)
                if grad_mode == "inv_freq":
                    total_range = float(self.max_prefix_len - self.min_prefix_len + 1)
                    prob = (float(self.max_prefix_len + 1) - torch.clamp(k_idx, min=float(self.min_prefix_len))) / max(1.0, total_range)
                    weights = torch.clamp(1.0 / prob, max=max_scale).view(1, cur_mem_len, 1)
                    memory_latents = memory_latents * 1.0
                    memory_latents.register_hook(lambda g, w=weights: g * w)
                elif grad_mode == "m_over_min_m":
                    scale = float(M) / max(1.0, float(self.min_prefix_len))
                    memory_latents = memory_latents * 1.0
                    memory_latents.register_hook(lambda g, s=scale: g * s)
                elif grad_mode == "combined":
                    total_range = float(self.max_prefix_len - self.min_prefix_len + 1)
                    prob = (float(self.max_prefix_len + 1) - torch.clamp(k_idx, min=float(self.min_prefix_len))) / max(1.0, total_range)
                    inv_p = torch.clamp(1.0 / prob, max=max_scale)
                    scale = math.sqrt(float(M) / max(1.0, float(self.min_prefix_len)))
                    weights = (inv_p * scale).view(1, M, 1)
                    memory_latents = memory_latents * 1.0
                    memory_latents.register_hook(lambda g, w=weights: g * w)

        # 4. Prepare teacher-forcing target text
        tgt_input = torch.zeros_like(input_ids)
        tgt_input[:, 0] = self.bos_token_id
        tgt_input[:, 1:] = input_ids[:, :-1]

        # Target Word Dropout with independent mask per exploration branch
        w_drop = word_dropout if word_dropout is not None else self.word_dropout
        if self.training and K > 1 and w_drop > 0.0:
            tgt_exp = tgt_input.unsqueeze(1).repeat(1, K, 1).view(B * K, L)
            drop_mask = torch.rand(tgt_exp.shape, device=device) < w_drop
            drop_mask[:, 0] = False
            tgt_exp = tgt_exp.masked_fill(drop_mask, self.pad_token_id)
        elif self.training and w_drop > 0.0:
            drop_mask = torch.rand(tgt_input.shape, device=device) < w_drop
            drop_mask[:, 0] = False
            tgt_exp = tgt_input.masked_fill(drop_mask, self.pad_token_id)
        else:
            tgt_exp = tgt_input

        # 5. Decode text with exploration
        logits = self.decoder(
            tgt_exp,
            memory_latents,
            k_samples=K if self.training else 1,
            noise_std=sigma if self.training else 0.0,
            exploration_mode=mode,
            selected_indices=topk_indices
        )

        non_pad_count = (input_ids != self.pad_token_id).sum(dim=1).clamp(min=1)

        # 6. Closest-Sample Selection (Winner-Takes-All / Min-of-K)
        if self.training and K > 1:
            flat_logits = logits.view(B * K * L, self.vocab_size)
            flat_targets = input_ids.unsqueeze(1).expand(B, K, L).contiguous().view(-1)

            token_losses = F.cross_entropy(
                flat_logits,
                flat_targets,
                ignore_index=self.pad_token_id,
                reduction='none'
            ).view(B, K, L)

            loss_per_candidate = token_losses.sum(dim=-1) / non_pad_count.unsqueeze(1) # [B, K]
            acc_per_candidate = self.compute_token_accuracy(logits, input_ids) # [B, K]

            if metric == "acc":
                best_k = torch.argmax(acc_per_candidate, dim=-1) # [B]
            else:
                best_k = torch.argmin(loss_per_candidate, dim=-1) # [B]

            b_indices = torch.arange(B, device=device)
            closest_rec_loss = loss_per_candidate[b_indices, best_k]
            rec_loss = closest_rec_loss.mean()

            true_acc = acc_per_candidate[b_indices, best_k]
            chosen_logits = logits[b_indices, best_k]
            candidate_variance = loss_per_candidate.var(dim=-1).mean().item()
            win_counts = (torch.bincount(best_k, minlength=K).float() / B).tolist()
        else:
            if logits.dim() == 4:
                logits = logits.squeeze(1)
            chosen_logits = logits
            loss_matrix = F.cross_entropy(
                chosen_logits.reshape(-1, self.vocab_size),
                input_ids.reshape(-1),
                ignore_index=self.pad_token_id,
                reduction='none'
            ).view(B, L)
            rec_loss = (loss_matrix.sum(dim=1) / non_pad_count).mean()
            true_acc = self.compute_token_accuracy(chosen_logits, input_ids)
            candidate_variance = 0.0
            win_counts = [1.0]

        mean_true_acc = true_acc.mean().item()

        # 7. Accuracy Probe Training
        m_tensor = torch.full((B,), M, dtype=torch.long, device=device)
        codes_for_probe = code_indices[:, :self.max_length]
        probe_codes = F.pad(codes_for_probe, (0, max(0, self.max_length - codes_for_probe.shape[1])), value=0)
        pred_acc = self.probe(probe_codes, prefix_len=m_tensor)
        probe_loss = F.mse_loss(pred_acc, true_acc.detach())
        mean_pred_acc = pred_acc.mean().item()

        # 8. Intra-Sequence Latent Diversity Loss (Anti-Collapse Penalty)
        if self.diversity_loss_weight > 0.0 and M > 1:
            # Enforce diversity directly on continuous encoder latents feeding the VQ codebook
            normed_mem = F.normalize(h_latents[:, :M, :], p=2, dim=-1) # [B, M, D]
            sim_matrix = torch.bmm(normed_mem, normed_mem.transpose(1, 2)) # [B, M, M]
            eye = torch.eye(M, device=device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
            off_diag = sim_matrix.masked_select(~eye)
            mean_intra_cos = off_diag.mean().item()

            # Violator-focused diversity: only penalize pairs where cosine exceeds threshold.
            # Normalizes ONLY by the count of violating pairs to prevent 4,000-pair dilution.
            violator_mask = off_diag > self.diversity_threshold
            if violator_mask.any():
                violator_diff = off_diag[violator_mask] - self.diversity_threshold
                if self.diversity_loss_type == "linear":
                    diversity_loss = violator_diff.mean()
                else: # default / "squared" / "hinge": squared excess over threshold
                    diversity_loss = (violator_diff ** 2).mean()
            else:
                diversity_loss = torch.tensor(0.0, device=device)
        else:
            diversity_loss = torch.tensor(0.0, device=device)
            mean_intra_cos = 0.0

        total_loss = rec_loss + (self.vq_loss_weight * vq_loss) + \
                     (self.probe_loss_weight * probe_loss if train_probe else 0.0) + \
                     (self.diversity_loss_weight * diversity_loss) + \
                     balance_loss

        return {
            "loss": total_loss,
            "rec_loss": rec_loss,
            "vq_loss": vq_loss,
            "probe_loss": probe_loss,
            "diversity_loss": diversity_loss,
            "balance_loss": balance_loss,
            "mean_intra_cos": mean_intra_cos,
            "logits": chosen_logits,
            "code_indices": code_indices,
            "prefix_len": M,
            "topk_indices": topk_indices,
            "true_acc": true_acc,
            "mean_true_acc": mean_true_acc,
            "mean_pred_acc": mean_pred_acc,
            "candidate_variance": candidate_variance,
            "win_counts": win_counts
        }

    @torch.no_grad()
    def generate(self, input_ids, prefix_len=32, max_length=64):
        """
        Compresses input_ids to M prefix latents and reconstructs full text autoregressively.
        """
        quantized, _, _, h_latents = self.encoder(input_ids, return_continuous=True)
        latents_source = h_latents if self.skip_bottleneck else quantized

        if getattr(self, "indexer_mode", "prefix") == "attention_topk" and getattr(self, "indexer", None) is not None:
            B = input_ids.shape[0]
            if getattr(self, "use_compression_embeddings", False) and getattr(self.encoder, "compression_embeddings", None) is not None:
                m_idx = torch.tensor(prefix_len, device=input_ids.device)
                c_m = self.encoder.compression_embeddings(m_idx).view(1, 1, -1).expand(B, 1, -1)
            else:
                c_m = latents_source[:, -1:, :] if latents_source.shape[1] > self.max_length else latents_source[:, :1, :]
            selected_latents, topk_indices, _, _ = self.indexer(c_m, latents_source[:, :self.max_length, :], k=prefix_len)
            has_appended = getattr(self, "append_m_query", False) and getattr(self, "use_compression_embeddings", False)
            if has_appended:
                memory_latents = torch.cat([selected_latents, c_m], dim=1)
            else:
                memory_latents = selected_latents
            return self.decoder.generate(memory_latents, start_token_id=self.bos_token_id, max_length=max_length, selected_indices=topk_indices)
        else:
            memory_latents = latents_source[:, :prefix_len, :]
            return self.decoder.generate(memory_latents, start_token_id=self.bos_token_id, max_length=max_length)


class SmallCascadingMemoryAutoencoder(CascadingMemoryAutoencoder):
    """
    Standard small-scale Cascading Cross-Attention Autoencoder with Exploration Trick.
    d_model=72, nhead=4, num_layers=3, decoder_num_layers=3, text_encoder_layers=3, codebook_size=4096.
    """
    def __init__(
        self,
        vocab_size=50265,
        d_model=72,
        nhead=4,
        num_layers=3,
        decoder_num_layers=3,
        text_encoder_layers=3,
        codebook_size=4096,
        max_length=64,
        k_samples=4,
        noise_std=0.1,
        exploration_mode="latent",
        selection_metric="loss",
        probe_hidden_dim=128,
        probe_loss_weight=1.0,
        use_cosine=True,
        reset_dead_codes=False,
        pos_dropout=0.0,
        use_tau_skip=False,
        init_tau=0.5,
        tau_l1_coeff=0.01,
        temperature=1.0,
        learnable_tau=False,
        fixed_prefix_len=32,
        prefix_distribution="fixed",
        min_prefix_len=8,
        max_prefix_len=64,
        word_dropout=0.1,
        bos_token_id=0,
        pad_token_id=1,
        skip_bottleneck=False,
        vq_loss_weight=1.0,
        norm_first=True,
        causal_cross_attn=True,
        causal_encoder_queries=True,
        normalize_prefix_pos=True,
        prefix_pos_type="interpolated",
        latent_grad_scale_mode="none",
        latent_grad_scale_max=20.0,
        adaptive_prefix_queries=False,
        use_compression_embeddings=False,
        append_m_query=True,
        diversity_loss_weight=0.5,
        diversity_threshold=0.2,
        diversity_loss_type="squared",
        latent_self_attn_layers=1,
        share_embeddings=True,
        tie_weights=True,
        mlp_prefix_pos=False,
        mlp_pos_hidden_dim=64,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            decoder_num_layers=decoder_num_layers,
            text_encoder_layers=text_encoder_layers,
            norm_first=norm_first,
            causal_cross_attn=causal_cross_attn,
            causal_encoder_queries=causal_encoder_queries,
            normalize_prefix_pos=normalize_prefix_pos,
            prefix_pos_type=prefix_pos_type,
            latent_grad_scale_mode=latent_grad_scale_mode,
            latent_grad_scale_max=latent_grad_scale_max,
            adaptive_prefix_queries=adaptive_prefix_queries,
            use_compression_embeddings=use_compression_embeddings,
            append_m_query=append_m_query,
            diversity_loss_weight=diversity_loss_weight,
            diversity_threshold=diversity_threshold,
            diversity_loss_type=diversity_loss_type,
            latent_self_attn_layers=latent_self_attn_layers,
            share_embeddings=share_embeddings,
            tie_weights=tie_weights,
            mlp_prefix_pos=mlp_prefix_pos,
            mlp_pos_hidden_dim=mlp_pos_hidden_dim,
            codebook_size=codebook_size,
            max_length=max_length,
            k_samples=k_samples,
            noise_std=noise_std,
            exploration_mode=exploration_mode,
            selection_metric=selection_metric,
            probe_hidden_dim=probe_hidden_dim,
            probe_loss_weight=probe_loss_weight,
            use_cosine=use_cosine,
            reset_dead_codes=reset_dead_codes,
            pos_dropout=pos_dropout,
            use_tau_skip=use_tau_skip,
            init_tau=init_tau,
            tau_l1_coeff=tau_l1_coeff,
            temperature=temperature,
            learnable_tau=learnable_tau,
            fixed_prefix_len=fixed_prefix_len,
            prefix_distribution=prefix_distribution,
            min_prefix_len=min_prefix_len,
            max_prefix_len=max_prefix_len,
            word_dropout=word_dropout,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            skip_bottleneck=skip_bottleneck,
            vq_loss_weight=vq_loss_weight,
            **kwargs
        )


class DecoderOnlyAblationModel(nn.Module):
    """
    Pure Autoregressive Next-Token Prediction Language Model (Encoder Ablation).
    - Completely ablates the encoder.
    - Completely skips cross-attention layers.
    - Does NOT feed any prefix or memory tokens into the decoder.
    - Standard causal autoregressive modeling: [<BOS>, x_0, ..., x_{L-2}] -> predicts [x_0, ..., x_{L-1}].
    - Evaluates pure next-token prediction accuracy on the dataset.
    """
    def __init__(
        self,
        vocab_size=50265,
        d_model=72,
        nhead=4,
        decoder_num_layers=3,
        max_length=64,
        bos_token_id=0,
        pad_token_id=1,
        norm_first=True,
        tie_weights=True,
        word_dropout=0.0,
        pos_dropout=0.0,
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_length = max_length
        self.decoder_num_layers = decoder_num_layers
        self.word_dropout = word_dropout
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.norm_first = norm_first
        self.codebook_size = 0
        self.skip_bottleneck = False

        self.text_embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.text_embedding.weight, mean=0.0, std=0.02)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_length + 64, d_model) * 0.02)
        self.pos_dropout = nn.Dropout(p=pos_dropout) if pos_dropout > 0.0 else nn.Identity()

        # Pure causal Transformer encoder layers (NO cross-attention layers)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=norm_first
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=decoder_num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size, bias=False if tie_weights else True)
        if tie_weights:
            self.fc_out.weight = self.text_embedding.weight

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def compute_token_accuracy(self, logits, target_ids):
        non_pad_mask = (target_ids != self.pad_token_id) & (target_ids != self.bos_token_id)
        if non_pad_mask.sum() == 0:
            non_pad_mask = (target_ids != self.pad_token_id)

        pred_ids = torch.argmax(logits, dim=-1)
        correct = (pred_ids == target_ids) & non_pad_mask
        acc = correct.sum(dim=-1).float() / non_pad_mask.sum(dim=-1).clamp(min=1).float()
        return acc

    def forward(
        self,
        input_ids,
        attention_mask=None,
        word_dropout=None,
        **kwargs
    ):
        B, L = input_ids.shape
        device = input_ids.device

        # Autoregressive teacher-forcing shifted input: [<BOS>, x_0, ..., x_{L-2}]
        tgt_input = torch.zeros_like(input_ids)
        tgt_input[:, 0] = self.bos_token_id
        tgt_input[:, 1:] = input_ids[:, :-1]

        # Word dropout if specified during training
        w_drop = word_dropout if word_dropout is not None else self.word_dropout
        if self.training and w_drop > 0.0:
            drop_mask = torch.rand(tgt_input.shape, device=device) < w_drop
            drop_mask[:, 0] = False
            tgt_input = tgt_input.masked_fill(drop_mask, self.pad_token_id)

        tgt_pos = self.pos_dropout(self.pos_encoder[:, :L, :])
        tgt_emb = self.text_embedding(tgt_input) + tgt_pos

        # Pure causal subsequent mask (strictly autoregressive, NO cross-attention)
        tgt_mask = self.generate_square_subsequent_mask(L, device)

        decoded = self.transformer(tgt_emb, mask=tgt_mask)
        decoded = self.norm(decoded)

        logits = self.fc_out(decoded) # [B, L, vocab_size]

        # Reconstruction loss and next-token accuracy
        non_pad_count = (input_ids != self.pad_token_id).sum(dim=1).clamp(min=1)
        loss_matrix = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            input_ids.reshape(-1),
            ignore_index=self.pad_token_id,
            reduction='none'
        ).view(B, L)
        rec_loss = (loss_matrix.sum(dim=1) / non_pad_count).mean()
        true_acc = self.compute_token_accuracy(logits, input_ids)

        dummy_codes = torch.zeros((B, 1), dtype=torch.long, device=device)

        return {
            "loss": rec_loss,
            "rec_loss": rec_loss,
            "vq_loss": torch.tensor(0.0, device=device),
            "probe_loss": torch.tensor(0.0, device=device),
            "diversity_loss": torch.tensor(0.0, device=device),
            "mean_intra_cos": 0.0,
            "logits": logits,
            "code_indices": dummy_codes,
            "prefix_len": 0,
            "true_acc": true_acc,
            "mean_true_acc": true_acc.mean().item(),
            "mean_pred_acc": 0.0,
            "candidate_variance": 0.0,
            "win_counts": [1.0]
        }


class SmallDecoderOnlyAblationModel(DecoderOnlyAblationModel):
    """
    Standard small-scale Decoder-Only Next-Token Prediction Language Model (Encoder Ablation).
    d_model=72, nhead=4, decoder_num_layers=3.
    """
    def __init__(
        self,
        vocab_size=50265,
        d_model=72,
        nhead=4,
        decoder_num_layers=3,
        max_length=64,
        bos_token_id=0,
        pad_token_id=1,
        norm_first=True,
        tie_weights=True,
        word_dropout=0.0,
        pos_dropout=0.0,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            decoder_num_layers=decoder_num_layers,
            max_length=max_length,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            norm_first=norm_first,
            tie_weights=tie_weights,
            word_dropout=word_dropout,
            pos_dropout=pos_dropout,
            **kwargs
        )

