"""
IterIS++: Enhanced LoRA Merging with MATS, CAMR, and DCS

This module implements the IterIS++ algorithm, an enhanced version of IterIS with:
- MATS (Momentum Accelerated Trajectory Stabilization): Anderson Acceleration for faster convergence
- CAMR (Curvature-Aware Manifold Regularization): Geometry-aware regularization
- DCS (Dynamic Conflict-aware Sample Reweighting): Sample weighting based on cross-model variance

Reference: IterIS: Iterative Inference-Solving Alignment for LoRA Merging (CVPR 2025)
"""

import os
import gc
import yaml
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from datasets import load_dataset
from safetensors import safe_open
from sklearn.metrics import f1_score
from eval_model import eval_iteris_model
from get_midfeatures import T5WithHooks, BartWithHooks, BlipWithHook
from transformers import AutoTokenizer, AutoProcessor
from get_midfeatures import get_all_midfeatures, get_samples, get_pretrain_matrix, get_lora_matrix

import warnings
import logging
from transformers import logging as transformers_logging

warnings.filterwarnings("ignore")

transformers_logging.set_verbosity_error()

GLUE_task_name = [
    "mnli", "rte",
    "cola", "sst2", "qqp",
    "qnli", "mrpc",
]
EMOTION_task_name = [
    "emoint", "emotion-cause",
    "tec", "isear",
]
SENTICAP_task_name = ['positive', 'negative']
FlickrStyle10k_task_name = ["roman", "humor"]
TASKS_blip_base = ['positive', 'negative', "roman", "humor"]


def get_loras_path(task_type, model_name):
    """Get paths to LoRA adapters based on task type and model name."""
    lora_path_dict = {}
    if 't5' in model_name and task_type == "GLUE_t5":
        for item in GLUE_task_name:
            lora_path_dict[item] = f"loras/GLUE-lora-t5/{item}"
    elif 'bart' in model_name and task_type == "GLUE_bart":
        for item in GLUE_task_name:
            lora_path_dict[item] = f"loras/GLUE-lora-bart/{item}"
    elif 't5-large' in model_name and task_type == "EMOTION_t5_large":
        for item in EMOTION_task_name:
            lora_path_dict[item] = f"loras/EMOTION-lora-t5/{item}"
    elif 'blip' in model_name and task_type == "TASKS_blip_base":
        for item in FlickrStyle10k_task_name:
            lora_path_dict[item] = f"loras/FlickrStyle10k-lora-blip/{item}"
        for item in SENTICAP_task_name:
            lora_path_dict[item] = f"loras/SENTICAP-lora-blip/{item}"
    return lora_path_dict


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# MATS: Momentum Accelerated Trajectory Stabilization (Anderson Acceleration)
# ============================================================================

class AndersonAccelerator:
    """
    Type-II Anderson Acceleration for fixed-point iterations.
    
    This accelerator uses historical residuals to compute optimal mixing coefficients
    that minimize the residual norm, achieving quasi-Newton convergence rates.
    
    Reference: Anderson, D.G. (1965). Iterative procedures for nonlinear integral equations.
    """
    
    def __init__(self, history_size=5, regularization=1e-6):
        """
        Initialize the Anderson Accelerator.
        
        Args:
            history_size: Number of previous iterations to use (m in the paper)
            regularization: Regularization parameter for solving the least squares problem
        """
        self.history_size = history_size
        self.regularization = regularization
        self.reset()
    
    def reset(self):
        """Reset the history buffers."""
        self.W_history = []  # Store past W values
        self.G_history = []  # Store past G(W) values (after one iteration)
        self.R_history = []  # Store past residuals R = G(W) - W
    
    def update(self, W_current, G_W_current):
        """
        Compute the accelerated update using Anderson mixing.
        
        Args:
            W_current: Current weight matrix (before iteration)
            G_W_current: Result of one iteration G(W_current) (the LS solution)
        
        Returns:
            W_accelerated: The accelerated weight matrix
        """
        # Compute residual
        R_current = G_W_current - W_current
        
        # Add to history
        self.W_history.append(W_current.clone())
        self.G_history.append(G_W_current.clone())
        self.R_history.append(R_current.clone())
        
        # Limit history size to history_size + 1 elements
        while len(self.R_history) > self.history_size + 1:
            self.W_history.pop(0)
            self.G_history.pop(0)
            self.R_history.pop(0)
        
        m = len(self.R_history) - 1  # Number of historical points to use
        
        if m < 1:
            # Not enough history, just return the LS solution
            return G_W_current
        
        # Build the residual difference matrix
        # Delta_R[:, j] = R_{k-m+j+1} - R_{k-m+j}
        Delta_R = []
        for j in range(m):
            delta = self.R_history[j + 1] - self.R_history[j]
            Delta_R.append(delta.flatten())
        
        # Stack into matrix [n x m]
        Delta_R = torch.stack(Delta_R, dim=1)
        R_k = self.R_history[-1].flatten()
        
        # Solve the least squares problem: min ||R_k - Delta_R @ gamma||^2
        # This gives us the optimal mixing coefficients
        try:
            # Add regularization for numerical stability
            ATA = Delta_R.T @ Delta_R + self.regularization * torch.eye(m, device=Delta_R.device, dtype=Delta_R.dtype)
            ATb = Delta_R.T @ R_k
            gamma = torch.linalg.solve(ATA, ATb)
        except Exception:
            # If solve fails, fall back to simple iteration
            return G_W_current
        
        # Compute the accelerated iterate
        # W_{k+1} = G(W_k) - sum_{j=0}^{m-1} gamma_j * (G(W_{k-m+j+1}) - G(W_{k-m+j}))
        W_accelerated = G_W_current.clone()
        for j in range(m):
            delta_G = self.G_history[j + 1] - self.G_history[j]
            W_accelerated = W_accelerated - gamma[j] * delta_G
        
        return W_accelerated


# ============================================================================
# CAMR: Curvature-Aware Manifold Regularization
# ============================================================================

def compute_camr_regularization(X_tilde_list, alpha, beta=1e-8, sample_weights=None):
    """
    Compute curvature-aware regularization matrix based on input covariance.
    
    This replaces the isotropic regularization alpha*I with a diagonal matrix
    that reflects the geometry of the parameter space based on activation statistics.
    
    Ridge Regression Principle for Closed-Form Solving:
    - High variance directions = good signal = stable inversion = LESS regularization needed
    - Low variance directions = near-singular = need regularization for numerical stability
    
    This is different from EWC/Fisher in SGD fine-tuning where you protect important directions.
    In closed-form solving, we regularize where the data is WEAK, not where it's strong.
    
    Args:
        X_tilde_list: Input features from the merged model [batch, features]
        alpha: Base regularization strength
        beta: Minimum regularization value for numerical stability
        sample_weights: Optional DCS weights for weighted covariance [batch]
    
    Returns:
        Lambda_reg: Diagonal regularization matrix
    """
    with torch.no_grad():
        # Apply sample weights if provided (module coordination with DCS)
        if sample_weights is not None and sample_weights.numel() > 0:
            # Weight the features before computing covariance
            batch_size = sample_weights.shape[0]
            # Ensure tensor has at least 3 dimensions
            if X_tilde_list.dim() >= 3 and X_tilde_list.shape[1] % batch_size == 0:
                features_per_batch = X_tilde_list.shape[1] // batch_size
                feature_dim = X_tilde_list.shape[-1]
                X_reshaped = X_tilde_list.view(X_tilde_list.shape[0], batch_size, features_per_batch, feature_dim)
                sqrt_weights = torch.sqrt(sample_weights).view(1, batch_size, 1, 1)
                X_weighted = X_reshaped * sqrt_weights
                X_tilde_list = X_weighted.view(X_tilde_list.shape[0], -1, feature_dim)
            elif X_tilde_list.dim() < 3:
                # Not enough dimensions for sample weighting
                pass
            else:
                # Dimension mismatch - log warning and skip sample weighting for CAMR
                print(f"[CAMR Warning] Dimension mismatch: X_tilde_list.shape[1]={X_tilde_list.shape[1]} "
                      f"not divisible by batch_size={batch_size}. Skipping DCS-CAMR coordination.")
        
        # Compute covariance of activations: C = X^T X
        # X_tilde_list shape: [layers, batch, features]
        covariance = torch.matmul(X_tilde_list.transpose(-1, -2), X_tilde_list)
        
        # Extract diagonal elements representing variance in each direction
        diag_cov = torch.diagonal(covariance, dim1=-2, dim2=-1)
        
        # Normalize to create relative importance weights
        diag_norm = diag_cov / (diag_cov.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Ridge Regression-aligned regularization for closed-form solving:
        # High variance = strong signal = stable inversion = LESS regularization
        # Low variance = weak signal = near-singular = MORE regularization
        # Lambda_reg = alpha * (1 - normalized_variance) + beta
        Lambda_reg = alpha * (1.0 - diag_norm) + beta
        
    return Lambda_reg


def reg_math_camr(term, lambda_reg):
    """
    Apply curvature-aware regularization to the term matrix.
    
    Args:
        term: The matrix to regularize [batch, d, d]
        lambda_reg: Diagonal regularization values [batch, d]
    
    Returns:
        Regularized matrix
    """
    # Vectorized approach: create diagonal matrices and add to term
    diag_reg = torch.diag_embed(lambda_reg)  # [batch, d, d]
    return term + diag_reg


# ============================================================================
# DCS: Dynamic Conflict-aware Sample Reweighting
# ============================================================================

def compute_sample_weights_dcs(outputs_list, sigma=1.0):
    """
    Compute dynamic sample weights based on cross-model output variance.
    
    Samples with high variance across different LoRA models are considered
    "conflicting" and are down-weighted to reduce interference during merging.
    
    Args:
        outputs_list: List of output tensors from each LoRA model [N, batch, output_dim]
        sigma: Temperature parameter for the Gaussian kernel
    
    Returns:
        weights: Sample weights [batch], higher weight = more consensus
    """
    with torch.no_grad():
        if len(outputs_list) < 2:
            # With only one model, no conflict possible
            return torch.ones(outputs_list[0].shape[0], device=outputs_list[0].device)
        
        # Stack outputs: [N, batch, output_dim]
        stacked = torch.stack(outputs_list, dim=0)
        
        # Compute mean output across models
        mean_output = stacked.mean(dim=0)  # [batch, output_dim]
        
        # Compute variance for each sample across models
        # V_s = (1/N) * sum_{i=1}^{N} ||y_{s,i} - mean_y_s||^2
        diff = stacked - mean_output.unsqueeze(0)  # [N, batch, output_dim]
        variance = (diff ** 2).sum(dim=-1).mean(dim=0)  # [batch]
        
        # Apply Gaussian kernel: w_s = exp(-V_s / sigma^2)
        weights = torch.exp(-variance / (sigma ** 2 + 1e-10))
        
        # Normalize weights to have mean 1 (preserve scale)
        weights = weights / (weights.mean() + 1e-10)
        
        return weights


def compute_feature_variance(X_list):
    """
    Compute variance of input features across different LoRA models.
    
    This is used as a proxy for conflict when we don't have access to output variance.
    
    Args:
        X_list: Input features [N, batch, features] where N is number of LoRAs
    
    Returns:
        variance: Per-sample variance [batch]
    """
    with torch.no_grad():
        # X_list shape: [N, batch, features] where N is number of LoRAs
        # Transpose to [batch, N, features] for easier computation
        X_transposed = X_list.transpose(0, 1)  # [batch, N, features]
        mean_features = X_transposed.mean(dim=1)  # [batch, features]
        diff = X_transposed - mean_features.unsqueeze(1)  # [batch, N, features]
        variance = (diff ** 2).sum(dim=-1).mean(dim=1)  # [batch]
        return variance


def compute_output_variance(W_list, X_list):
    """
    Compute variance of outputs (W @ X) across different LoRA models.
    
    This is a more accurate proxy for gradient conflict than input feature variance,
    as it captures differences in the actual transformations applied by each LoRA.
    
    The variance is computed per-sample (per batch element) by first calculating
    per-position variance, then averaging across all sequence positions. Samples 
    with high cross-model disagreement (high variance) will be down-weighted in
    the DCS algorithm to reduce their influence during merging.
    
    Args:
        W_list: LoRA weight matrices [N, out_dim, in_dim]
        X_list: Input features [N, batch, features] or [N, batch, seq_len, features]
    
    Returns:
        variance: Per-sample output variance with shape [batch].
                  For 4D input, sequence positions are aggregated by mean averaging
                  across the sequence dimension.
    """
    with torch.no_grad():
        N = W_list.shape[0]
        
        # Handle 4D input: [N, batch, seq_len, features]
        # We need to compute per-sample variance, not per-token variance
        has_seq_dim = (X_list.dim() == 4)
        if has_seq_dim:
            batch_size = X_list.shape[1]
            seq_len = X_list.shape[2]
            # Flatten sequence dimension for computation: [N, batch*seq_len, features]
            X_flat = X_list.flatten(start_dim=1, end_dim=2)
        else:
            batch_size = X_list.shape[1]
            seq_len = 1
            X_flat = X_list
        
        # Compute outputs for each LoRA: [N, batch*seq_len, out_dim]
        outputs = torch.matmul(X_flat, W_list.transpose(-1, -2))
        
        # Transpose to [batch*seq_len, N, out_dim]
        outputs_transposed = outputs.transpose(0, 1)
        
        # Compute mean output across LoRAs: [batch*seq_len, out_dim]
        mean_output = outputs_transposed.mean(dim=1)
        
        # Compute per-position variance across LoRAs
        diff = outputs_transposed - mean_output.unsqueeze(1)  # [batch*seq_len, N, out_dim]
        per_position_variance = (diff ** 2).sum(dim=-1).mean(dim=1)  # [batch*seq_len]
        
        # Aggregate to per-sample variance by averaging over sequence positions
        if has_seq_dim:
            # Reshape to [batch, seq_len] and average over seq_len
            per_position_variance = per_position_variance.view(batch_size, seq_len)
            variance = per_position_variance.mean(dim=1)  # [batch]
        else:
            variance = per_position_variance  # [batch]
        
        return variance


def compute_adaptive_sigma(variance, scale_factor=1.0):
    """
    Compute adaptive sigma based on variance distribution.
    
    This avoids the need for manual tuning of the sigma parameter by
    adapting it to the actual variance distribution of the data.
    
    Args:
        variance: Per-sample variance tensor [batch]
        scale_factor: Scaling factor for the adaptive sigma
    
    Returns:
        sigma: Adaptive sigma value
    """
    with torch.no_grad():
        # Use median absolute deviation (more robust than std)
        median_var = torch.median(variance)
        mad = torch.median(torch.abs(variance - median_var))
        # Scale factor to approximate std from MAD
        sigma = (mad * 1.4826 + 1e-8) * scale_factor
        return sigma.item()


# ============================================================================
# IterIS++ Core Algorithm
# ============================================================================

def reg_math(term, alpha):
    """Original isotropic regularization (kept for compatibility)."""
    term_list = [term[i] + alpha[i] * torch.eye(term[i].size(0), dtype=term.dtype, device=term.device) 
                 for i in range(term.shape[0])]
    return torch.stack(term_list)


def solution_matrix_plus(
    W_list,
    X_list,
    X_tilde_list,
    ceof_list,
    manual_ceof,
    alpha_1=1e-7,
    alpha_2=1e-7,
    reg_ceof=5e-4,
    sample_weights=None,
    use_camr=True,
    camr_alpha=1e-7,
    camr_beta=1e-8,
):
    """
    Enhanced solution matrix computation with Auto-Adaptive N support.
    
    This is the core "Solving" step of IterIS++, enhanced with:
    - Auto-correction for coefficient dimensions
    - Type safety for manual_ceof (int -> float)
    - Dynamic expansion of results based on task count N
    """
    with torch.no_grad():
        # 1. 获取当前实际的任务数量 N
        N = W_list.shape[0]

        # ==========================================
        # 🛠️ 核心修改：自动适配 manual_ceof + 强制转浮点
        # ==========================================
        # 即使 config 里写错了长度或者写成了整数，这里都会自动修正
        if isinstance(manual_ceof, list):
            if len(manual_ceof) != N:
                # print(f"[Auto-Fix] manual_ceof length mismatch. Resetting to all 1.0 for {N} tasks.")
                manual_ceof = torch.ones(N).to('cuda')
            else:
                # 关键：加上 .float()，兼容配置文件里写整数 [1, 1, ...] 的情况
                manual_ceof = torch.tensor(manual_ceof).float().to('cuda')
        elif isinstance(manual_ceof, torch.Tensor):
             if manual_ceof.shape[0] != N:
                manual_ceof = torch.ones(N).to('cuda')
             else:
                manual_ceof = manual_ceof.float().to('cuda')
        else:
            # 兜底：如果传入的是标量或其他情况
            manual_ceof = torch.ones(N).to('cuda')
        # ==========================================

        X_list, X_tilde_list = X_list.transpose(0, 1).flatten(start_dim=1, end_dim=2), \
                               X_tilde_list.transpose(0, 1).flatten(start_dim=1, end_dim=2)

        X_tilde_list = (1 - reg_ceof) * X_tilde_list + reg_ceof * X_list
        
        # Apply DCS sample weights if provided
        if sample_weights is not None and sample_weights.numel() > 0:
            batch_size = sample_weights.shape[0]
            sqrt_weights = torch.sqrt(sample_weights)
            
            features_per_batch = X_list.shape[1] // batch_size
            if features_per_batch > 0 and X_list.shape[1] % batch_size == 0:
                X_list_reshaped = X_list.view(N, batch_size, features_per_batch, -1)
                X_tilde_reshaped = X_tilde_list.view(N, batch_size, features_per_batch, -1)
                
                # Broadcasting weights
                X_list_reshaped = X_list_reshaped * sqrt_weights.view(1, batch_size, 1, 1)
                X_tilde_reshaped = X_tilde_reshaped * sqrt_weights.view(1, batch_size, 1, 1)
                
                X_list = X_list_reshaped.view(N, -1, X_list_reshaped.shape[-1])
                X_tilde_list = X_tilde_reshaped.view(N, -1, X_tilde_reshaped.shape[-1])
        
        X_X_tilde = torch.matmul(X_list.transpose(-1, -2), X_tilde_list)
        X_X_tilde_norm = torch.norm(X_X_tilde, p='fro', dim=[-2, -1]) * alpha_1
        X_X_tilde = reg_math(X_X_tilde, X_X_tilde_norm)

        X_tilde_X_tilde = torch.matmul(X_tilde_list.transpose(-1, -2), X_tilde_list)
        
        # Apply CAMR regularization if enabled
        # 注意：请确保你的环境里有 reg_math_camr 和 compute_camr_regularization 函数
        # 如果没有，请将下面 if 块替换为 else 块的内容
        if use_camr:
            try:
                Lambda_reg = compute_camr_regularization(X_tilde_list, camr_alpha, camr_beta, sample_weights)
                X_tilde_X_tilde = reg_math_camr(X_tilde_X_tilde, Lambda_reg)
            except NameError:
                # 如果找不到 CAMR 函数，自动回退到普通正则化
                X_tilde_X_tilde_norm = torch.norm(X_tilde_X_tilde, p='fro', dim=[-2, -1]) * alpha_2
                X_tilde_X_tilde = reg_math(X_tilde_X_tilde, X_tilde_X_tilde_norm)
        else:
            X_tilde_X_tilde_norm = torch.norm(X_tilde_X_tilde, p='fro', dim=[-2, -1]) * alpha_2
            X_tilde_X_tilde = reg_math(X_tilde_X_tilde, X_tilde_X_tilde_norm)

        # 这里的 .view(N, 1, 1) 现在绝对安全了
        term1 = torch.sum(torch.matmul(W_list, X_X_tilde) * (ceof_list * manual_ceof).view(N, 1, 1), dim=0).double()
        term2 = torch.sum(X_tilde_X_tilde * (ceof_list * manual_ceof).view(N, 1, 1), dim=0).double()
        results = torch.linalg.solve(term2.t(), term1.t()).double().t()

        return results.to('cpu')


def update_param_plus(
    seed,
    max_iter,
    lora_path,
    model_name,
    task_targets,
    manual_ceof,
    shuffle,
    with_pretrain_matrix=0,
    max_length=512,
    lora_alpha=[32, 32],
    alpha_1=1e-7,
    alpha_2=1e-7,
    reg_ceof=5e-4,
    rank=8,
    select_long=40,
    inner_num=2,
    outer_num=10,
    samples_num=20,
    if_divide=True,
    if_balance=True,
    # IterIS++ specific parameters
    use_mats=True,
    mats_history_size=5,
    mats_regularization=1e-6,
    use_camr=True,
    camr_alpha=1e-7,
    camr_beta=1e-8,
    use_dcs=True,
    dcs_sigma=1.0,
    convergence_threshold=1e-6,  # For early stopping
    **generation_kwargs,
):
    """
    IterIS++ main update function with MATS, CAMR, and DCS innovations.
    
    This function implements the complete IterIS++ algorithm:
    1. Initialization: Average of LoRA weights
    2. Preparation: Compute CAMR regularization matrix
    3. Iteration Loop:
       - Inference: Get features from current merged model
       - DCS: Compute sample weights based on output variance
       - Solving: Weighted least squares with CAMR regularization
       - MATS: Anderson acceleration on the weight updates
       - Convergence check for early stopping
    
    Args:
        seed: Random seed
        max_iter: Maximum number of iterations
        lora_path: Paths to LoRA adapters
        model_name: Base model name
        task_targets: List of task names
        manual_ceof: Manual weighting coefficients
        shuffle: Whether to shuffle data
        use_mats: Enable MATS (Anderson Acceleration)
        mats_history_size: History depth for Anderson Acceleration
        mats_regularization: Regularization for MATS least squares
        use_camr: Enable CAMR (Curvature-Aware Regularization)
        camr_alpha: CAMR regularization strength
        camr_beta: CAMR minimum regularization
        use_dcs: Enable DCS (Dynamic Sample Weighting)
        dcs_sigma: Scale factor for adaptive DCS sigma
        convergence_threshold: Threshold for early stopping based on weight change
        **generation_kwargs: Additional generation arguments
    
    Returns:
        Merged model with optimized weights
    """
    print("=" * 60)
    print("IterIS++ Algorithm Starting")
    print("=" * 60)
    print(f"Innovations enabled:")
    print(f"  - MATS (Anderson Acceleration): {use_mats}")
    print(f"  - CAMR (Curvature-Aware Reg.): {use_camr}")
    print(f"  - DCS (Dynamic Sample Weight): {use_dcs}")
    print(f"  - Convergence threshold: {convergence_threshold}")
    print("=" * 60)
    
    # Get all mid-features from each LoRA model
    input_ids_list, X_dict = get_all_midfeatures(
        rank=rank,
        seed=seed,
        select_long=select_long,
        lora_path=lora_path,
        model_name=model_name,
        max_length=max_length,
        task_targets=task_targets,
        if_divide=if_divide,
        if_balance=if_balance,
        shuffle=shuffle,
        inner_num=inner_num,
        outer_num=outer_num,
        samples_num=samples_num,
        **generation_kwargs,
    )

    pretrain_matrix_dict = get_pretrain_matrix(X_dict.keys(), model_name=model_name)

    lora_adapter_path_list = [
        lora_adapter_path + "/adapter_model.safetensors" for lora_adapter_path in lora_path
    ]
    tensors_lora = [safe_open(tensor_lora, framework='pt') for tensor_lora in lora_adapter_path_list]
    torch.cuda.empty_cache()
    
    X_tilde_dict = {}
    
    # Initialize Anderson Accelerators for each layer (MATS)
    anderson_accelerators = {}
    if use_mats:
        for idx in X_dict.keys():
            anderson_accelerators[idx] = AndersonAccelerator(
                history_size=mats_history_size,
                regularization=mats_regularization
            )
    
    # Previous iteration weights for MATS
    prev_tar_lora_list = {}
    
    for iteration in range(max_iter):
        torch.cuda.empty_cache()
        gc.collect()
        tar_lora_list = {}
        print(f"\n-----------IterIS++ Iteration: {iteration + 1}/{max_iter}---------------")
        print("Computing optimal solution with IterIS++ enhancements...")
        
        with torch.no_grad():
            for idx in X_dict.keys():
                W_list, X_list = torch.stack(
                    [get_lora_matrix(model_name, tensors_lora[i], idx, lora_alpha[i], rank=rank, no_weight=True) 
                     for i in range(len(tensors_lora))]
                ).to('cuda'), X_dict[idx].to('cuda')
                
                N = W_list.shape[0]
                merge_W = W_list + pretrain_matrix_dict[idx].unsqueeze(0).repeat(N, 1, 1).to('cuda')
                ceof_list = torch.norm(merge_W, p='fro', dim=[-2, -1]) ** 2 / \
                            torch.sum(torch.norm(torch.matmul(X_list, merge_W.transpose(1, 2)), p='fro', dim=[-2, -1]) ** 2, dim=0)
                
                # DCS: Compute sample weights based on output variance
                # Using output variance (W @ X) provides a more accurate proxy for gradient conflict
                # than input feature variance alone
                sample_weights = None
                if use_dcs:
                    # Compute output variance (per-sample, aggregated over sequence positions)
                    output_variance = compute_output_variance(W_list, X_list.transpose(0, 1))
                    
                    # Log DCS statistics on first layer of first iteration for debugging
                    first_layer_key = next(iter(X_dict.keys()))
                    if iteration == 0 and idx == first_layer_key:
                        logging.info(f"[DCS] Sample variance stats - min: {output_variance.min().item():.4e}, "
                              f"max: {output_variance.max().item():.4e}, "
                              f"mean: {output_variance.mean().item():.4e}, "
                              f"shape: {output_variance.shape}")
                    
                    # Use adaptive sigma based on variance distribution
                    effective_sigma = compute_adaptive_sigma(output_variance, scale_factor=dcs_sigma)
                    
                    # Compute sample weights using Gaussian kernel
                    # Higher variance (cross-model disagreement) → lower weight (down-weighted)
                    sample_weights = torch.exp(-output_variance / (effective_sigma ** 2 + 1e-10))
                    sample_weights = sample_weights / (sample_weights.mean() + 1e-10)
                    
                    # Log DCS weight statistics on first layer of first iteration
                    if iteration == 0 and idx == first_layer_key:
                        logging.info(f"[DCS] Effective sigma: {effective_sigma:.4e}")
                        logging.info(f"[DCS] Sample weights stats - min: {sample_weights.min().item():.4f}, "
                              f"max: {sample_weights.max().item():.4f}, "
                              f"std: {sample_weights.std().item():.4f}")
                
                X_tilde = X_list if iteration == 0 else X_tilde_dict[idx].to('cuda')
                
                if with_pretrain_matrix == 0:
                    W_ls = solution_matrix_plus(
                        W_list, X_list, X_tilde, ceof_list, manual_ceof,
                        alpha_1, alpha_2, reg_ceof,
                        sample_weights=sample_weights,
                        use_camr=use_camr,
                        camr_alpha=camr_alpha,
                        camr_beta=camr_beta,
                    )
                elif with_pretrain_matrix == 1:
                    W_ls = solution_matrix_plus(
                        merge_W, X_list, X_tilde, ceof_list, manual_ceof,
                        alpha_1, alpha_2, reg_ceof,
                        sample_weights=sample_weights,
                        use_camr=use_camr,
                        camr_alpha=camr_alpha,
                        camr_beta=camr_beta,
                    )
                
                # MATS: Apply Anderson Acceleration
                if use_mats and idx in prev_tar_lora_list:
                    W_current = prev_tar_lora_list[idx].to('cuda')
                    W_ls_cuda = W_ls.to('cuda')
                    W_accelerated = anderson_accelerators[idx].update(W_current, W_ls_cuda)
                    tar_lora_list[idx] = W_accelerated.to('cpu')
                else:
                    tar_lora_list[idx] = W_ls.to('cpu')
                
                torch.cuda.empty_cache()
                gc.collect()
        
        # Convergence check: compute total weight change (before storing new weights)
        converged = False
        if iteration > 0 and convergence_threshold > 0 and prev_tar_lora_list:
            total_change = 0.0
            total_norm = 0.0
            for k, v in tar_lora_list.items():
                if k in prev_tar_lora_list:
                    total_change += torch.norm(v - prev_tar_lora_list[k]).item()
                    total_norm += torch.norm(v).item()
            relative_change = total_change / (total_norm + 1e-10)
            print(f"Weight relative change: {relative_change:.2e}")
            if relative_change < convergence_threshold:
                print(f"✓ Converged at iteration {iteration + 1} (relative change {relative_change:.2e} < {convergence_threshold})")
                converged = True
        
        # Store current weights for next iteration's MATS (after convergence check)
        prev_tar_lora_list = {k: v.clone() for k, v in tar_lora_list.items()}
        
        print("Calculation Done!")
        print("Loading and updating the merged model...")
        
        model = None
        if 't5' in model_name:
            model = T5WithHooks.from_pretrained(model_name, lora_path=lora_path[0] + '/adapter_model.safetensors').to('cuda')
        elif 'bart' in model_name:
            model = BartWithHooks.from_pretrained(model_name, lora_path=lora_path[0] + '/adapter_model.safetensors').to('cuda')
        elif 'blip' in model_name:
            model = BlipWithHook.from_pretrained(model_name).to('cuda')
        
        # Update model with computed weights
        number_update = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name[:-7] in tar_lora_list.keys():
                    lora_matrix = tar_lora_list[name[:-7]].to('cuda')
                    if with_pretrain_matrix == 0:
                        param.copy_(lora_matrix + param)
                    elif with_pretrain_matrix == 1:
                        param.copy_(lora_matrix)
                    number_update += 1
        
        if number_update == len(tar_lora_list.keys()):
            print("All LoRA targets updated successfully!")
        else:
            print("Warning: Some targets were not updated.")
        
        torch.cuda.empty_cache()
        max_memory = torch.cuda.max_memory_allocated()
        print(f"Max memory usage: {max_memory / 1024 ** 2:.2f} MB", flush=True)
        
        # Check for convergence or max iterations reached
        if iteration == max_iter - 1 or converged:
            print("\n" + "=" * 60)
            if converged:
                print(f"IterIS++ Algorithm Complete (Early Stopped at iteration {iteration + 1})")
            else:
                print("IterIS++ Algorithm Complete")
            print("=" * 60)
            return model
        
        # Record mid-features of updated model for next iteration
        records_list = []
        if if_divide:
            assert inner_num * outer_num == len(input_ids_list[0])
            for input_ids in input_ids_list:
                print("Generating merged model midfeatures...")
                dict_record_item = {}
                for i in range(outer_num):
                    with torch.no_grad():
                        outputs = model.generate(input_ids[i * inner_num:(i + 1) * inner_num, :].to('cuda'))
                    temp_dict = dict(model.inputs_to_track.items())
                    dict_record_item = temp_dict if i == 0 else {
                        key: torch.cat([value, temp_dict[key]], dim=0) 
                        for key, value in dict_record_item.items()
                    }
                    model.inputs_to_track.clear()
                    torch.cuda.empty_cache()
                records_list.append(dict_record_item)
        else:
            for input_ids in input_ids_list:
                model.inputs_to_track.clear()
                torch.cuda.empty_cache()
                print("Generating merged model midfeatures...")
                with torch.no_grad():
                    if 'blip' in model_name:
                        outputs = model.generate(**input_ids, max_length=max_length)
                    else:
                        outputs = model.generate(input_ids.to('cuda'))
                records_list.append(dict(model.inputs_to_track.items()))

        for item in records_list[0].keys():
            X_tilde_dict[item] = torch.cat(
                [records[item].unsqueeze(dim=1) for records in records_list],
                dim=1,
            ).to('cpu')


def generate_output_filename(task_type, use_mats, use_camr, use_dcs):
    """Generate unique filename based on task type and activation flags."""
    output_dir = f'outputs_{task_type.lower()}'
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create base filename
    mats_str = 'MATS1' if use_mats else 'MATS0'
    camr_str = 'CAMR1' if use_camr else 'CAMR0'
    dcs_str = 'DCS1' if use_dcs else 'DCS0'
    base_name = f'{mats_str}_{camr_str}_{dcs_str}'
    
    # Check for existing files and add number if needed
    counter = 0
    while True:
        if counter == 0:
            filename = f'{output_dir}/{base_name}.txt'
        else:
            filename = f'{output_dir}/{base_name}_{counter}.txt'
        
        if not os.path.exists(filename):
            return filename
        counter += 1


def format_results_table(task_type, task_targets, eval_results, config_data, 
                        use_mats, use_camr, use_dcs, elapsed_time):
    """Format evaluation results into a nice table."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"IterIS++ EXPERIMENT RESULTS - {task_type}")
    lines.append("=" * 80)
    lines.append("")
    
    # Configuration section
    lines.append("CONFIGURATION:")
    lines.append("-" * 80)
    lines.append(f"Task Type:              {task_type}")
    lines.append(f"Model:                  {config_data[task_type]['model_name']}")
    lines.append(f"Tasks:                  {', '.join(task_targets)}")
    lines.append(f"Max Iterations:         {config_data[task_type]['max_iter']}")
    lines.append(f"LoRA Rank:              {config_data[task_type]['rank']}")
    lines.append(f"LoRA Alpha:             {config_data[task_type]['lora_alpha']}")
    lines.append(f"Alpha 1 (reg):          {config_data[task_type]['alpha_1']}")
    lines.append(f"Alpha 2 (reg):          {config_data[task_type]['alpha_2']}")
    lines.append(f"Sample Numbers:         {config_data[task_type]['samples_num']}")
    lines.append("")
    lines.append("IterIS++ Innovations:")
    lines.append(f"  - MATS (Anderson Acceleration):    {'✓ ENABLED' if use_mats else '✗ DISABLED'}")
    lines.append(f"  - CAMR (Curvature-Aware Reg):      {'✓ ENABLED' if use_camr else '✗ DISABLED'}")
    lines.append(f"  - DCS (Dynamic Sample Weight):      {'✓ ENABLED' if use_dcs else '✗ DISABLED'}")
    if use_mats:
        lines.append(f"    - MATS history size:             {config_data[task_type].get('mats_history_size', 5)}")
        lines.append(f"    - MATS regularization:           {config_data[task_type].get('mats_regularization', 1e-6)}")
    if use_camr:
        lines.append(f"    - CAMR alpha:                    {config_data[task_type].get('camr_alpha', config_data[task_type]['alpha_1'])}")
        lines.append(f"    - CAMR beta:                     {config_data[task_type].get('camr_beta', 1e-8)}")
    if use_dcs:
        lines.append(f"    - DCS sigma:                     {config_data[task_type].get('dcs_sigma', 1.0)}")
    lines.append(f"  - Convergence threshold:           {config_data[task_type].get('convergence_threshold', 1e-6)}")
    lines.append("")
    lines.append(f"Total Training Time:    {elapsed_time:.2f} seconds")
    lines.append("")
    
    # Results section
    lines.append("EVALUATION RESULTS:")
    lines.append("=" * 80)
    
    if task_type == 'TASKS_blip_base':
        # Vision & Language task format
        lines.append("Method                Acc(pos, neg)     CIDEr    B-1    B-2    B-3    B-4")
        lines.append("-" * 80)
        
        # Parse results for each task
        for task in task_targets:
            if task in eval_results:
                result = eval_results[task]
                lines.append(f"{task:20}  {result}")
        
        lines.append("-" * 80)
        lines.append("IterIS++ (This Run)   [Results from evaluation above]")
        
    elif task_type == 'GLUE_t5':
        # GLUE task format
        lines.append("Task      Metric    Score")
        lines.append("-" * 80)
        
        for task in task_targets:
            if task in eval_results:
                result = eval_results[task]
                lines.append(f"{task:10} {result}")
        
        lines.append("-" * 80)
        
    elif task_type == 'EMOTION_t5_large':
        # Emotion task format
        lines.append("Task             Metric    Score")
        lines.append("-" * 80)
        
        for task in task_targets:
            if task in eval_results:
                result = eval_results[task]
                lines.append(f"{task:15}  {result}")
        
        lines.append("-" * 80)
    
    lines.append("=" * 80)
    lines.append("")
    lines.append("* Best performance indicated where applicable")
    lines.append("")
    
    return '\n'.join(lines)



def format_blip_results(task_name, results):
    """Format BLIP evaluation results into a nice table"""
    lines = []
    lines.append(f"Task: {task_name}")
    lines.append("-" * 80)
    lines.append("")
    
    # Image Captioning Metrics
    lines.append("Image Captioning Metrics:")
    lines.append("  BLEU-1:        {:.4f}".format(results['bleu'][0]))
    lines.append("  BLEU-2:        {:.4f}".format(results['bleu'][1]))
    lines.append("  BLEU-3:        {:.4f}".format(results['bleu'][2]))
    lines.append("  BLEU-4:        {:.4f}".format(results['bleu'][3]))
    lines.append("  CIDEr:         {:.4f}".format(results['cider']))
    lines.append("  ROUGE-L:       {:.4f}".format(results['rougeL']))
    lines.append("")
    
    # Classification Metrics
    lines.append("Classification:")
    lines.append("  Accuracy:      {:.4f}".format(results['acc']))
    lines.append("")
    
    # Diversity Metrics
    lines.append("Diversity:")
    lines.append("  Div-1:         {:.4f}".format(results['div_1']))
    lines.append("  Div-2:         {:.4f}".format(results['div_2']))
    lines.append("  Vocab Size:    {}".format(results['vocab_size']))
    lines.append("")
    
    return "\n".join(lines)


def format_glue_results(task_name, results):
    """Format GLUE evaluation results into a nice table"""
    lines = []
    lines.append(f"Task: {task_name.upper()}")
    lines.append("-" * 80)
    
    # 根据任务类型显示相应的指标
    if 'acc' in results:
        lines.append(f"  Accuracy:      {results['acc']:.4f}")
    if 'f1' in results:
        lines.append(f"  F1 Score:      {results['f1']:.4f}")
    if 'mcc' in results:
        lines.append(f"  Matthews Corr: {results['mcc']:.4f}")
    if 'acc_and_f1' in results:
        lines.append(f"  Acc & F1:      {results['acc_and_f1']:.4f}")
    
    lines.append("")
    return "\n".join(lines)


def format_emotion_results(task_name, results):
    """Format emotion classification results into a nice table"""
    lines = []
    lines.append(f"Task: {task_name}")
    lines.append("-" * 80)
    
    if 'f1' in results:
        lines.append(f"  F1 Score:      {results['f1']:.4f}")
    if 'acc' in results:
        lines.append(f"  Accuracy:      {results['acc']:.4f}")
    if 'precision' in results:
        lines.append(f"  Precision:     {results['precision']:.4f}")
    if 'recall' in results:
        lines.append(f"  Recall:        {results['recall']:.4f}")
    
    lines.append("")
    return "\n".join(lines)

def main():
    """Main entry point for IterIS++."""


    parser = argparse.ArgumentParser(description="IterIS++ Training Script")
    parser.add_argument('--config', type=str, default="config/methods-config/iteris-plus-config.yaml",
                        help="Path to the config file")
    parser.add_argument('--task_type', type=str, 
                        choices=['GLUE_t5', 'EMOTION_t5_large', 'TASKS_blip_base'],
                        default='GLUE_t5', help="Choose a task type")
    
    # IterIS++ specific arguments (can override config)
    parser.add_argument('--use_mats', type=int, default=None, help="Enable MATS (0 or 1)")
    parser.add_argument('--use_camr', type=int, default=None, help="Enable CAMR (0 or 1)")
    parser.add_argument('--use_dcs', type=int, default=None, help="Enable DCS (0 or 1)")
    
    args = parser.parse_args()
    task_type = args.task_type
    
    # Load configuration
    with open(args.config, 'r') as file:
        config_data = yaml.safe_load(file)
    
    set_seed(config_data['seed'])
    model_name = config_data[task_type]['model_name']
    task_targets = config_data[task_type]['task_targets']
    lora_path = [get_loras_path(task_type, model_name)[item] for item in task_targets]
    with_pretrain_matrix = config_data[task_type]['with_pretrain_matrix']
    tokenizer = AutoTokenizer.from_pretrained(model_name) if 'blip' not in model_name \
                else AutoProcessor.from_pretrained(model_name)
    save = config_data[task_type].get('save', 0)
    
    # Get IterIS++ specific parameters
    use_mats = config_data[task_type].get('use_mats', True)
    use_camr = config_data[task_type].get('use_camr', True)
    use_dcs = config_data[task_type].get('use_dcs', True)
    
    # Command line overrides
    if args.use_mats is not None:
        use_mats = bool(args.use_mats)
    if args.use_camr is not None:
        use_camr = bool(args.use_camr)
    if args.use_dcs is not None:
        use_dcs = bool(args.use_dcs)

    # Run IterIS++ algorithm
    start_time = time.time()
    model = update_param_plus(
        task_targets=task_targets,
        lora_path=lora_path,
        model_name=model_name,
        with_pretrain_matrix=with_pretrain_matrix,
        max_iter=config_data[task_type]['max_iter'],
        max_length=config_data[task_type]['max_length'],
        lora_alpha=config_data[task_type]['lora_alpha'],
        alpha_1=config_data[task_type]['alpha_1'],
        alpha_2=config_data[task_type]['alpha_2'],
        reg_ceof=config_data[task_type]['reg_ceof'],
        rank=config_data[task_type]['rank'],
        samples_num=config_data[task_type]['samples_num'],
        manual_ceof=config_data[task_type]['manual_ceof'],
        if_divide=config_data[task_type]['if_divide'],
        if_balance=config_data[task_type]['if_balance'],
        inner_num=config_data[task_type]['inner_num'],
        outer_num=config_data[task_type]['outer_num'],
        seed=config_data['seed'],
        select_long=config_data[task_type]['select_long'],
        shuffle=config_data[task_type]['shuffle'],
        # IterIS++ specific parameters
        use_mats=use_mats,
        mats_history_size=int(config_data[task_type].get('mats_history_size', 5)),
        mats_regularization=float(config_data[task_type].get('mats_regularization', 1e-6)),
        use_camr=use_camr,
        camr_alpha=float(config_data[task_type].get('camr_alpha', config_data[task_type]['alpha_1'])),
        camr_beta=float(config_data[task_type].get('camr_beta', 1e-8)),
        use_dcs=use_dcs,
        dcs_sigma=float(config_data[task_type].get('dcs_sigma', 1.0)),
        convergence_threshold=float(config_data[task_type].get('convergence_threshold', 1e-6)),
    )
    
    if save == 1:
        torch.save(model, "merged_model/merged_model_plus.pth")
        print("Model saved to merged_model/merged_model_plus.pth")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal time: {elapsed_time:.2f} seconds")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()

    # ========================================================================
    # Collect and save evaluation results
    # ========================================================================
    
    # Collect evaluation metrics
    eval_results = {}
    original_eval_fn = eval_iteris_model.__wrapped__ if hasattr(eval_iteris_model, '__wrapped__') else None
    
    # Generate output filename

    
    output_filename = generate_output_filename(task_type, use_mats, use_camr, use_dcs)

    
    

    
    # Collect evaluation results

    
    eval_results = {}

    
    

    
    # Model evaluation
    # ========================================================================
    # Save evaluation results to file (First save config and open file)
    # ========================================================================
    output_file = generate_output_filename(task_type, use_mats, use_camr, use_dcs)
    
    # Save configuration first
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("IterIS++ EXPERIMENT RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Configuration
        f.write("CONFIGURATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Task Type: {task_type}\n")
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Task Targets: {task_targets}\n")
        f.write(f"Max Iterations: {config_data[task_type]['max_iter']}\n")
        f.write(f"LoRA Rank: {config_data[task_type]['rank']}\n")
        f.write(f"LoRA Alpha: {config_data[task_type]['lora_alpha']}\n")
        f.write(f"Samples Number: {config_data[task_type]['samples_num']}\n")
        f.write(f"Alpha 1 (reg): {config_data[task_type]['alpha_1']}\n")
        f.write(f"Alpha 2 (reg): {config_data[task_type]['alpha_2']}\n")
        f.write(f"\nIterIS++ Innovations:\n")
        f.write(f"  - MATS (Anderson Acceleration):    {'✓ ENABLED' if use_mats else '✗ DISABLED'}\n")
        f.write(f"  - CAMR (Curvature-Aware Reg):      {'✓ ENABLED' if use_camr else '✗ DISABLED'}\n")
        f.write(f"  - DCS (Dynamic Sample Weight):      {'✓ ENABLED' if use_dcs else '✗ DISABLED'}\n")
        if use_mats:
            f.write(f"    - MATS history size:             {int(config_data[task_type].get('mats_history_size', 5))}\n")
            f.write(f"    - MATS regularization:           {float(config_data[task_type].get('mats_regularization', 1e-6))}\n")
        if use_camr:
            f.write(f"    - CAMR alpha:                    {float(config_data[task_type].get('camr_alpha', config_data[task_type]['alpha_1']))}\n")
            f.write(f"    - CAMR beta:                     {float(config_data[task_type].get('camr_beta', 1e-8))}\n")
        if use_dcs:
            f.write(f"    - DCS sigma:                     {float(config_data[task_type].get('dcs_sigma', 1.0))}\n")
        f.write(f"  - Convergence threshold:           {float(config_data[task_type].get('convergence_threshold', 1e-6))}\n")
        f.write(f"\nTotal Training Time: {elapsed_time:.2f} seconds\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
    
    # Model evaluation and append results to file
    print(f"\n{'='*80}")
    print(f"Running evaluation and saving results to: {output_file}")
    print(f"{'='*80}\n")
    
    for task_name in task_targets:
        print(f"\n--- Evaluating {task_name} ---")
        
        # Capture evaluation output
        import sys, re
        from io import StringIO
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            eval_iteris_model(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                task_name=task_name,
                max_length=config_data[task_type]['max_length'],
                per_device_eval_batch_size=config_data[task_type]['per_device_eval_batch_size'],
            )
            captured = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        
        # Print captured output to console
        print(captured)
        
        # Parse results from captured output
        try:
            # Extract the results dictionary
            result_match = re.search(r"Eval results[-]*\n({.*?})", captured, re.DOTALL)
            if result_match:
                results_str = result_match.group(1)
                results = eval(results_str)
                
                # Format results based on task type
                if task_type == "TASKS_blip_base":
                    formatted = format_blip_results(task_name, results)
                elif task_type == "GLUE_t5":
                    formatted = format_glue_results(task_name, results)
                elif task_type == "EMOTION_t5_large":
                    formatted = format_emotion_results(task_name, results)
                else:
                    formatted = f"Task: {task_name}\n" + "-"*80 + "\n" + captured
                
                # Save formatted results to file
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(formatted)
                    f.write("\n")
            else:
                # Fallback: save raw output
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(f"Task: {task_name}\n")
                    f.write("-" * 80 + "\n")
                    f.write(captured)
                    f.write("\n")
        except Exception as e:
            print(f"Warning: Failed to parse results for {task_name}: {e}")
            # Save raw output as fallback
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"Task: {task_name}\n")
                f.write("-" * 80 + "\n")
                f.write(captured)
                f.write("\n")

    # Add footer to file
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("* Best performance indicated where applicable\n")
        f.write("=" * 80 + "\n")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
    
    print(f"\n{'='*80}")
    print(f"✓ All results saved to: {output_file}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()


# """
# IterIS++: Enhanced LoRA Merging with MATS, CAMR, and DCS

# This module implements the IterIS++ algorithm, an enhanced version of IterIS with:
# - MATS (Momentum Accelerated Trajectory Stabilization): Anderson Acceleration for faster convergence
# - CAMR (Curvature-Aware Manifold Regularization): Geometry-aware regularization
# - DCS (Dynamic Conflict-aware Sample Reweighting): Sample weighting based on cross-model variance

# Reference: IterIS: Iterative Inference-Solving Alignment for LoRA Merging (CVPR 2025)
# """

# import gc
# import yaml
# import time
# import torch
# import random
# import argparse
# import numpy as np
# import torch.nn as nn
# from datasets import load_dataset
# from safetensors import safe_open
# from sklearn.metrics import f1_score
# from eval_model import eval_iteris_model
# from get_midfeatures import T5WithHooks, BartWithHooks, BlipWithHook
# from transformers import AutoTokenizer, AutoProcessor
# from get_midfeatures import get_all_midfeatures, get_samples, get_pretrain_matrix, get_lora_matrix

# GLUE_task_name = [
#     "mnli", "rte",
#     "cola", "sst2", "qqp",
#     "qnli", "mrpc",
# ]
# EMOTION_task_name = [
#     "emoint", "emotion-cause",
#     "tec", "isear",
# ]
# SENTICAP_task_name = ['positive', 'negative']
# FlickrStyle10k_task_name = ["roman", "humor"]
# TASKS_blip_base = ['positive', 'negative', "roman", "humor"]


# def get_loras_path(task_type, model_name):
#     """Get paths to LoRA adapters based on task type and model name."""
#     lora_path_dict = {}
#     if 't5' in model_name and task_type == "GLUE_t5":
#         for item in GLUE_task_name:
#             lora_path_dict[item] = f"loras/GLUE-lora-t5/{item}"
#     elif 'bart' in model_name and task_type == "GLUE_bart":
#         for item in GLUE_task_name:
#             lora_path_dict[item] = f"loras/GLUE-lora-bart/{item}"
#     elif 't5-large' in model_name and task_type == "EMOTION_t5_large":
#         for item in EMOTION_task_name:
#             lora_path_dict[item] = f"loras/EMOTION-lora-t5/{item}"
#     elif 'blip' in model_name and task_type == "TASKS_blip_base":
#         for item in FlickrStyle10k_task_name:
#             lora_path_dict[item] = f"loras/FlickrStyle10k-lora-blip/{item}"
#         for item in SENTICAP_task_name:
#             lora_path_dict[item] = f"loras/SENTICAP-lora-blip/{item}"
#     return lora_path_dict


# def set_seed(seed):
#     """Set all random seeds for reproducibility."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# # ============================================================================
# # MATS: Momentum Accelerated Trajectory Stabilization (Anderson Acceleration)
# # ============================================================================

# class AndersonAccelerator:
#     """
#     Type-II Anderson Acceleration for fixed-point iterations.
    
#     This accelerator uses historical residuals to compute optimal mixing coefficients
#     that minimize the residual norm, achieving quasi-Newton convergence rates.
    
#     Reference: Anderson, D.G. (1965). Iterative procedures for nonlinear integral equations.
#     """
    
#     def __init__(self, history_size=5, regularization=1e-6):
#         """
#         Initialize the Anderson Accelerator.
        
#         Args:
#             history_size: Number of previous iterations to use (m in the paper)
#             regularization: Regularization parameter for solving the least squares problem
#         """
#         self.history_size = history_size
#         self.regularization = regularization
#         self.reset()
    
#     def reset(self):
#         """Reset the history buffers."""
#         self.W_history = []  # Store past W values
#         self.G_history = []  # Store past G(W) values (after one iteration)
#         self.R_history = []  # Store past residuals R = G(W) - W
    
#     def update(self, W_current, G_W_current):
#         """
#         Compute the accelerated update using Anderson mixing.
        
#         Args:
#             W_current: Current weight matrix (before iteration)
#             G_W_current: Result of one iteration G(W_current) (the LS solution)
        
#         Returns:
#             W_accelerated: The accelerated weight matrix
#         """
#         # Compute residual
#         R_current = G_W_current - W_current
        
#         # Add to history
#         self.W_history.append(W_current.clone())
#         self.G_history.append(G_W_current.clone())
#         self.R_history.append(R_current.clone())
        
#         # Limit history size to history_size + 1 elements
#         while len(self.R_history) > self.history_size + 1:
#             self.W_history.pop(0)
#             self.G_history.pop(0)
#             self.R_history.pop(0)
        
#         m = len(self.R_history) - 1  # Number of historical points to use
        
#         if m < 1:
#             # Not enough history, just return the LS solution
#             return G_W_current
        
#         # Build the residual difference matrix
#         # Delta_R[:, j] = R_{k-m+j+1} - R_{k-m+j}
#         Delta_R = []
#         for j in range(m):
#             delta = self.R_history[j + 1] - self.R_history[j]
#             Delta_R.append(delta.flatten())
        
#         # Stack into matrix [n x m]
#         Delta_R = torch.stack(Delta_R, dim=1)
#         R_k = self.R_history[-1].flatten()
        
#         # Solve the least squares problem: min ||R_k - Delta_R @ gamma||^2
#         # This gives us the optimal mixing coefficients
#         try:
#             # Add regularization for numerical stability
#             ATA = Delta_R.T @ Delta_R + self.regularization * torch.eye(m, device=Delta_R.device, dtype=Delta_R.dtype)
#             ATb = Delta_R.T @ R_k
#             gamma = torch.linalg.solve(ATA, ATb)
#         except Exception:
#             # If solve fails, fall back to simple iteration
#             return G_W_current
        
#         # Compute the accelerated iterate
#         # W_{k+1} = G(W_k) - sum_{j=0}^{m-1} gamma_j * (G(W_{k-m+j+1}) - G(W_{k-m+j}))
#         W_accelerated = G_W_current.clone()
#         for j in range(m):
#             delta_G = self.G_history[j + 1] - self.G_history[j]
#             W_accelerated = W_accelerated - gamma[j] * delta_G
        
#         return W_accelerated


# # ============================================================================
# # CAMR: Curvature-Aware Manifold Regularization
# # ============================================================================

# def compute_camr_regularization(X_tilde_list, alpha, beta=1e-8, sample_weights=None):
#     """
#     Compute curvature-aware regularization matrix based on input covariance.
    
#     This replaces the isotropic regularization alpha*I with a diagonal matrix
#     that reflects the geometry of the parameter space based on activation statistics.
    
#     Ridge Regression Principle for Closed-Form Solving:
#     - High variance directions = good signal = stable inversion = LESS regularization needed
#     - Low variance directions = near-singular = need regularization for numerical stability
    
#     This is different from EWC/Fisher in SGD fine-tuning where you protect important directions.
#     In closed-form solving, we regularize where the data is WEAK, not where it's strong.
    
#     Args:
#         X_tilde_list: Input features from the merged model [batch, features]
#         alpha: Base regularization strength
#         beta: Minimum regularization value for numerical stability
#         sample_weights: Optional DCS weights for weighted covariance [batch]
    
#     Returns:
#         Lambda_reg: Diagonal regularization matrix
#     """
#     with torch.no_grad():
#         # Apply sample weights if provided (module coordination with DCS)
#         if sample_weights is not None and sample_weights.numel() > 0:
#             # Weight the features before computing covariance
#             batch_size = sample_weights.shape[0]
#             # Ensure tensor has at least 3 dimensions
#             if X_tilde_list.dim() >= 3 and X_tilde_list.shape[1] % batch_size == 0:
#                 features_per_batch = X_tilde_list.shape[1] // batch_size
#                 feature_dim = X_tilde_list.shape[-1]
#                 X_reshaped = X_tilde_list.view(X_tilde_list.shape[0], batch_size, features_per_batch, feature_dim)
#                 sqrt_weights = torch.sqrt(sample_weights).view(1, batch_size, 1, 1)
#                 X_weighted = X_reshaped * sqrt_weights
#                 X_tilde_list = X_weighted.view(X_tilde_list.shape[0], -1, feature_dim)
#             elif X_tilde_list.dim() < 3:
#                 # Not enough dimensions for sample weighting
#                 pass
#             else:
#                 # Dimension mismatch - log warning and skip sample weighting for CAMR
#                 print(f"[CAMR Warning] Dimension mismatch: X_tilde_list.shape[1]={X_tilde_list.shape[1]} "
#                       f"not divisible by batch_size={batch_size}. Skipping DCS-CAMR coordination.")
        
#         # Compute covariance of activations: C = X^T X
#         # X_tilde_list shape: [layers, batch, features]
#         covariance = torch.matmul(X_tilde_list.transpose(-1, -2), X_tilde_list)
        
#         # Extract diagonal elements representing variance in each direction
#         diag_cov = torch.diagonal(covariance, dim1=-2, dim2=-1)
        
#         # Normalize to create relative importance weights
#         diag_norm = diag_cov / (diag_cov.sum(dim=-1, keepdim=True) + 1e-10)
        
#         # Ridge Regression-aligned regularization for closed-form solving:
#         # High variance = strong signal = stable inversion = LESS regularization
#         # Low variance = weak signal = near-singular = MORE regularization
#         # Lambda_reg = alpha * (1 - normalized_variance) + beta
#         Lambda_reg = alpha * (1.0 - diag_norm) + beta
        
#     return Lambda_reg


# def reg_math_camr(term, lambda_reg):
#     """
#     Apply curvature-aware regularization to the term matrix.
    
#     Args:
#         term: The matrix to regularize [batch, d, d]
#         lambda_reg: Diagonal regularization values [batch, d]
    
#     Returns:
#         Regularized matrix
#     """
#     # Vectorized approach: create diagonal matrices and add to term
#     diag_reg = torch.diag_embed(lambda_reg)  # [batch, d, d]
#     return term + diag_reg


# # ============================================================================
# # DCS: Dynamic Conflict-aware Sample Reweighting
# # ============================================================================

# def compute_sample_weights_dcs(outputs_list, sigma=1.0):
#     """
#     Compute dynamic sample weights based on cross-model output variance.
    
#     Samples with high variance across different LoRA models are considered
#     "conflicting" and are down-weighted to reduce interference during merging.
    
#     Args:
#         outputs_list: List of output tensors from each LoRA model [N, batch, output_dim]
#         sigma: Temperature parameter for the Gaussian kernel
    
#     Returns:
#         weights: Sample weights [batch], higher weight = more consensus
#     """
#     with torch.no_grad():
#         if len(outputs_list) < 2:
#             # With only one model, no conflict possible
#             return torch.ones(outputs_list[0].shape[0], device=outputs_list[0].device)
        
#         # Stack outputs: [N, batch, output_dim]
#         stacked = torch.stack(outputs_list, dim=0)
        
#         # Compute mean output across models
#         mean_output = stacked.mean(dim=0)  # [batch, output_dim]
        
#         # Compute variance for each sample across models
#         # V_s = (1/N) * sum_{i=1}^{N} ||y_{s,i} - mean_y_s||^2
#         diff = stacked - mean_output.unsqueeze(0)  # [N, batch, output_dim]
#         variance = (diff ** 2).sum(dim=-1).mean(dim=0)  # [batch]
        
#         # Apply Gaussian kernel: w_s = exp(-V_s / sigma^2)
#         weights = torch.exp(-variance / (sigma ** 2 + 1e-10))
        
#         # Normalize weights to have mean 1 (preserve scale)
#         weights = weights / (weights.mean() + 1e-10)
        
#         return weights


# def compute_feature_variance(X_list):
#     """
#     Compute variance of input features across different LoRA models.
    
#     This is used as a proxy for conflict when we don't have access to output variance.
    
#     Args:
#         X_list: Input features [N, batch, features] where N is number of LoRAs
    
#     Returns:
#         variance: Per-sample variance [batch]
#     """
#     with torch.no_grad():
#         # X_list shape: [N, batch, features] where N is number of LoRAs
#         # Transpose to [batch, N, features] for easier computation
#         X_transposed = X_list.transpose(0, 1)  # [batch, N, features]
#         mean_features = X_transposed.mean(dim=1)  # [batch, features]
#         diff = X_transposed - mean_features.unsqueeze(1)  # [batch, N, features]
#         variance = (diff ** 2).sum(dim=-1).mean(dim=1)  # [batch]
#         return variance


# def compute_output_variance(W_list, X_list):
#     """
#     Compute variance of outputs (W @ X) across different LoRA models.
    
#     This is a more accurate proxy for gradient conflict than input feature variance,
#     as it captures differences in the actual transformations applied by each LoRA.
    
#     Args:
#         W_list: LoRA weight matrices [N, out_dim, in_dim]
#         X_list: Input features [N, batch, features]
    
#     Returns:
#         variance: Per-sample output variance [batch]
#     """
#     with torch.no_grad():
#         N = W_list.shape[0]
#         # Compute outputs for each LoRA: Y_i = W_i @ X_i^T
#         # X_list shape: [N, batch, features], W_list shape: [N, out_dim, in_dim]
#         # We compute the output norm difference across LoRAs
        
#         # For each LoRA, compute output: [N, batch, out_dim]
#         outputs = torch.matmul(X_list, W_list.transpose(-1, -2))  # [N, batch, out_dim]
        
#         # Transpose to [batch, N, out_dim]
#         outputs_transposed = outputs.transpose(0, 1)
        
#         # Compute mean output across LoRAs
#         mean_output = outputs_transposed.mean(dim=1)  # [batch, out_dim]
        
#         # Compute variance
#         diff = outputs_transposed - mean_output.unsqueeze(1)  # [batch, N, out_dim]
#         variance = (diff ** 2).sum(dim=-1).mean(dim=1)  # [batch]
        
#         return variance


# def compute_adaptive_sigma(variance, scale_factor=1.0):
#     """
#     Compute adaptive sigma based on variance distribution.
    
#     This avoids the need for manual tuning of the sigma parameter by
#     adapting it to the actual variance distribution of the data.
    
#     Args:
#         variance: Per-sample variance tensor [batch]
#         scale_factor: Scaling factor for the adaptive sigma
    
#     Returns:
#         sigma: Adaptive sigma value
#     """
#     with torch.no_grad():
#         # Use median absolute deviation (more robust than std)
#         median_var = torch.median(variance)
#         mad = torch.median(torch.abs(variance - median_var))
#         # Scale factor to approximate std from MAD
#         sigma = (mad * 1.4826 + 1e-8) * scale_factor
#         return sigma.item()


# # ============================================================================
# # IterIS++ Core Algorithm
# # ============================================================================

# def reg_math(term, alpha):
#     """Original isotropic regularization (kept for compatibility)."""
#     term_list = [term[i] + alpha[i] * torch.eye(term[i].size(0), dtype=term.dtype, device=term.device) 
#                  for i in range(term.shape[0])]
#     return torch.stack(term_list)


# def solution_matrix_plus(
#     W_list,
#     X_list,
#     X_tilde_list,
#     ceof_list,
#     manual_ceof,
#     alpha_1=1e-7,
#     alpha_2=1e-7,
#     reg_ceof=5e-4,
#     sample_weights=None,
#     use_camr=True,
#     camr_alpha=1e-7,
#     camr_beta=1e-8,
# ):
#     """
#     Enhanced solution matrix computation with CAMR and DCS support.
    
#     This is the core "Solving" step of IterIS++, enhanced with:
#     - CAMR: Curvature-aware regularization based on activation covariance
#     - DCS: Sample-level weighting based on cross-model variance
    
#     Args:
#         W_list: LoRA weight matrices [N, out_dim, in_dim]
#         X_list: Target input features [N, batch, features]
#         X_tilde_list: Current merged model features [N, batch, features]
#         ceof_list: Task-level coefficients
#         manual_ceof: Manual weighting coefficients
#         alpha_1, alpha_2: Regularization coefficients for inner products
#         reg_ceof: Feature mixing coefficient
#         sample_weights: Per-sample weights from DCS [batch]
#         use_camr: Whether to use curvature-aware regularization
#         camr_alpha: CAMR regularization strength
#         camr_beta: CAMR minimum regularization
    
#     Returns:
#         Optimal merged weight matrix
#     """
#     with torch.no_grad():
#         N = W_list.shape[0]
#         manual_ceof = torch.tensor(manual_ceof).to('cuda')
#         X_list, X_tilde_list = X_list.transpose(0, 1).flatten(start_dim=1, end_dim=2), \
#                                X_tilde_list.transpose(0, 1).flatten(start_dim=1, end_dim=2)

#         X_tilde_list = (1 - reg_ceof) * X_tilde_list + reg_ceof * X_list
        
#         # Apply DCS sample weights if provided
#         # Weight the features by sqrt(w) so that the final weighted sum is w * (x^T x)
#         if sample_weights is not None and sample_weights.numel() > 0:
#             # sample_weights shape: [batch]
#             # X_list shape after transpose: [N, batch*features_dim, features]
#             # We need to weight the batch dimension
#             batch_size = sample_weights.shape[0]
#             sqrt_weights = torch.sqrt(sample_weights)  # [batch]
            
#             # Reshape to apply weights to features
#             # Each sample's features should be weighted by sqrt(w)
#             features_per_batch = X_list.shape[1] // batch_size
#             # Check if exact division is possible (i.e., no remainder)
#             if features_per_batch > 0 and X_list.shape[1] % batch_size == 0:
#                 # Reshape to [N, batch, features_per_batch, feature_dim]
#                 X_list_reshaped = X_list.view(N, batch_size, features_per_batch, -1)
#                 X_tilde_reshaped = X_tilde_list.view(N, batch_size, features_per_batch, -1)
                
#                 # Apply weights: [N, batch, features_per_batch, feature_dim] * [batch, 1, 1]
#                 X_list_reshaped = X_list_reshaped * sqrt_weights.view(1, batch_size, 1, 1)
#                 X_tilde_reshaped = X_tilde_reshaped * sqrt_weights.view(1, batch_size, 1, 1)
                
#                 # Reshape back to [N, batch*features_per_batch, feature_dim]
#                 X_list = X_list_reshaped.view(N, -1, X_list_reshaped.shape[-1])
#                 X_tilde_list = X_tilde_reshaped.view(N, -1, X_tilde_reshaped.shape[-1])
        
#         X_X_tilde = torch.matmul(X_list.transpose(-1, -2), X_tilde_list)
#         X_X_tilde_norm = torch.norm(X_X_tilde, p='fro', dim=[-2, -1]) * alpha_1
#         X_X_tilde = reg_math(X_X_tilde, X_X_tilde_norm)

#         X_tilde_X_tilde = torch.matmul(X_tilde_list.transpose(-1, -2), X_tilde_list)
        
#         # Apply CAMR regularization if enabled
#         # Pass sample_weights for module coordination (DCS weights influence CAMR covariance)
#         if use_camr:
#             Lambda_reg = compute_camr_regularization(X_tilde_list, camr_alpha, camr_beta, sample_weights)
#             X_tilde_X_tilde = reg_math_camr(X_tilde_X_tilde, Lambda_reg)
#         else:
#             X_tilde_X_tilde_norm = torch.norm(X_tilde_X_tilde, p='fro', dim=[-2, -1]) * alpha_2
#             X_tilde_X_tilde = reg_math(X_tilde_X_tilde, X_tilde_X_tilde_norm)

#         term1 = torch.sum(torch.matmul(W_list, X_X_tilde) * (ceof_list * manual_ceof).view(N, 1, 1), dim=0).double()
#         term2 = torch.sum(X_tilde_X_tilde * (ceof_list * manual_ceof).view(N, 1, 1), dim=0).double()
#         results = torch.linalg.solve(term2.t(), term1.t()).double().t()

#         return results.to('cpu')


# def update_param_plus(
#     seed,
#     max_iter,
#     lora_path,
#     model_name,
#     task_targets,
#     manual_ceof,
#     shuffle,
#     with_pretrain_matrix=0,
#     max_length=512,
#     lora_alpha=[32, 32],
#     alpha_1=1e-7,
#     alpha_2=1e-7,
#     reg_ceof=5e-4,
#     rank=8,
#     select_long=40,
#     inner_num=2,
#     outer_num=10,
#     samples_num=20,
#     if_divide=True,
#     if_balance=True,
#     # IterIS++ specific parameters
#     use_mats=True,
#     mats_history_size=5,
#     mats_regularization=1e-6,
#     use_camr=True,
#     camr_alpha=1e-7,
#     camr_beta=1e-8,
#     use_dcs=True,
#     dcs_sigma=1.0,
#     convergence_threshold=1e-6,  # For early stopping
#     **generation_kwargs,
# ):
#     """
#     IterIS++ main update function with MATS, CAMR, and DCS innovations.
    
#     This function implements the complete IterIS++ algorithm:
#     1. Initialization: Average of LoRA weights
#     2. Preparation: Compute CAMR regularization matrix
#     3. Iteration Loop:
#        - Inference: Get features from current merged model
#        - DCS: Compute sample weights based on output variance
#        - Solving: Weighted least squares with CAMR regularization
#        - MATS: Anderson acceleration on the weight updates
#        - Convergence check for early stopping
    
#     Args:
#         seed: Random seed
#         max_iter: Maximum number of iterations
#         lora_path: Paths to LoRA adapters
#         model_name: Base model name
#         task_targets: List of task names
#         manual_ceof: Manual weighting coefficients
#         shuffle: Whether to shuffle data
#         use_mats: Enable MATS (Anderson Acceleration)
#         mats_history_size: History depth for Anderson Acceleration
#         mats_regularization: Regularization for MATS least squares
#         use_camr: Enable CAMR (Curvature-Aware Regularization)
#         camr_alpha: CAMR regularization strength
#         camr_beta: CAMR minimum regularization
#         use_dcs: Enable DCS (Dynamic Sample Weighting)
#         dcs_sigma: Scale factor for adaptive DCS sigma
#         convergence_threshold: Threshold for early stopping based on weight change
#         **generation_kwargs: Additional generation arguments
    
#     Returns:
#         Merged model with optimized weights
#     """
#     print("=" * 60)
#     print("IterIS++ Algorithm Starting")
#     print("=" * 60)
#     print(f"Innovations enabled:")
#     print(f"  - MATS (Anderson Acceleration): {use_mats}")
#     print(f"  - CAMR (Curvature-Aware Reg.): {use_camr}")
#     print(f"  - DCS (Dynamic Sample Weight): {use_dcs}")
#     print(f"  - Convergence threshold: {convergence_threshold}")
#     print("=" * 60)
    
#     # Get all mid-features from each LoRA model
#     input_ids_list, X_dict = get_all_midfeatures(
#         rank=rank,
#         seed=seed,
#         select_long=select_long,
#         lora_path=lora_path,
#         model_name=model_name,
#         max_length=max_length,
#         task_targets=task_targets,
#         if_divide=if_divide,
#         if_balance=if_balance,
#         shuffle=shuffle,
#         inner_num=inner_num,
#         outer_num=outer_num,
#         samples_num=samples_num,
#         **generation_kwargs,
#     )

#     pretrain_matrix_dict = get_pretrain_matrix(X_dict.keys(), model_name=model_name)

#     lora_adapter_path_list = [
#         lora_adapter_path + "/adapter_model.safetensors" for lora_adapter_path in lora_path
#     ]
#     tensors_lora = [safe_open(tensor_lora, framework='pt') for tensor_lora in lora_adapter_path_list]
#     torch.cuda.empty_cache()
    
#     X_tilde_dict = {}
    
#     # Initialize Anderson Accelerators for each layer (MATS)
#     anderson_accelerators = {}
#     if use_mats:
#         for idx in X_dict.keys():
#             anderson_accelerators[idx] = AndersonAccelerator(
#                 history_size=mats_history_size,
#                 regularization=mats_regularization
#             )
    
#     # Previous iteration weights for MATS
#     prev_tar_lora_list = {}
    
#     for iteration in range(max_iter):
#         torch.cuda.empty_cache()
#         gc.collect()
#         tar_lora_list = {}
#         print(f"\n-----------IterIS++ Iteration: {iteration + 1}/{max_iter}---------------")
#         print("Computing optimal solution with IterIS++ enhancements...")
        
#         with torch.no_grad():
#             for idx in X_dict.keys():
#                 W_list, X_list = torch.stack(
#                     [get_lora_matrix(model_name, tensors_lora[i], idx, lora_alpha[i], rank=rank, no_weight=True) 
#                      for i in range(len(tensors_lora))]
#                 ).to('cuda'), X_dict[idx].to('cuda')
                
#                 N = W_list.shape[0]
#                 merge_W = W_list + pretrain_matrix_dict[idx].unsqueeze(0).repeat(N, 1, 1).to('cuda')
#                 ceof_list = torch.norm(merge_W, p='fro', dim=[-2, -1]) ** 2 / \
#                             torch.sum(torch.norm(torch.matmul(X_list, merge_W.transpose(1, 2)), p='fro', dim=[-2, -1]) ** 2, dim=0)
                
#                 # DCS: Compute sample weights based on output variance
#                 # Using output variance (W @ X) provides a more accurate proxy for gradient conflict
#                 # than input feature variance alone
#                 sample_weights = None
#                 if use_dcs:
#                     # Use output variance for better conflict detection
#                     output_variance = compute_output_variance(W_list, X_list.transpose(0, 1))
                    
#                     # Use adaptive sigma based on variance distribution (more robust than fixed sigma)
#                     effective_sigma = compute_adaptive_sigma(output_variance, scale_factor=dcs_sigma)
                    
#                     # Compute sample weights using Gaussian kernel
#                     sample_weights = torch.exp(-output_variance / (effective_sigma ** 2 + 1e-10))
#                     sample_weights = sample_weights / (sample_weights.mean() + 1e-10)
                
#                 X_tilde = X_list if iteration == 0 else X_tilde_dict[idx].to('cuda')
                
#                 if with_pretrain_matrix == 0:
#                     W_ls = solution_matrix_plus(
#                         W_list, X_list, X_tilde, ceof_list, manual_ceof,
#                         alpha_1, alpha_2, reg_ceof,
#                         sample_weights=sample_weights,
#                         use_camr=use_camr,
#                         camr_alpha=camr_alpha,
#                         camr_beta=camr_beta,
#                     )
#                 elif with_pretrain_matrix == 1:
#                     W_ls = solution_matrix_plus(
#                         merge_W, X_list, X_tilde, ceof_list, manual_ceof,
#                         alpha_1, alpha_2, reg_ceof,
#                         sample_weights=sample_weights,
#                         use_camr=use_camr,
#                         camr_alpha=camr_alpha,
#                         camr_beta=camr_beta,
#                     )
                
#                 # MATS: Apply Anderson Acceleration
#                 if use_mats and idx in prev_tar_lora_list:
#                     W_current = prev_tar_lora_list[idx].to('cuda')
#                     W_ls_cuda = W_ls.to('cuda')
#                     W_accelerated = anderson_accelerators[idx].update(W_current, W_ls_cuda)
#                     tar_lora_list[idx] = W_accelerated.to('cpu')
#                 else:
#                     tar_lora_list[idx] = W_ls.to('cpu')
                
#                 torch.cuda.empty_cache()
#                 gc.collect()
        
#         # Convergence check: compute total weight change (before storing new weights)
#         converged = False
#         if iteration > 0 and convergence_threshold > 0 and prev_tar_lora_list:
#             total_change = 0.0
#             total_norm = 0.0
#             for k, v in tar_lora_list.items():
#                 if k in prev_tar_lora_list:
#                     total_change += torch.norm(v - prev_tar_lora_list[k]).item()
#                     total_norm += torch.norm(v).item()
#             relative_change = total_change / (total_norm + 1e-10)
#             print(f"Weight relative change: {relative_change:.2e}")
#             if relative_change < convergence_threshold:
#                 print(f"✓ Converged at iteration {iteration + 1} (relative change {relative_change:.2e} < {convergence_threshold})")
#                 converged = True
        
#         # Store current weights for next iteration's MATS (after convergence check)
#         prev_tar_lora_list = {k: v.clone() for k, v in tar_lora_list.items()}
        
#         print("Calculation Done!")
#         print("Loading and updating the merged model...")
        
#         model = None
#         if 't5' in model_name:
#             model = T5WithHooks.from_pretrained(model_name, lora_path=lora_path[0] + '/adapter_model.safetensors').to('cuda')
#         elif 'bart' in model_name:
#             model = BartWithHooks.from_pretrained(model_name, lora_path=lora_path[0] + '/adapter_model.safetensors').to('cuda')
#         elif 'blip' in model_name:
#             model = BlipWithHook.from_pretrained(model_name).to('cuda')
        
#         # Update model with computed weights
#         number_update = 0
#         with torch.no_grad():
#             for name, param in model.named_parameters():
#                 if name[:-7] in tar_lora_list.keys():
#                     lora_matrix = tar_lora_list[name[:-7]].to('cuda')
#                     if with_pretrain_matrix == 0:
#                         param.copy_(lora_matrix + param)
#                     elif with_pretrain_matrix == 1:
#                         param.copy_(lora_matrix)
#                     number_update += 1
        
#         if number_update == len(tar_lora_list.keys()):
#             print("All LoRA targets updated successfully!")
#         else:
#             print("Warning: Some targets were not updated.")
        
#         torch.cuda.empty_cache()
#         max_memory = torch.cuda.max_memory_allocated()
#         print(f"Max memory usage: {max_memory / 1024 ** 2:.2f} MB", flush=True)
        
#         # Check for convergence or max iterations reached
#         if iteration == max_iter - 1 or converged:
#             print("\n" + "=" * 60)
#             if converged:
#                 print(f"IterIS++ Algorithm Complete (Early Stopped at iteration {iteration + 1})")
#             else:
#                 print("IterIS++ Algorithm Complete")
#             print("=" * 60)
#             return model
        
#         # Record mid-features of updated model for next iteration
#         records_list = []
#         if if_divide:
#             assert inner_num * outer_num == len(input_ids_list[0])
#             for input_ids in input_ids_list:
#                 print("Generating merged model midfeatures...")
#                 dict_record_item = {}
#                 for i in range(outer_num):
#                     with torch.no_grad():
#                         outputs = model.generate(input_ids[i * inner_num:(i + 1) * inner_num, :].to('cuda'))
#                     temp_dict = dict(model.inputs_to_track.items())
#                     dict_record_item = temp_dict if i == 0 else {
#                         key: torch.cat([value, temp_dict[key]], dim=0) 
#                         for key, value in dict_record_item.items()
#                     }
#                     model.inputs_to_track.clear()
#                     torch.cuda.empty_cache()
#                 records_list.append(dict_record_item)
#         else:
#             for input_ids in input_ids_list:
#                 model.inputs_to_track.clear()
#                 torch.cuda.empty_cache()
#                 print("Generating merged model midfeatures...")
#                 with torch.no_grad():
#                     if 'blip' in model_name:
#                         outputs = model.generate(**input_ids, max_length=max_length)
#                     else:
#                         outputs = model.generate(input_ids.to('cuda'))
#                 records_list.append(dict(model.inputs_to_track.items()))

#         for item in records_list[0].keys():
#             X_tilde_dict[item] = torch.cat(
#                 [records[item].unsqueeze(dim=1) for records in records_list],
#                 dim=1,
#             ).to('cpu')


# def main():
#     """Main entry point for IterIS++."""
#     parser = argparse.ArgumentParser(description="IterIS++ Training Script")
#     parser.add_argument('--config', type=str, default="config/methods-config/iteris-plus-config.yaml",
#                         help="Path to the config file")
#     parser.add_argument('--task_type', type=str, 
#                         choices=['GLUE_t5', 'EMOTION_t5_large', 'TASKS_blip_base'],
#                         default='GLUE_t5', help="Choose a task type")
    
#     # IterIS++ specific arguments (can override config)
#     parser.add_argument('--use_mats', type=int, default=None, help="Enable MATS (0 or 1)")
#     parser.add_argument('--use_camr', type=int, default=None, help="Enable CAMR (0 or 1)")
#     parser.add_argument('--use_dcs', type=int, default=None, help="Enable DCS (0 or 1)")
    
#     args = parser.parse_args()
#     task_type = args.task_type
    
#     # Load configuration
#     with open(args.config, 'r') as file:
#         config_data = yaml.safe_load(file)
    
#     set_seed(config_data['seed'])
#     model_name = config_data[task_type]['model_name']
#     task_targets = config_data[task_type]['task_targets']
#     lora_path = [get_loras_path(task_type, model_name)[item] for item in task_targets]
#     with_pretrain_matrix = config_data[task_type]['with_pretrain_matrix']
#     tokenizer = AutoTokenizer.from_pretrained(model_name) if 'blip' not in model_name \
#                 else AutoProcessor.from_pretrained(model_name)
#     save = config_data[task_type].get('save', 0)
    
#     # Get IterIS++ specific parameters
#     use_mats = config_data[task_type].get('use_mats', True)
#     use_camr = config_data[task_type].get('use_camr', True)
#     use_dcs = config_data[task_type].get('use_dcs', True)
    
#     # Command line overrides
#     if args.use_mats is not None:
#         use_mats = bool(args.use_mats)
#     if args.use_camr is not None:
#         use_camr = bool(args.use_camr)
#     if args.use_dcs is not None:
#         use_dcs = bool(args.use_dcs)

#     # Run IterIS++ algorithm
#     start_time = time.time()
#     model = update_param_plus(
#         task_targets=task_targets,
#         lora_path=lora_path,
#         model_name=model_name,
#         with_pretrain_matrix=with_pretrain_matrix,
#         max_iter=config_data[task_type]['max_iter'],
#         max_length=config_data[task_type]['max_length'],
#         lora_alpha=config_data[task_type]['lora_alpha'],
#         alpha_1=config_data[task_type]['alpha_1'],
#         alpha_2=config_data[task_type]['alpha_2'],
#         reg_ceof=config_data[task_type]['reg_ceof'],
#         rank=config_data[task_type]['rank'],
#         samples_num=config_data[task_type]['samples_num'],
#         manual_ceof=config_data[task_type]['manual_ceof'],
#         if_divide=config_data[task_type]['if_divide'],
#         if_balance=config_data[task_type]['if_balance'],
#         inner_num=config_data[task_type]['inner_num'],
#         outer_num=config_data[task_type]['outer_num'],
#         seed=config_data['seed'],
#         select_long=config_data[task_type]['select_long'],
#         shuffle=config_data[task_type]['shuffle'],
#         # IterIS++ specific parameters
#         use_mats=use_mats,
#         mats_history_size=config_data[task_type].get('mats_history_size', 5),
#         mats_regularization=config_data[task_type].get('mats_regularization', 1e-6),
#         use_camr=use_camr,
#         camr_alpha=config_data[task_type].get('camr_alpha', config_data[task_type]['alpha_1']),
#         camr_beta=config_data[task_type].get('camr_beta', 1e-8),
#         use_dcs=use_dcs,
#         dcs_sigma=config_data[task_type].get('dcs_sigma', 1.0),
#         convergence_threshold=config_data[task_type].get('convergence_threshold', 1e-6),
#     )
    
#     if save == 1:
#         torch.save(model, "merged_model/merged_model_plus.pth")
#         print("Model saved to merged_model/merged_model_plus.pth")
    
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"\nTotal time: {elapsed_time:.2f} seconds")
    
#     torch.cuda.empty_cache()
#     torch.cuda.reset_max_memory_allocated()
#     gc.collect()

#     # Model evaluation
#     for task_name in task_targets:
#         eval_iteris_model(
#             model=model,
#             tokenizer=tokenizer,
#             model_name=model_name,
#             task_name=task_name,
#             max_length=config_data[task_type]['max_length'],
#             per_device_eval_batch_size=config_data[task_type]['per_device_eval_batch_size'],
#         )
    
#     torch.cuda.empty_cache()
#     torch.cuda.reset_max_memory_allocated()
#     gc.collect()


# if __name__ == "__main__":
#     main()
