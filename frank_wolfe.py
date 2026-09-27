"""Spectral Trust-Region Frank–Wolfe solver。
maximize  q(D) = <G̃, D> - (η/2)<D, C[D]>   s.t. ||D||_op <= ρ
LMO: S_k = ρ·polar(R_k), R_k = G̃ - η C[D_k]; exact line search γ=clip(g_k/h_k,0,1)。

性能: 利用 C 线性 + D_{k+1}=(1-γ)D_k+γS_k, 缓存 C[D_k], 每轮只新算一次 C[S_k]:
  C[E_k]=C[S_k]-C[D_k], C[D_{k+1}]=C[D_k]+γ C[E_k]; obj/backtrack 用标量二次型 → 每轮仅 1 次 BDA。

正确性: 收敛判据在更新前检查; final_* 都对应返回的 D(不错位一步)。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import torch

from .config import FWConfig, PolarConfig
from .curvature import KFACCurvatureOperator
from .polar import polar_svd, polar_error
from .newton_schulz import newton_schulz_polar
from .projections import project_operator_norm_ball, operator_norm


def _make_polar(polar_config: Optional[PolarConfig]):
    if polar_config is None or polar_config.method == "svd":
        return polar_svd
    if polar_config.method == "newton_schulz":
        pc = polar_config
        return lambda X, rho: newton_schulz_polar(
            X, rho, steps=pc.ns_steps, coefficients=pc.ns_coefficients, eps=pc.ns_epsilon,
            normalization=pc.ns_normalization, svd_fallback=pc.svd_fallback,
            fallback_tol=pc.svd_fallback_tolerance)
    raise ValueError(f"unknown polar method {polar_config.method}")


@dataclass
class FWSolverInfo:
    num_steps: int = 0
    gaps: List[float] = field(default_factory=list)
    relative_gaps: List[float] = field(default_factory=list)
    gammas: List[float] = field(default_factory=list)
    objectives: List[float] = field(default_factory=list)
    operator_norms: List[float] = field(default_factory=list)
    polar_errors: List[float] = field(default_factory=list)
    curvature_evals: int = 0          # C[·] 调用次数(性能诊断)
    final_gap: float = 0.0
    final_relative_gap: float = 0.0
    final_objective: float = 0.0
    final_predicted_decrease: float = 0.0
    final_operator_norm: float = 0.0
    converged: bool = False
    stop_reason: str = ""

# ---- 批处理快路径: 同形状层 stack 成 [n,do,di] 跑固定步 FW(bmm), 全程 on-device 标量 ----
_NS_QUINTIC = (3.4445, -4.7750, 2.0315)
_NS_CUBIC = (1.5, -0.5, 0.0)

# 批处理快路径的重型 bmm(polar 内积矩阵 + 曲率 B@X@A) 默认走 bf16(对标 Muon, tensor core 满速);
# 标量内积/线搜/状态量保持 float32。实测 60m solve ~1.7x, polar 方向 cosine 偏差 ~3e-4(可忽略)。
# 置 False 回退全 float32。
BATCHED_BF16 = True

# True: FW 求解的【状态量】(D/CD/R/S/E/CE)也走 bf16 → 循环里 pointwise/线搜的访存减半(~2x),
# 曲率不再 .float() materialize。代价: line search 精度降到 bf16(优化器对精度不敏感, 需 A/B)。
# 打的是"非-NS"那部分(NS/polar 在 compile 下已很快且对步数不敏感)。默认 False = 现状(状态 fp32)。
BATCHED_BF16_STATE = True

# @torch.compile
# @torch.compile(dynamic=False)
@torch.compile(dynamic=True)
def _polar_batched(R: torch.Tensor, rho: float, steps: int, coefficients: str, eps: float = 1e-7):
    """批量 Newton–Schulz polar, 与 newton_schulz_polar 同数学(fro 归一化按 batch 元素)。"""
    a, b, c = _NS_QUINTIC if coefficients == "quintic" else _NS_CUBIC
    Z = R.bfloat16() if BATCHED_BF16 else R.float()
    transposed = Z.shape[-2] > Z.shape[-1]
    if transposed:
        Z = Z.mT
    Z = Z / (Z.norm(dim=(-2, -1), keepdim=True) + eps)
    cubic = (c == 0.0)
    for _ in range(steps):
        ZZt = Z @ Z.mT
        if cubic:
            Z = a * Z + b * (ZZt @ Z)
        else:
            Z = a * Z + (b * ZZt + c * (ZZt @ ZZt)) @ Z
    if transposed:
        Z = Z.mT
    return (rho * Z).to(R.dtype)

@torch.compile(dynamic=True)
def solve_fw_batched(Gs: torch.Tensor, As: torch.Tensor, Bs: torch.Tensor,
                     eta: float, rho: float, steps: int, damping: float, curvature_scale: float,
                     polar_config: Optional[PolarConfig] = None,
                     D_init: Optional[torch.Tensor] = None,
                     project_init: bool = True,
                     per_layer_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    """对一组同形状层批量解 FW(固定 steps 步, exact line search)。
    Gs:[n,do,di] As:[n,di,di] Bs:[n,do,do] -> Ds:[n,do,di]。dense 因子。
    D_init: warm-start 起点(每层上一步方向 stack); None=冷启动(D=0)。
    project_init: warm-start 时把 D_init 批量投影到算子范数球 ‖·‖_op≤rho。
    per_layer_scale: [n] per-layer 曲率尺度校准系数(仅乘 BDA 项, 不乘 damping)。None=不做。
    数学同逐层 solver(去掉数据依赖提前停止与单调回溯; 实测对最终方向影响 ~1e-3)。"""
    coeff = polar_config.ns_coefficients if (polar_config and polar_config.method == "newton_schulz") else "quintic"
    nstep = polar_config.ns_steps if (polar_config and polar_config.method == "newton_schulz") else 5
    use_ns = polar_config is not None and polar_config.method == "newton_schulz"

    cs = curvature_scale if per_layer_scale is None else curvature_scale * per_layer_scale.view(-1, 1, 1)
    state_bf16 = BATCHED_BF16 and BATCHED_BF16_STATE
    if state_bf16:
        Gs = Gs.bfloat16()
        if torch.is_tensor(cs):
            cs = cs.bfloat16()
    if BATCHED_BF16:
        Bs_c, As_c = Bs.bfloat16(), As.bfloat16()        # 因子转一次 bf16, C 内复用
        if state_bf16:
            def C(X):
                # scale = 1.5
                # return scale * (cs * (Bs_c @ X @ As_c) + damping * X)          # 全 bf16, 无 .float()
                scale = cs
                return scale * ((Bs_c @ X @ As_c)  + damping * X )         # 全 bf16, 无 .float()
        else:
            def C(X):
                # 重型曲率 bmm 走 bf16(tensor core), damping 项与返回保持 float32
                return cs * (Bs_c @ X.bfloat16() @ As_c).float() + damping * X
    else:
        def C(X):
            scale = 1.0
            return scale * (cs * (Bs @ X @ As) + damping * X)

    def polar(R):
        if use_ns:
            return _polar_batched(R, rho, nstep, coeff)
        # SVD polar 按 batch
        U, _, Vh = torch.linalg.svd(R.float(), full_matrices=False)
        return (rho * (U @ Vh)).to(R.dtype)

    def polar_unit(R):   # 单位 polar(算子范数=1, 不含 rho), 用于自适应扩 rho
        if use_ns:
            return _polar_batched(R, 1.0, nstep, coeff)
        U, _, Vh = torch.linalg.svd(R.float(), full_matrices=False)
        return (U @ Vh).to(R.dtype)

    def rdot(X, Y):  # 逐层 Frobenius 内积 -> [n,1,1]
        return (X * Y).sum(dim=(-2, -1), keepdim=True)

    if D_init is None:
        D = torch.zeros_like(Gs)
        CD = torch.zeros_like(Gs)
    else:
        D = D_init.to(Gs.dtype)                            # 跟状态 dtype(state_bf16 时为 bf16)
        if project_init:                                  # 批量投影到 ‖D‖_op≤rho(一次 batched SVD)
            opn = torch.linalg.svdvals(D.float()).amax(dim=-1)        # [n] 每层最大奇异值
            scale = (rho / opn.clamp_min(rho)).to(D.dtype)           # >rho 才缩, 否则 1
            D = D * scale.view(-1, *([1] * (D.dim() - 1)))
        CD = C(D)

    for _ in range(steps):
        R = Gs - eta * CD
        S = polar(R)
        CS = C(S)
        E = S - D
        CE = CS - CD
        g = rdot(R, E)
        qEE = rdot(E, CE)
        h = eta * qEE
        gamma = torch.clamp(g / (h + 1e-12), 0.0, 1.0)
        gamma = torch.where(g <= 0, torch.zeros_like(gamma), gamma)
        # print(gamma)
        D = D + gamma * E
        CD = CD + gamma * CE
    return D
