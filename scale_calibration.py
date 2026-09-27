"""Per-type Fisher/GGN scale calibration for K-FAC-FW.

K-FAC 用 A⊗B(假设 a⊥δ)近似每层对角 GGN block, 其**尺度**会偏(实测 c_proj tr 偏 3~30×)。
这里用**精确 GGN 沿更新方向 D 的二次型**去校准 K-FAC 的尺度, 按模块类型(attn.q/k/v, mlp.c_fc/c_proj)
分别校准:

    真实:   Q_true(type)  = <D_type, F D_type> = (J·D_type)^T (diag(p)-pp^T) (J·D_type)   (GGN-JVP, 无采样噪声)
    K-FAC:  Q_kfac(type)  = <D_type, scale·B D A>   (**不含 damping**, 按你的要求)
    校准:   scale_cal[type] = EMA( Q_true(type) / <D_type, B D A>_type )      (使 <D, scale_cal·BDA> ≈ Q_true)

damping 不参与校准; 校准后由 adaptive damping 另行加入。

GGN 的曲率全在输出层 H=diag(p)-pp^T; 中间层只是 Jacobian 传播, 所以对任意 block 都是
"JVP 把 D_type 传到 logits + 输出层闭式 H"。用 math-SDPA 保证 flash-attn 也支持 forward-mode。
"""
from __future__ import annotations
import re
import torch
import torch.nn.functional as F
from torch.func import jvp, functional_call

from .curvature import KFACCurvatureOperator


def _module_type(name: str) -> str:
    # transformer.h.3.attn.v_attn -> attn.v_attn
    return re.sub(r"^(module\.|_orig_mod\.)*", "", re.sub(r"^.*?h\.\d+\.", "", name))


def _ggn_quad_along(model, x, params, tangents) -> float:
    """<D, F D> = (J D)^T (diag(p)-pp^T) (J D), 用 GGN-JVP。tangents: 只有目标层非零。
    labels=x 只为强制全位置 logits(forward 在 labels=None 时只算最后一位)。"""
    def f(p):
        logits, _ = functional_call(model, p, (x, x), {"return_logits": True})
        return logits
    # math SDPA: 保证 flash-attention 也支持 forward-mode AD
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        logits, JD = jvp(f, (params,), (tangents,))
    p = torch.softmax(logits.float(), dim=-1)
    jd = JD.float()
    # 逐 token: Σ_i p_i jd_i^2 - (Σ_i p_i jd_i)^2  (= (JD)^T H (JD))
    q = (p * jd * jd).sum(-1) - (p * jd).sum(-1) ** 2
    return float(q.mean())    # per-token 均值, 与 K-FAC 的 per-token 均值因子 A/B 同口径


@torch.no_grad()
def _kfac_bda_by_type(kfac, curv, layers_by_type):
    """每 type 的 Σ_l <D_l, B_l D_l A_l>  (无 damping, scale=1)。返回 {type: float}。"""
    out = {}
    for t, states in layers_by_type.items():
        s = 0.0
        for st in states:
            D = st.previous_direction
            if D is None or st.A is None:
                continue
            s += float(curv.quadratic_form(D.float(), st.A, st.B, 0.0, 1.0))  # damping=0, scale=1
        out[t] = s
    return out


def calibrate_pertype(model, x, kfac, beta=0.95, clamp=(0.05, 50.0),
                      only_types=None, cal_batch=4):
    """算 per-type 真实 GGN, 更新 kfac._scale_cal(EMA)。返回本次 raw 的 {type: scale}。

    ⚠️ 权重要求: GGN-JVP 的 forward 用**当前权重**, 而 D(=previous_direction)是在 apply 前的
    权重 θ_t 上解出的、K-FAC 因子 A/B 也在 θ_t。所以**必须在权重 = θ_t 时调用**, 否则
    Q_true(θ_{t+1}) 与 Q_KFAC(θ_t) 错配。step() 会把权重改成 θ_{t+1}, 因此调用协议是:
        存 θ_t 快照 -> optimizer.step() -> 把权重恢复回 θ_t -> calibrate_pertype -> 恢复 θ_{t+1}
    (不能简单地"在 step 之后直接调"——那时权重已是 θ_{t+1}。)

    cal_batch: JVP 用的小 batch。注意 heavy-tail: 单 batch CV~300%, 需池化 ~128+ batch(见验证)。
    """
    curv = KFACCurvatureOperator()
    x = x[:cal_batch]                        # 校准用小 batch, 避开 full-vocab CE 的显存
    params = dict(model.named_parameters())

    # 按 type 分组 KFAC 层(有 previous_direction 且有因子)
    layers_by_type = {}
    for name, st in kfac.layer_states.items():
        if st.previous_direction is None or st.A is None:
            continue
        t = _module_type(name)
        if only_types is not None and t not in only_types:
            continue
        layers_by_type.setdefault(t, []).append((name, st))

    # K-FAC 侧(无 damping)
    lbt_states = {t: [st for _, st in v] for t, v in layers_by_type.items()}
    bda = _kfac_bda_by_type(kfac, curv, lbt_states)

    # 真实 GGN: 每 type 一次 JVP, tangent 只在该 type 的层 = D_l
    capture_was = kfac.capture_context.enabled
    kfac.capture_context.enabled = False           # JVP 期间不采因子
    raw = {}
    try:
        for t, states in layers_by_type.items():
            tangents = {n: torch.zeros_like(p) for n, p in params.items()}
            for name, st in states:
                pname = name + ".weight"
                if pname in tangents:
                    tangents[pname] = st.previous_direction.to(params[pname].dtype).reshape(params[pname].shape)
            q_true = _ggn_quad_along(model, x, params, tangents)
            denom = bda.get(t, 0.0)
            if denom > 1e-20 and q_true > 0:
                raw[t] = q_true / denom                     # 使 <D, scale·BDA> = Q_true 的 scale
    finally:
        kfac.capture_context.enabled = capture_was

    # EMA 更新 kfac._scale_cal
    if not hasattr(kfac, "_scale_cal"):
        kfac._scale_cal = {}
    lo, hi = clamp
    for t, s in raw.items():
        s = float(min(max(s, lo), hi))
        prev = kfac._scale_cal.get(t)
        kfac._scale_cal[t] = s if prev is None else beta * prev + (1 - beta) * s
    return raw


@torch.no_grad()
def _bda(curv, st):
    return float(curv.quadratic_form(st.previous_direction.float(), st.A, st.B, 0.0, 1.0))


def calibrate_perlayer(kfac, model, mbs, *, curv=None, ratio=0.1, beta=0.7,
                       clamp=(0.05, 50.0), cal_batch=4, world_size=1, device=None, eps=0.1):
    """per-layer 曲率尺度校准, 就地 EMA 更新每层 st.scale_cal(供 optimizer._use_scale_cal 使用)。

    在【当前权重 = θ_t】上调用(step 后须先把权重 restore 回 θ_t, 见 train loop 钩子):
    对每 KFAC 层, 用 mbs(当前大 batch 的 microbatch 列表)按 ratio 子采样, 算
    Q_true = <D, F D> = (J D)^T H (J D)(D=st.previous_direction), s = Q_true / <D,B D A>, EMA 进 st.scale_cal。

    J·D 用**有限差分** (f(θ+εD)−f(θ))/ε 估(普通 flash 前向, fp32; 共享 f(θ)), 而非前向模式 JVP
    (后者被迫用 math-SDPA, 慢 ~10-20×)。实测 ε=1e-2 相对精确 JVP 误差 ~0%、无符号错。

    model: **raw**(非 DDP/compile, 名字无 module./_orig_mod. 前缀)模型。mbs: [(x,y),...] 当前大 batch。
    对齐建议: 调用步取 KFAC 因子更新间隔的整数倍, 此时 D/A/B 最新鲜。
    """
    curv = curv or KFACCurvatureOperator()
    params = dict(model.named_parameters())
    if not mbs:
        return 0
    seqs = torch.cat([x for x, _ in mbs], dim=0)
    nk = max(cal_batch, int(round(ratio * seqs.shape[0])))
    xb_list = [seqs[i:i + cal_batch] for i in range(0, nk, cal_batch)]
    lo, hi = clamp
    dev = device or seqs.device
    n_updated = 0
    was = kfac.capture_context.enabled
    kfac.capture_context.enabled = False           # 校准前向期间不采因子
    # FD 抵消需要全 fp32: 训练可能设了 TF32(set_float32_matmul_precision('high')), TF32 只有 ~10 位
    # 尾数, 会把 (f(θ+ΔW)−f(θ)) 的抵消噪声放大 → 二次型 Q_true 正偏置 → scale 虚高。校准期间强制 highest。
    _prec = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.no_grad():
            # 共享 f(θ)(fp32 flash 前向)与输出层 H 用的 p=softmax(f0)
            f0s = [model(xb, xb, return_logits=True)[0].float() for xb in xb_list]
            p0s = [torch.softmax(f0, -1) for f0 in f0s]
            for n, st in kfac.layer_states.items():
                if st.previous_direction is None or st.A is None:
                    continue
                bda = _bda(curv, st)
                if bda <= 1e-20:
                    continue
                pw = n + ".weight"
                if pw not in params:               # 名字对不上(如 _orig_mod. 前缀未剥)→ 会全跳过
                    continue
                W = params[pw]
                D = st.previous_direction.to(W.dtype).reshape(W.shape)
                Dn = float(D.norm())
                if Dn < 1e-20:
                    continue
                # 扰动归一化到固定幅度 eps·D̂(与 ‖D‖ 无关), 否则 D 变小时 fp32 抵消噪声/‖D‖² 正反馈爆炸
                alpha = eps / Dn                   # ΔW = eps·(D/‖D‖)
                scale_jd = Dn / eps                # J·D = (‖D‖/eps)·(f(θ+ΔW) − f(θ))
                W0 = W.data.clone()                # 精确还原用
                qs = nt = 0.0
                for xb, f0, p0 in zip(xb_list, f0s, p0s):
                    W.data.add_(D, alpha=alpha)    # θ + eps·D̂
                    fp = model(xb, xb, return_logits=True)[0].float()
                    W.data.copy_(W0)               # 还原 θ
                    jd = (fp - f0) * scale_jd      # ≈ J·D
                    q = (p0 * jd * jd).sum(-1) - (p0 * jd).sum(-1) ** 2   # (JD)^T H (JD)
                    qs += float(q.sum()); nt += q.numel()
                if world_size > 1:
                    import torch.distributed as dist
                    t = torch.tensor([qs, nt], device=dev); dist.all_reduce(t); qs, nt = float(t[0]), float(t[1])
                if nt <= 0:
                    continue
                s = min(max((qs / nt) / bda, lo), hi)
                st.scale_cal = beta * float(getattr(st, "scale_cal", 1.0)) + (1 - beta) * s
                n_updated += 1
    finally:
        kfac.capture_context.enabled = was
        torch.set_float32_matmul_precision(_prec)   # 恢复训练的 TF32 设置
    if n_updated:
        kfac._scale_cal_dirty = True   # scale_cal 已变 → 下一步 _step_batched 重建 scale_vec(免滞后一个刷新周期)
    return n_updated


def enable_scale_calib(kfac, init=1.0):
    """开启 per-layer scale 校准: 打开优化器 flag + 初始化每层 scale_cal。"""
    kfac._use_scale_cal = True
    for st in kfac.layer_states.values():
        st.scale_cal = float(init)
