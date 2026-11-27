"""Real open-weight model tensor extraction with deterministic synthetic fallback."""

import gc
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_PROMPT = (
    "Systolic arrays multiply stationary weights against streaming activations "
    "across synchronized clock cycles in neural network accelerators."
)

_LAYER_SUFFIX_MAP = {
    "attn_q_proj": "self_attn.q_proj",
    "attn_k_proj": "self_attn.k_proj",
    "attn_v_proj": "self_attn.v_proj",
    "attn_o_proj": "self_attn.o_proj",
    "mlp_gate_proj": "mlp.gate_proj",
    "mlp_up_proj": "mlp.up_proj",
    "mlp_down_proj": "mlp.down_proj",
}


@dataclass(frozen=True)
class LayerWorkload:
    """One 2D matrix multiply workload C = activations @ weights."""

    name: str
    model_id: str
    activations: np.ndarray
    weights: np.ndarray
    source: Literal["huggingface", "synthetic"]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return (M, K, N) dimensions for activations (M, K) and weights (K, N)."""
        return (self.activations.shape[0], self.activations.shape[1], self.weights.shape[1])


def generate_synthetic_workloads(
    seed: int = 42,
    m_dim: int = 16,
    k_dim: int = 128,
    n_dim: int = 128,
) -> dict[str, LayerWorkload]:
    """Build deterministic transformer-like synthetic activations and weights."""
    if m_dim <= 0 or k_dim <= 0 or n_dim <= 0:
        raise ValueError(f"Dimensions must be positive, got ({m_dim}, {k_dim}, {n_dim})")

    rng = np.random.default_rng(seed)

    # Attention-like projection: Gaussian weights plus 2% heavy-tailed outlier channels
    a_attn = rng.normal(loc=0.0, scale=0.65, size=(m_dim, k_dim)).astype(np.float32)
    num_outliers = max(1, int(np.ceil(0.02 * k_dim)))
    outlier_cols = rng.choice(k_dim, size=num_outliers, replace=False)
    a_attn[:, outlier_cols] *= np.float32(8.0)
    w_attn = rng.normal(loc=0.0, scale=1.0 / np.sqrt(k_dim), size=(k_dim, n_dim)).astype(
        np.float32
    )

    # MLP down-like projection: SiLU-like skewed activations and scaled weights
    k_mlp = max(k_dim, 2 * k_dim)
    pre_act = rng.normal(loc=0.0, scale=1.1, size=(m_dim, k_mlp)).astype(np.float32)
    silu_act = (pre_act * (1.0 / (1.0 + np.exp(-np.clip(pre_act, -20.0, 20.0))))).astype(
        np.float32
    )
    mlp_outliers = rng.choice(k_mlp, size=max(1, int(np.ceil(0.015 * k_mlp))), replace=False)
    silu_act[:, mlp_outliers] *= np.float32(6.0)
    w_mlp = rng.normal(loc=0.0, scale=1.0 / np.sqrt(k_mlp), size=(k_mlp, n_dim)).astype(
        np.float32
    )

    return {
        "attn_q_proj": LayerWorkload(
            name="attn_q_proj",
            model_id="synthetic-transformer-v1",
            activations=a_attn,
            weights=w_attn,
            source="synthetic",
            metadata={
                "seed": seed,
                "outlier_channels": int(num_outliers),
                "act_max_abs": float(np.max(np.abs(a_attn))),
            },
        ),
        "mlp_down_proj": LayerWorkload(
            name="mlp_down_proj",
            model_id="synthetic-transformer-v1",
            activations=silu_act,
            weights=w_mlp,
            source="synthetic",
            metadata={
                "seed": seed,
                "outlier_channels": int(len(mlp_outliers)),
                "act_max_abs": float(np.max(np.abs(silu_act))),
            },
        ),
    }


def _resolve_target_modules(
    model: Any,
    layer_idx: int | None,
    target_layers: tuple[str, ...],
) -> dict[str, tuple[str, Any]]:
    """Locate target nn.Linear modules in standard causal LM architectures."""
    named_modules = dict(model.named_modules())

    num_hidden_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if layer_idx is None:
        resolved_idx = (num_hidden_layers // 2) if isinstance(num_hidden_layers, int) else 0
    else:
        resolved_idx = layer_idx

    matched: dict[str, tuple[str, Any]] = {}
    for label in target_layers:
        suffix = _LAYER_SUFFIX_MAP.get(label, label)
        candidate_prefixes = (
            f"model.layers.{resolved_idx}.{suffix}",
            f"transformer.h.{resolved_idx}.{suffix}",
        )
        found_name = None
        for cand in candidate_prefixes:
            if cand in named_modules:
                found_name = cand
                break

        if found_name is None:
            layer_token = f".{resolved_idx}."
            for mod_name in named_modules:
                if mod_name.endswith(suffix) and (layer_token in mod_name or resolved_idx == 0):
                    found_name = mod_name
                    break

        if found_name is None:
            raise KeyError(
                f"Could not find module for '{label}' (suffix '{suffix}') at layer {resolved_idx}"
            )
        matched[label] = (found_name, named_modules[found_name])

    return matched


def load_huggingface_workloads(
    model_id: str = DEFAULT_MODEL_ID,
    layer_idx: int | None = None,
    target_layers: tuple[str, ...] = ("attn_q_proj", "mlp_down_proj"),
    seq_len: int = 16,
    batch_size: int = 1,
    max_k: int | None = None,
    max_n: int | None = None,
    seed: int = 42,
    prompt: str | None = None,
) -> dict[str, LayerWorkload]:
    """Extract real activations and weights from a compact HuggingFace causal LM."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    text = DEFAULT_PROMPT if prompt is None else prompt

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()

    try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            encoded = tokenizer(
                [text] * batch_size,
                return_tensors="pt",
                truncation=True,
                max_length=seq_len,
            )
            input_ids = encoded["input_ids"]
            if input_ids.shape[1] == 0:
                raise ValueError("Tokenizer produced empty sequence")
            if input_ids.shape[1] < seq_len:
                repeats = (seq_len // input_ids.shape[1]) + 1
                input_ids = input_ids.repeat(1, repeats)[:, :seq_len]
        except Exception:
            vocab_size = int(getattr(model.config, "vocab_size", 32000))
            rng = np.random.default_rng(seed)
            ids_np = rng.integers(0, vocab_size, size=(batch_size, seq_len), dtype=np.int64)
            input_ids = torch.from_numpy(ids_np)

        modules = _resolve_target_modules(model, layer_idx=layer_idx, target_layers=target_layers)
        captured_inputs: dict[str, np.ndarray] = {}
        hooks = []

        def make_hook(label: str):
            def hook_fn(_module: Any, inputs: tuple[Any, ...], _output: Any) -> None:
                x = inputs[0].detach().to(device="cpu", dtype=torch.float32)
                captured_inputs[label] = x.reshape(-1, x.shape[-1]).numpy().copy()

            return hook_fn

        for label, (_mod_name, mod) in modules.items():
            hooks.append(mod.register_forward_hook(make_hook(label)))

        try:
            with torch.no_grad():
                model(input_ids=input_ids, use_cache=False)
        finally:
            for handle in hooks:
                handle.remove()

        workloads: dict[str, LayerWorkload] = {}
        for label, (mod_name, mod) in modules.items():
            if label not in captured_inputs:
                raise RuntimeError(f"Forward hook did not capture activations for '{label}'")

            act = captured_inputs[label]
            # nn.Linear stores weight as (out_features, in_features), so transpose to (K, N)
            w_t = (
                mod.weight.detach()
                .to(device="cpu", dtype=torch.float32)
                .t()
                .contiguous()
                .numpy()
                .copy()
            )

            if max_k is not None:
                act = act[:, :max_k].copy()
                w_t = w_t[:max_k, :].copy()
            if max_n is not None:
                w_t = w_t[:, :max_n].copy()

            workloads[label] = LayerWorkload(
                name=label,
                model_id=model_id,
                activations=act,
                weights=w_t,
                source="huggingface",
                metadata={
                    "module_name": mod_name,
                    "batch_size": batch_size,
                    "seq_len": int(input_ids.shape[1]),
                    "act_max_abs": float(np.max(np.abs(act))),
                    "weight_max_abs": float(np.max(np.abs(w_t))),
                },
            )
        return workloads
    finally:
        del model
        gc.collect()


def load_workloads(
    source: Literal["auto", "huggingface", "synthetic"] = "auto",
    model_id: str = DEFAULT_MODEL_ID,
    layer_idx: int | None = None,
    target_layers: tuple[str, ...] = ("attn_q_proj", "mlp_down_proj"),
    seq_len: int = 16,
    batch_size: int = 1,
    max_k: int | None = None,
    max_n: int | None = None,
    seed: int = 42,
) -> dict[str, LayerWorkload]:
    """Load real model workloads or fall back cleanly to deterministic synthetic tensors."""
    if source == "synthetic":
        m_dim = batch_size * seq_len
        k_dim = 128 if max_k is None else max_k
        n_dim = 128 if max_n is None else max_n
        return generate_synthetic_workloads(seed=seed, m_dim=m_dim, k_dim=k_dim, n_dim=n_dim)

    if source == "huggingface":
        return load_huggingface_workloads(
            model_id=model_id,
            layer_idx=layer_idx,
            target_layers=target_layers,
            seq_len=seq_len,
            batch_size=batch_size,
            max_k=max_k,
            max_n=max_n,
            seed=seed,
        )

    if source != "auto":
        raise ValueError(f"Unsupported workload source '{source}'")

    try:
        return load_huggingface_workloads(
            model_id=model_id,
            layer_idx=layer_idx,
            target_layers=target_layers,
            seq_len=seq_len,
            batch_size=batch_size,
            max_k=max_k,
            max_n=max_n,
            seed=seed,
        )
    except Exception as exc:
        logger.warning("Falling back to synthetic workloads (%s): %s", model_id, exc)
        m_dim = batch_size * seq_len
        k_dim = 128 if max_k is None else max_k
        n_dim = 128 if max_n is None else max_n
        return generate_synthetic_workloads(seed=seed, m_dim=m_dim, k_dim=k_dim, n_dim=n_dim)
