"""
Train an MNIST predictive coding network with Orchard & Sun-style decay.

This file is intentionally a thin notebook-friendly wrapper around helpers
implemented in the trainer utilities.

Orchard & Sun (2019, arXiv:1910.12151) key idea for generative reconstructions:
- Add L2 decay on node activities and weights to bias inference toward minimum-norm solutions.

In the original project code, the trainer utilities provide:
- `orchard_sun_train_mnist(...)`
- `orchard_sun_reconstruct_mnist(...)`

NOTE: In this implementation the sensory input x is clamped (not a Vode state) during training,
so `l2_x` mainly matters for generation/inversion where x is free. It's included for completeness.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from topological_dl.config import CONFIG, dataset_results_dir
trainer_mod = None


def _load_trainer_orchard():
    global trainer_mod
    if trainer_mod is not None:
        return trainer_mod
    try:
        import Trainer_orchard as loaded_trainer_mod
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "generate_mnist_orchard_reconstructions.py is a legacy wrapper that "
            "requires an external Trainer_orchard.py module. It is retained for "
            "provenance, but it is not currently a self-contained public "
            "replication entry point."
        ) from exc
    trainer_mod = loaded_trainer_mod
    return trainer_mod


@dataclass(frozen=True)
class Config:
    # Data
    dataset: str = "MNIST"
    data_root: str = str(CONFIG.data_dir)

    # Model
    hidden_dims: tuple[int, int] = (600, 600)
    act: str = "tanh"  # "tanh" or "relu"
    residual: bool = False
    untie_feedback_weights: bool = True

    # Orchard & Sun decay coefficients
    l2_w: float = 5e-5
    l2_x: float = 5e-4
    l2_h: float = 5e-4

    # Training
    batch_size: int = 128
    epochs: int = 10
    T_infer: int = 20
    init_lr_w: float = 5e-4
    init_lr_h: float = 3e-2
    trans_mult: int = 1
    decay_rate: float = 0.98
    norms_every_batches: int = 0  # 0 disables batch-level logging
    norms_every_epochs: int = 1   # print norms every N epochs (>=1)

    # Repro
    seed: int = 0

    # Saving
    out_dir: str = str(dataset_results_dir("MNIST") / "orchard_sun_mnist_run")

    # Reconstruction (label-conditioned generation)
    do_reconstruct: bool = True
    recon_label: int = 0
    recon_samples: int = 64
    recon_steps: int = 200
    recon_lr: float = 0.05
    recon_noise_sigma: float = 0.0
    recon_temp: float = 1.0
    recon_init_kind: str = "normal"
    recon_init_scale: float = 0.5
    recon_batch_size: int = 32
    recon_state_T: int = 0
    recon_state_lr: float = 0.05
    recon_save_npz: bool = False
    recon_save_png: bool = False


def _act_fn(name: str):
    import jax

    name = name.lower()
    if name == "tanh":
        return jax.nn.tanh
    if name == "relu":
        return jax.nn.relu
    raise ValueError(f"Unsupported act: {name}")

def train_pcn(cfg: Config):
    trainer_mod = _load_trainer_orchard()
    act_name = cfg.act
    act_fn = _act_fn(act_name)
    model, run_info = trainer_mod.orchard_sun_train_mnist(
        data_root=cfg.data_root,
        out_dir=cfg.out_dir,
        seed=cfg.seed,
        hidden_dims=list(cfg.hidden_dims),
        act_fn=act_fn,
        act_name=act_name,
        residual=cfg.residual,
        untie_feedback_weights=cfg.untie_feedback_weights,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        T_infer=cfg.T_infer,
        init_lr_w=cfg.init_lr_w,
        init_lr_h=cfg.init_lr_h,
        trans_mult=cfg.trans_mult,
        decay_rate=cfg.decay_rate,
        l2_w=cfg.l2_w,
        l2_x=cfg.l2_x,
        l2_h=cfg.l2_h,
        norms_every_epochs=cfg.norms_every_epochs,
    )
    info = {
        "run_id": run_info.run_id,
        "ckpt_path": run_info.ckpt_path,
        "metrics_path": run_info.metrics_path,
        "input_dim": run_info.input_dim,
        "output_dim": run_info.output_dim,
    }
    return model, info


def _fmt_float_for_fname(x: float) -> str:
    if x == 0:
        return "0"
    s = f"{x:.3g}"
    s = s.replace("+", "")
    s = s.replace(".", "p")
    return s


def build_orchard_sun_run_id(config: Config) -> str:
    hd = "x".join(str(d) for d in config.hidden_dims)
    return (
        f"{config.dataset}"
        f"_hd{hd}"
        f"_{config.act}"
        f"_T{int(config.T_infer)}"
        f"_bs{int(config.batch_size)}"
        f"_l2w{_fmt_float_for_fname(float(config.l2_w))}"
        f"_l2x{_fmt_float_for_fname(float(config.l2_x))}"
        f"_l2h{_fmt_float_for_fname(float(config.l2_h))}"
        f"_ufb{int(bool(config.untie_feedback_weights))}"
        f"_res{int(bool(config.residual))}"
        f"_seed{int(config.seed)}"
    )


def load_trained_model(cfg: Config):
    trainer_mod = _load_trainer_orchard()
    import jax
    import jax.numpy as jnp
    """
    Load a trained model from disk into a fresh instance.
    Useful if you want to do reconstructions in a separate notebook session.
    """
    # Delegate to the trainer helper: keep this wrapper minimal.
    # (A pure loader can be added to the trainer utilities later if needed.)
    act_fn = _act_fn(cfg.act)
    _, _, input_dim, output_dim = trainer_mod.load_dataset(cfg.dataset, root=cfg.data_root)
    key = jax.random.PRNGKey(int(cfg.seed))
    model_key, _ = jax.random.split(key)
    model = trainer_mod.Model(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=list(cfg.hidden_dims),
        act_fn=act_fn,
        model_key=model_key,
        residual=cfg.residual,
        l2_w=cfg.l2_w,
        l2_x=cfg.l2_x,
        l2_h=cfg.l2_h,
        untie_feedback_weights=cfg.untie_feedback_weights,
    )
    with trainer_mod.pxu.step(model, trainer_mod.pxc.STATUS.INIT, clear_params=trainer_mod.pxc.VodeParam.Cache):
        trainer_mod.forward(jnp.zeros((cfg.batch_size, input_dim)), None, model=model)

    run_id = build_orchard_sun_run_id(cfg)
    ckpt_path = os.path.join(cfg.out_dir, f"model_best_{run_id}")
    
    trainer_mod.pxu.load_params(model, ckpt_path)
    return model, {"input_dim": int(input_dim), "output_dim": int(output_dim), "model_key": model_key}


def _reconstruct_with_trainer_utils(
    *,
    model: "trainer_mod.Model",
    output_dim: int,
    label: int,
    num_samples: int,
    steps: int,
    lr: float,
    noise_sigma: float,
    temp: float,
    init_kind: str,
    init_scale: float,
    batch_size: int,
    seed: int,
    out_dir: str,
    state_T: int = 0,
    state_lr: float = 0.05,
    save_npz: bool = False,
    save_png: bool = False,
) -> None:
    trainer_mod = _load_trainer_orchard()
    _ = trainer_mod.orchard_sun_reconstruct_mnist(
        model=model,
        label=label,
        num_samples=num_samples,
        steps=steps,
        lr=lr,
        noise_sigma=noise_sigma,
        temp=temp,
        init_kind=init_kind,
        init_scale=init_scale,
        batch_size=batch_size,
        seed=seed,
        state_T=state_T,
        state_lr=state_lr,
        save_dir=out_dir,
        save_npz=save_npz,
        save_png=save_png,
        run_tag=os.path.basename(out_dir),
    )


def run(cfg: Config | None = None) -> None:
    """
    Notebook-friendly entrypoint.

    Usage in a Jupyter notebook:

    ```python
    from orchard_sun_mnist_train import Config, run

    cfg = Config(
        out_dir="./runs/os_mnist",
        seed=0,
        untie_feedback_weights=True,
        l2_w=5e-5,
        l2_x=5e-4,
        l2_h=5e-4,
        recon_label=3,
        recon_noise_sigma=0.07,
    )
    run(cfg)
    ```
    """
    if cfg is None:
        cfg = Config()

    model, info = train_pcn(cfg)

    # ---------------- Reconstructions / generation ----------------
    if cfg.do_reconstruct:
        # Use the in-memory trained model by default, so you can call reconstruction repeatedly.
        recon_model = model
        recon_out_dir = os.path.join(cfg.out_dir, "reconstructions", info["run_id"])
        _reconstruct_with_trainer_utils(
            model=recon_model,
            output_dim=info["output_dim"],
            label=cfg.recon_label,
            num_samples=cfg.recon_samples,
            steps=cfg.recon_steps,
            lr=cfg.recon_lr,
            noise_sigma=cfg.recon_noise_sigma,
            temp=cfg.recon_temp,
            init_kind=cfg.recon_init_kind,
            init_scale=cfg.recon_init_scale,
            batch_size=cfg.recon_batch_size,
            seed=cfg.seed,
            out_dir=recon_out_dir,
            state_T=cfg.recon_state_T,
            state_lr=cfg.recon_state_lr,
            save_npz=cfg.recon_save_npz,
            save_png=cfg.recon_save_png,
        )
        print(f"Saved reconstructions to: {recon_out_dir}")


# No CLI / __main__ entrypoint on purpose.
# This file is intended to be imported (or copied) into a notebook.
