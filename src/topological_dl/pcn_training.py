"""Training and optimization loops for predictive-coding models."""

from __future__ import annotations

try:
    from .pcn_backend import jax, jnp, optax, np, pxc, pxf, pxnn, pxu
    from .pcn_model import Model, energy, forward
except ImportError:
    from pcn_backend import jax, jnp, optax, np, pxc, pxf, pxnn, pxu
    from pcn_model import Model, energy, forward


def train_on_batch(
    T: int,
    x: jax.Array,
    y: jax.Array,
    *,
    model: Model,
    optim_w: pxu.Optim,
    optim_h: pxu.Optim,
    print_energy: bool = False
):
    # This only sets an internal flag to be "train" (instead of "eval")
    model.train()

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model)

    # As it is explained later, we initialise the state optimizer for the current batch.
    # We specify to ignore the `VodeParams` which have the `frozen` attribute set to True.
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))

    # Inference steps
    for t in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (e, y_), g = pxf.value_and_grad(
                pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]),
                has_aux=True
            )(energy)(x, model=model)

        if print_energy:
            jax.debug.print("step {} energy {}", t, e)

        optim_h.step(model, g["model"])

    optim_h.clear()

    # Weight update step
    model.clear_params(pxc.VodeParam.Cache) # Clear cache before weight update
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        (e, y_), g = pxf.value_and_grad(pxu.M(pxnn.LayerParam).to([False, True]), has_aux=True)(energy)(x, model=model)

    # Since the energy function returns the sum of the energies over the batch dimension, we need to scale the
    # gradient according to the number of samples in the batch.
    optim_w.step(model, g["model"], scale_by=1.0/x.shape[0])
    

@pxf.jit()
def eval_on_batch(x: jax.Array, y: jax.Array, *, model: Model):
    model.eval()

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        y_ = forward(x, None, model=model).argmax(axis=-1)

    return (y_ == y).mean(), y_
    

# Standard training loop
def train(dl, T, *, model: Model, optim_w: pxu.Optim, optim_h: pxu.Optim):
    for i, (x, y) in enumerate(dl):
        x_flat = x.reshape(x.shape[0], -1)
        x_jax = jnp.asarray(x_flat)
        y_jax = jnp.asarray(y)
        train_on_batch(T, x_jax, jax.nn.one_hot(y_jax, model.output_dim.get()), model=model, optim_w=optim_w, optim_h=optim_h, print_energy=False)
        

# Standard evaluation loop
def eval(dl, *, model: Model):
    acc = []
    ys_ = []

    for x, y in dl:
        x_flat = x.reshape(x.shape[0], -1)
        x_jax = jnp.asarray(x_flat)
        y_jax = jnp.asarray(y)
        a, y_ = eval_on_batch(x_jax, y_jax, model=model)
        acc.append(a)
        ys_.append(y_)

    return np.mean(acc), np.concatenate(ys_)


def get_opts(model, init_w, init_h, transition_steps, decay_rate, T):
    optim_w = pxu.Optim(
        lambda: optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(model.l2_w.get()),
            optax.adamw(
                learning_rate=optax.exponential_decay(
                    init_value=init_w,
                    transition_steps=transition_steps,
                    decay_rate=decay_rate,
                    staircase=False,
                )
            )
        ),
        pxu.M(pxnn.LayerParam)(model)
    )

    optim_h = pxu.Optim(
        lambda: optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.sgd(
                learning_rate=optax.cosine_decay_schedule(
                    init_value=init_h,
                    decay_steps=T,
                    alpha=0.05,
                ),
                momentum=0.9,
                nesterov=True
            )
        )
    )

    return optim_w, optim_h
