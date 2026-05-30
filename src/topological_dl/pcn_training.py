"""Training and optimization loops for predictive-coding models."""

from __future__ import annotations

try:
    from .pcn_backend import USE_PCX2, jax, jnp, jtu, optax, np, pxc, pxf, pxnn, pxu
    from .pcn_model import Model, energy, forward
except ImportError:
    from pcn_backend import USE_PCX2, jax, jnp, jtu, optax, np, pxc, pxf, pxnn, pxu
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
    if not USE_PCX2:
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


    params = pxu.M(pxnn.LayerParam)(model)

    # Label parameters for different weight decay
    labels_raw = pxu.label_params_for_weight_decay(
        params,
        forward_label="forward",
        feedback_label="feedback",
        default_label="none"
    )

    import jax.tree_util as jtu
    from pcx2.core._parameter import BaseParam
    from pcx2.core._module import BaseModule
    
    def _to_dict_structure(obj):
        """Convert Model/Module instances to dicts to make them non-callable."""
        if isinstance(obj, BaseModule):
            # Convert Module to dict by accessing __dict__
            return {k: _to_dict_structure(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, (list, tuple)):
            # Convert lists/tuples recursively
            return type(obj)(_to_dict_structure(item) for item in obj)
        elif isinstance(obj, dict):
            # Convert dicts recursively
            return {k: _to_dict_structure(v) for k, v in obj.items()}
        else:
            # For everything else (strings, numbers, etc.), return as-is
            return obj
    
    # Convert the labeled tree to a pure dict structure (not a Model instance)
    labels_full = _to_dict_structure(labels_raw)
    
    # Create a callable that filters labels to match filtered parameters
    # This is needed because when pxu.M(pxnn.LayerParam) filters the model,
    # the labels structure must match the filtered parameters structure
    def get_filtered_labels(params):
        """Filter labels to match the structure of filtered parameters.
        
        When params is filtered (e.g., by pxu.M(pxnn.LayerParam)), non-matching
        parameters are set to None. We need to return labels that match this structure,
        keeping labels only where params is not None.
        
        The key insight: labels_full has the same structure as the full model, with
        strings ('forward', 'feedback', 'none') at BaseParam positions. When params
        is filtered, we need to return a pytree with the same structure as params,
        where labels are preserved only where params is not None.
        
        Returns a pure dict/list structure (not a Model) to avoid callable issues.
        """
        def map_fn(label, param):
            # If param is None (filtered out), return None for the label
            if param is None:
                return None
            # If param is a BaseParam (like LayerParam), return the corresponding label string
            # The label should be a string ('forward', 'feedback', or 'none')
            if isinstance(param, BaseParam):
                # label should be a string at this position in labels_full
                if isinstance(label, str):
                    return label
                else:
                    # Fallback: if label is not a string (shouldn't happen), default to 'none'
                    return 'none'
            # For non-BaseParam nodes, preserve the structure
            # Since params is filtered, most non-BaseParam nodes should be None,
            # but we preserve the label structure for consistency
            return label
        
        # Map labels_full (dict structure) and params together, keeping labels where params is not None
        # Both should have the same structure since labels_full was created from the same model
        # that params was filtered from
        filtered_labels = jtu.tree_map(
            map_fn,
            labels_full,
            params,
            is_leaf=lambda x: isinstance(x, BaseParam) or x is None
        )
        
        # Ensure the result is also a pure dict structure (not a Model)
        # This handles the case where params might still have Model structure
        filtered_labels = _to_dict_structure(filtered_labels)
        
        return filtered_labels
    
    tx = optax.multi_transform(
        {
            'forward': optax.adamw(
                learning_rate=optax.exponential_decay(
                    init_value=init_w,
                    transition_steps=transition_steps,
                    decay_rate=decay_rate,
                    staircase=False,
                ),
                weight_decay=model.l2_w.get()
            ),  # Primary weight decay
            'feedback': optax.adamw(
                learning_rate=optax.exponential_decay(
                    init_value=init_w,
                    transition_steps=transition_steps,
                    decay_rate=decay_rate,
                    staircase=False,
                ),
                weight_decay=model.l2_w.get() / 2
            ),  # Half decay on feedback weights
            'none': optax.adamw(
                learning_rate=optax.exponential_decay(
                    init_value=init_w,
                    transition_steps=transition_steps,
                    decay_rate=decay_rate,
                    staircase=False,
                ),
                weight_decay=0.0
            ),  # No decay on bias
        },
        get_filtered_labels
    )
    
    optim_w = pxu.Optim(lambda: optax.chain(tx), params)
    
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
