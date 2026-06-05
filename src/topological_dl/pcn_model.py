"""Predictive-coding model definition and energy functions."""

from __future__ import annotations

try:
    from .pcn_backend import jax, jnp, px, pxc, pxnn, pxf, pxu
except ImportError:
    from pcn_backend import jax, jnp, px, pxc, pxnn, pxf, pxu


class Model(pxc.EnergyModule):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims,
        act_fn,
        model_key,
        residual=False,
        l2_w=0.0,
        l2_x=0.0,
        l2_h=0.0
    ):
        super().__init__()

        self.input_dim = px.static(input_dim)
        self.output_dim = px.static(output_dim)
        self.act_fn = px.static(act_fn)
        self.residual = px.static(residual)
        self.l2_w = px.static(float(l2_w))  # weight decay strength
        self.l2_x = px.static(float(l2_x))  # input activity decay
        self.l2_h = px.static(float(l2_h))  # hidden activity decay

        model_seed = int(jax.random.randint(model_key, (), 0, 2**31 - 1))
        model_rkg = px.RandomKeyGenerator(model_seed)

        # Input projection layer
        self.input_layer = pxnn.Linear(input_dim, hidden_dims[0], rkg=model_rkg)

        # Hidden layers
        self.hidden_layers = []
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                pxnn.Linear(hidden_dims[i], hidden_dims[i + 1], rkg=model_rkg)
            )

        # Projection shortcuts for mismatched dims
        self.projections = []
        for i in range(len(hidden_dims) - 1):
            if residual and hidden_dims[i] != hidden_dims[i + 1]:
                # learnable projection
                self.projections.append(
                    pxnn.Linear(hidden_dims[i], hidden_dims[i + 1], rkg=model_rkg)
                )
            else:
                # no projection needed (identity)
                self.projections.append(None)

        # Output layer
        self.output_layer = pxnn.Linear(hidden_dims[-1], output_dim, rkg=model_rkg)

        # Layer normalization for each hidden layer
        self.layer_norms = []
        for dim in hidden_dims:
            self.layer_norms.append(pxnn.LayerNorm(dim))

        # Create Vodes for predictive coding
        # One for input projection, one for each hidden layer, one for output
        self.vodes = [
            pxc.Vode() for _ in range(len(hidden_dims) + 1)
        ]

        # Replace last vode with cross-entropy energy for classification
        self.vodes[-1] = pxc.Vode(pxc.ce_energy)
        self.vodes[-1].h.frozen = True

    def __call__(self, x, y):
        # Input projection
        x = self.input_layer(x)
        x = self.layer_norms[0](x)
        x = self.vodes[0](self.act_fn(x))

        # # Hidden layers with residual connections
        # for i, (layer, vode) in enumerate(zip(self.hidden_layers, self.vodes[1:-1])):
        #     if self.residual.get():
        #         residual = x

        #     x_new = layer(x)
        #     x_new = self.layer_norms[i + 1](x_new)

        #     x_activated = self.act_fn(x_new)

        #     # Add residual connection before vode
        #     if self.residual.get():
        #         x_activated = x_activated + residual

        #     x = vode(x_activated)

        # Hidden layers with residual connections + projection if needed
        for i, (layer, vode, proj) in enumerate(
            zip(self.hidden_layers, self.vodes[1:-1], self.projections)
        ):
            if self.residual.get():
                residual = x

            x_new = layer(x)
            x_new = self.layer_norms[i + 1](x_new)

            x_activated = self.act_fn(x_new)

            # Apply projection shortcut if dims mismatch
            if proj is not None:  # is always None if self.residual = False
                residual = proj(residual)

            # Add residual connection before vode
            if self.residual.get():
                x_activated = x_activated + residual

            x = vode(x_activated)

        # Output layer
        x = self.vodes[-1](self.output_layer(x))

        if y is not None:
            self.vodes[-1].set("h", y)

        return self.vodes[-1].get("u")


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model: Model):
    return model(x, y)
    

@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energy(x, *, model: Model):
    y_ = model(x, None)

    # base per-sample energy
    e = model.energy()  # scalar for this sample (because we're inside vmap)

    # regularization as per Orchard & Sun
    l2_h = model.l2_h.get()
    if l2_h > 0:
        h_pen = 0.0
        # exclude output layer
        for vode in model.vodes[:-1]:
            h = vode.get("h")
            h_pen = h_pen + jnp.sum(h * h)
        e = e + 0.5 * l2_h * h_pen
        
    return jax.lax.psum(e, "batch"), y_
