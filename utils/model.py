"""BoTorch / GPyTorch model helpers for reliability-constrained BO."""

import numpy as np


try:
    import botorch
    import gpytorch
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models.gp_regression import SingleTaskGP
    from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
    from gpytorch.mlls.exact_marginal_log_likelihood import (
        ExactMarginalLogLikelihood,
    )
except ImportError:  # pragma: no cover - exercised only in minimal envs
    botorch = None
    gpytorch = None
    torch = None
    fit_gpytorch_mll = None
    SingleTaskGP = None
    SingleTaskMultiFidelityGP = None
    ExactMarginalLogLikelihood = None


def _require_bo_deps():
    if torch is None or gpytorch is None or botorch is None:
        raise ImportError(
            "BoTorch, GPyTorch, and PyTorch are required for model fitting. "
            "Install the dependencies in requirements.txt before running the "
            "full source-influence analysis."
        )


if gpytorch is not None:

    class MultivariateQuadraticMean(gpytorch.means.Mean):
        """Quadratic mean module used by the original SingleTaskGP models."""

        def __init__(self, input_dim=3, batch_shape=torch.Size(), bias=True):
            super().__init__()
            self.register_parameter(
                name="second",
                parameter=torch.nn.Parameter(
                    torch.randn(*batch_shape, input_dim, 1, dtype=torch.float64)
                ),
            )
            self.register_parameter(
                name="first",
                parameter=torch.nn.Parameter(
                    torch.randn(*batch_shape, input_dim, 1, dtype=torch.float64)
                ),
            )
            if bias:
                self.register_parameter(
                    name="bias",
                    parameter=torch.nn.Parameter(
                        torch.randn(*batch_shape, 1, dtype=torch.float64)
                    ),
                )
            else:
                self.bias = None

        def forward(self, x):
            res = x.pow(2).matmul(self.second).squeeze(-1) + x.matmul(
                self.first
            ).squeeze(-1)
            if self.bias is not None:
                res = res + self.bias.squeeze(-1)
            return res


    class GPModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
        """Compatibility class from the original scripts."""

        _num_outputs = 1

        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = MultivariateQuadraticMean(
                input_dim=train_x.shape[-1]
            )
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(
                    nu=2.5, ard_num_dims=train_x.shape[-1]
                )
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

else:

    class MultivariateQuadraticMean(object):  # pragma: no cover
        def __init__(self, *args, **kwargs):
            _require_bo_deps()


    class GPModel(object):  # pragma: no cover
        def __init__(self, *args, **kwargs):
            _require_bo_deps()


def fit_gp_model(train_x, train_y, num_train_iters=1500):
    """Fit the original hand-trained exact GP compatibility model."""
    _require_bo_deps()
    noise = 1e-4
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(train_x, train_y, likelihood)
    model.likelihood.noise = noise

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()
    for _ in range(num_train_iters):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()
    return model, likelihood


def fit_stgp_model(train_x, train_y):
    """Fit a BoTorch SingleTaskGP with the original quadratic mean."""
    _require_bo_deps()
    model = SingleTaskGP(
        train_x,
        train_y,
        mean_module=MultivariateQuadraticMean(input_dim=train_x.shape[-1]),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model, mll


def fit_MultiFidelity_gp_model(train_x, train_yf, data_fidelity=3):
    """Fit the multi-fidelity GP used by the original BO scripts."""
    _require_bo_deps()
    torch.set_default_dtype(torch.double)
    model = SingleTaskMultiFidelityGP(
        train_x, train_yf, data_fidelity=data_fidelity
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model, mll


def build_qnehvi_policy(model, ref_point, train_y, x_baseline):
    """Build qNEHVI with compatibility across BoTorch versions."""
    _require_bo_deps()
    from botorch.acquisition.multi_objective.monte_carlo import (
        qNoisyExpectedHypervolumeImprovement,
    )
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        NondominatedPartitioning,
    )

    ref_list = ref_point.detach().cpu().numpy().tolist()
    try:
        return qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_list,
            X_baseline=x_baseline,
        )
    except TypeError:
        return qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            partitioning=NondominatedPartitioning(ref_point, train_y),
            X_baseline=x_baseline,
        )


def evaluate_acquisition(policy, x_points):
    """Evaluate an acquisition function point-by-point with shape fallback."""
    _require_bo_deps()
    values = []
    for point in x_points:
        try:
            value = policy(point.view(1, 1, -1))
        except Exception:
            value = policy(point.view(1, -1))
        values.append(float(value.detach().cpu().item()))
    return np.asarray(values, dtype=float)
