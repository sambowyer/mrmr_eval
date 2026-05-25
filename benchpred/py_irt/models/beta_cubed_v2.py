import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints

from .abstract_model import IrtModel


@IrtModel.register("beta_cubed_v2")
class BetaCubedV2(IrtModel):
    """Beta-cubed (beta^3) model (Chen et al. 2019) – v2 variant.

    Differs from the v1 Pyro implementation only in the variational
    posterior family: uses logit-normal posteriors for theta and delta
    (Normal in unconstrained space, pushed through Sigmoid) instead of
    Beta posteriors. This matches the TensorFlow reference implementation:
    https://github.com/yc14600/beta3_IRT/blob/master/models/beta_irt.py

    Inherently 1-dimensional.
    theta_i ~ Beta(1,1),  delta_j ~ Beta(1,1),  a_j ~ N(1,1)
    alpha_ij = (theta_i / delta_j)^a_j
    beta_ij  = ((1-theta_i) / (1-delta_j))^a_j
    Y_ij ~ Beta(alpha_ij, beta_ij)

    E[Y_ij] = sigma(a_j * (logit(theta_i) - logit(delta_j)))
    """

    def __init__(
        self,
        *,
        num_items: int,
        num_subjects: int,
        dims: int = 1,
        verbose=False,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            device=device, num_items=num_items, num_subjects=num_subjects, verbose=verbose
        )
        self.dims = 1

    def export(self):
        theta_loc = pyro.param("theta_loc").data
        theta_scale = pyro.param("theta_scale").data
        theta_mean = torch.sigmoid(theta_loc)
        theta_logit = theta_loc

        delta_loc = pyro.param("delta_loc").data
        delta_scale = pyro.param("delta_scale").data
        delta_mean = torch.sigmoid(delta_loc)
        delta_logit = delta_loc

        a_loc = pyro.param("loc_a").data

        disc = a_loc.unsqueeze(-1)
        diff = (a_loc * delta_logit).unsqueeze(-1)
        ability = theta_logit.unsqueeze(-1)

        return {
            "disc": disc.tolist(),
            "diff": diff.tolist(),
            "ability": ability.tolist(),
        }

    def get_model(self):
        return self.model

    def get_guide(self):
        return self.guide

    def model(self, subjects, items, obs):
        with pyro.plate("thetas", self.num_subjects, device=self.device):
            theta = pyro.sample("theta", dist.Beta(
                torch.tensor(1.0, device=self.device),
                torch.tensor(1.0, device=self.device),
            ))

        with pyro.plate("deltas", self.num_items, device=self.device):
            delta = pyro.sample("delta", dist.Beta(
                torch.tensor(1.0, device=self.device),
                torch.tensor(1.0, device=self.device),
            ))

        with pyro.plate("as", self.num_items, device=self.device):
            a = pyro.sample("a", dist.Normal(
                torch.tensor(1.0, device=self.device),
                torch.tensor(1.0, device=self.device),
            ))

        with pyro.plate("observe_data", obs.size(0)):
            t = theta[subjects].clamp(1e-6, 1 - 1e-6)
            d = delta[items].clamp(1e-6, 1 - 1e-6)
            log_alpha = a[items] * (torch.log(t) - torch.log(d))
            log_beta = a[items] * (torch.log(1 - t) - torch.log(1 - d))
            alpha_val = torch.exp(log_alpha.clamp(-20, 20)) + 1e-6
            beta_val = torch.exp(log_beta.clamp(-20, 20)) + 1e-6
            pyro.sample("obs", dist.Beta(alpha_val, beta_val), obs=obs)

    def guide(self, subjects, items, obs):
        theta_loc = pyro.param(
            "theta_loc",
            torch.zeros(self.num_subjects, device=self.device),
        )
        theta_scale = pyro.param(
            "theta_scale",
            torch.ones(self.num_subjects, device=self.device),
            constraint=constraints.positive,
        )

        delta_loc = pyro.param(
            "delta_loc",
            torch.zeros(self.num_items, device=self.device),
        )
        delta_scale = pyro.param(
            "delta_scale",
            torch.ones(self.num_items, device=self.device),
            constraint=constraints.positive,
        )

        loc_a = pyro.param("loc_a", torch.ones(self.num_items, device=self.device))
        scale_a = pyro.param(
            "scale_a",
            torch.ones(self.num_items, device=self.device),
            constraint=constraints.positive,
        )

        with pyro.plate("thetas", self.num_subjects, device=self.device):
            pyro.sample(
                "theta",
                dist.TransformedDistribution(
                    dist.Normal(theta_loc, theta_scale),
                    dist.transforms.SigmoidTransform(),
                ),
            )

        with pyro.plate("deltas", self.num_items, device=self.device):
            pyro.sample(
                "delta",
                dist.TransformedDistribution(
                    dist.Normal(delta_loc, delta_scale),
                    dist.transforms.SigmoidTransform(),
                ),
            )

        with pyro.plate("as", self.num_items, device=self.device):
            pyro.sample("a", dist.Normal(loc_a, scale_a))
