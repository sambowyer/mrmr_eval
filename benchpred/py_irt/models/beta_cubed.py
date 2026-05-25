import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints

from .abstract_model import IrtModel


@IrtModel.register("beta_cubed")
class BetaCubed(IrtModel):
    """Beta-cubed (beta^3) model (Chen et al. 2019).

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
        self.dims = 1  # always 1D

    def export(self):
        theta_conc1 = pyro.param("theta_conc1").data
        theta_conc0 = pyro.param("theta_conc0").data
        theta_mean = theta_conc1 / (theta_conc1 + theta_conc0)
        theta_logit = torch.log(theta_mean.clamp(1e-6, 1 - 1e-6) / (1 - theta_mean.clamp(1e-6, 1 - 1e-6)))

        delta_conc1 = pyro.param("delta_conc1").data
        delta_conc0 = pyro.param("delta_conc0").data
        delta_mean = delta_conc1 / (delta_conc1 + delta_conc0)
        delta_logit = torch.log(delta_mean.clamp(1e-6, 1 - 1e-6) / (1 - delta_mean.clamp(1e-6, 1 - 1e-6)))

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
        theta_conc1 = pyro.param(
            "theta_conc1",
            torch.ones(self.num_subjects, device=self.device),
            constraint=constraints.positive,
        )
        theta_conc0 = pyro.param(
            "theta_conc0",
            torch.ones(self.num_subjects, device=self.device),
            constraint=constraints.positive,
        )

        delta_conc1 = pyro.param(
            "delta_conc1",
            torch.ones(self.num_items, device=self.device),
            constraint=constraints.positive,
        )
        delta_conc0 = pyro.param(
            "delta_conc0",
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
            pyro.sample("theta", dist.Beta(theta_conc1, theta_conc0))

        with pyro.plate("deltas", self.num_items, device=self.device):
            pyro.sample("delta", dist.Beta(delta_conc1, delta_conc0))

        with pyro.plate("as", self.num_items, device=self.device):
            pyro.sample("a", dist.Normal(loc_a, scale_a))
