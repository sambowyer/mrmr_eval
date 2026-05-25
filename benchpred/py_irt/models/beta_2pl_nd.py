import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints

from .abstract_model import IrtModel


@IrtModel.register("beta_2pl_nd")
class Beta2PLND(IrtModel):
    """Beta 2PL-ND model (Chen et al. 2019, extending Noel & Dauvier 2007).

    Y_ij ~ Beta(mu_ij, 1/mu_ij) where mu_ij = exp(z_ij / 2)
    and z_ij = sum_d gamma_{j,d} * theta_{i,d} - b_j.

    E[Y_ij] = sigma(z_ij), so the sigmoid item_curve remains valid.
    """

    def __init__(
        self,
        *,
        num_items: int,
        num_subjects: int,
        dims: int = 2,
        verbose=False,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            device=device, num_items=num_items, num_subjects=num_subjects, verbose=verbose
        )
        self.dims = dims

    def export(self):
        return {
            "ability": pyro.param("loc_ability").data.tolist(),
            "scale_ability": pyro.param("scale_ability").data.tolist(),
            "diff": pyro.param("loc_diff").data.tolist(),
            "disc": pyro.param("loc_disc").data.tolist(),
            "loc_mu_theta": pyro.param("loc_mu_theta").data.tolist(),
            "scale_mu_theta": pyro.param("scale_mu_theta").data.tolist(),
            "alpha_theta": pyro.param("alpha_theta").data.tolist(),
            "beta_theta": pyro.param("beta_theta").data.tolist(),
        }

    def get_model(self):
        return self.model_hierarchical

    def get_guide(self):
        return self.guide_hierarchical

    def model_hierarchical(self, subjects, items, obs):
        with pyro.plate("mu_b_plate", 1):
            mu_b = pyro.sample(
                "mu_b",
                dist.Normal(
                    torch.tensor(0.0, device=self.device),
                    torch.tensor(1.0e1, device=self.device),
                ),
            )

        with pyro.plate("u_b_plate", 1):
            u_b = pyro.sample(
                "u_b",
                dist.Gamma(
                    torch.tensor(1.0, device=self.device),
                    torch.tensor(1.0, device=self.device),
                ),
            )

        with pyro.plate("mu_theta_plate", self.dims):
            mu_theta = pyro.sample(
                "mu_theta",
                dist.Normal(
                    torch.tensor(0.0, device=self.device),
                    torch.tensor(1.0e1, device=self.device),
                ),
            )

        with pyro.plate("u_theta_plate", self.dims):
            u_theta = pyro.sample(
                "u_theta",
                dist.Gamma(
                    torch.tensor(1.0, device=self.device),
                    torch.tensor(1.0, device=self.device),
                ),
            )

        with pyro.plate("mu_gamma_plate", self.dims):
            mu_gamma = pyro.sample(
                "mu_gamma",
                dist.Normal(
                    torch.tensor(0.0, device=self.device),
                    torch.tensor(1.0e1, device=self.device),
                ),
            )

        with pyro.plate("u_gamma_plate", self.dims):
            u_gamma = pyro.sample(
                "u_gamma",
                dist.Gamma(
                    torch.tensor(1.0, device=self.device),
                    torch.tensor(1.0, device=self.device),
                ),
            )

        with pyro.plate("thetas", self.num_subjects, dim=-2, device=self.device):
            with pyro.plate("theta_dims", self.dims, dim=-1):
                ability = pyro.sample("theta", dist.Normal(mu_theta, 1.0 / u_theta))

        with pyro.plate("bs", self.num_items, dim=-2, device=self.device):
            with pyro.plate("bs_dims", 1, dim=-1):
                diff = pyro.sample("b", dist.Normal(mu_b, 1.0 / u_b))

        with pyro.plate("gammas", self.num_items, dim=-2, device=self.device):
            with pyro.plate("gamma_dims", self.dims, dim=-1):
                disc = pyro.sample("gamma", dist.Normal(mu_gamma, 1.0 / u_gamma))

        with pyro.plate("observe_data", obs.size(0)):
            linear = (disc[items] * ability[subjects] - diff[items]).sum(dim=-1)
            linear = torch.clamp(linear, -20, 20)
            mu = torch.exp(linear / 2.0)
            alpha = mu + 1e-6
            beta_param = (1.0 / mu) + 1e-6
            pyro.sample("obs", dist.Beta(alpha, beta_param), obs=obs)

    def guide_hierarchical(self, subjects, items, obs):
        loc_mu_b_param = pyro.param("loc_mu_b", torch.zeros(1, device=self.device))
        scale_mu_b_param = pyro.param(
            "scale_mu_b",
            torch.ones(1, device=self.device),
            constraint=constraints.positive,
        )

        loc_mu_theta_param = pyro.param("loc_mu_theta", torch.zeros(self.dims, device=self.device))
        scale_mu_theta_param = pyro.param(
            "scale_mu_theta",
            torch.ones(self.dims, device=self.device),
            constraint=constraints.positive,
        )

        loc_mu_gamma_param = pyro.param("loc_mu_gamma", torch.zeros(self.dims, device=self.device))
        scale_mu_gamma_param = pyro.param(
            "scale_mu_gamma",
            torch.ones(self.dims, device=self.device),
            constraint=constraints.positive,
        )

        alpha_b_param = pyro.param(
            "alpha_b",
            torch.ones(1, device=self.device),
            constraint=constraints.positive,
        )
        beta_b_param = pyro.param(
            "beta_b",
            torch.ones(1, device=self.device),
            constraint=constraints.positive,
        )

        alpha_theta_param = pyro.param(
            "alpha_theta",
            torch.ones(self.dims, device=self.device),
            constraint=constraints.positive,
        )
        beta_theta_param = pyro.param(
            "beta_theta",
            torch.ones(self.dims, device=self.device),
            constraint=constraints.positive,
        )

        alpha_gamma_param = pyro.param(
            "alpha_gamma",
            torch.ones(self.dims, device=self.device),
            constraint=constraints.positive,
        )
        beta_gamma_param = pyro.param(
            "beta_gamma",
            torch.ones(self.dims, device=self.device),
            constraint=constraints.positive,
        )

        m_theta_param = pyro.param(
            "loc_ability", torch.zeros([self.num_subjects, self.dims], device=self.device)
        )
        s_theta_param = pyro.param(
            "scale_ability",
            torch.ones([self.num_subjects, self.dims], device=self.device),
            constraint=constraints.positive,
        )

        m_b_param = pyro.param(
            "loc_diff", torch.zeros([self.num_items, 1], device=self.device)
        )
        s_b_param = pyro.param(
            "scale_diff",
            torch.ones([self.num_items, 1], device=self.device),
            constraint=constraints.positive,
        )

        m_gamma_param = pyro.param(
            "loc_disc", torch.zeros([self.num_items, self.dims], device=self.device)
        )
        s_gamma_param = pyro.param(
            "scale_disc",
            torch.ones([self.num_items, self.dims], device=self.device),
            constraint=constraints.positive,
        )

        with pyro.plate("mu_b_plate", 1):
            pyro.sample("mu_b", dist.Normal(loc_mu_b_param, scale_mu_b_param))

        with pyro.plate("u_b_plate", 1):
            pyro.sample("u_b", dist.Gamma(alpha_b_param, beta_b_param))

        with pyro.plate("mu_theta_plate", self.dims):
            pyro.sample("mu_theta", dist.Normal(loc_mu_theta_param, scale_mu_theta_param))

        with pyro.plate("u_theta_plate", self.dims):
            pyro.sample("u_theta", dist.Gamma(alpha_theta_param, beta_theta_param))

        with pyro.plate("mu_gamma_plate", self.dims):
            pyro.sample("mu_gamma", dist.Normal(loc_mu_gamma_param, scale_mu_gamma_param))

        with pyro.plate("u_gamma_plate", self.dims):
            pyro.sample("u_gamma", dist.Gamma(alpha_gamma_param, beta_gamma_param))

        with pyro.plate("thetas", self.num_subjects, dim=-2, device=self.device):
            with pyro.plate("theta_dims", self.dims, dim=-1):
                pyro.sample("theta", dist.Normal(m_theta_param, s_theta_param))

        with pyro.plate("bs", self.num_items, dim=-2, device=self.device):
            with pyro.plate("bs_dims", 1, dim=-1):
                pyro.sample("b", dist.Normal(m_b_param, s_b_param))

        with pyro.plate("gammas", self.num_items, dim=-2, device=self.device):
            with pyro.plate("gamma_dims", self.dims, dim=-1, device=self.device):
                pyro.sample("gamma", dist.Normal(m_gamma_param, s_gamma_param))
