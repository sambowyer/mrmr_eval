import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints

from .abstract_model import IrtModel


@IrtModel.register("gaussian_irt")
class GaussianIRT(IrtModel):
    """Gaussian heteroskedastic IRT (Balkir et al. 2026). 1PL -- no disc.

    mu_ij = sigma(theta_i - b_j)
    Y_ij ~ N(mu_ij, k_j * mu_ij * (1 - mu_ij))

    Effective discrimination for clustering: a = 1/sqrt(k).
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
        self.dims = 1  # always 1PL

    def export(self):
        return {
            "ability": pyro.param("loc_ability").data.tolist(),
            "scale_ability": pyro.param("scale_ability").data.tolist(),
            "diff": pyro.param("loc_diff").data.tolist(),
            "k": pyro.param("loc_k").data.tolist(),
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

        with pyro.plate("mu_theta_plate", 1):
            mu_theta = pyro.sample(
                "mu_theta",
                dist.Normal(
                    torch.tensor(0.0, device=self.device),
                    torch.tensor(1.0e1, device=self.device),
                ),
            )

        with pyro.plate("u_theta_plate", 1):
            u_theta = pyro.sample(
                "u_theta",
                dist.Gamma(
                    torch.tensor(1.0, device=self.device),
                    torch.tensor(1.0, device=self.device),
                ),
            )

        with pyro.plate("thetas", self.num_subjects, dim=-2, device=self.device):
            with pyro.plate("theta_dims", 1, dim=-1):
                ability = pyro.sample("theta", dist.Normal(mu_theta, 1.0 / u_theta))

        with pyro.plate("bs", self.num_items, dim=-2, device=self.device):
            with pyro.plate("bs_dims", 1, dim=-1):
                diff = pyro.sample("b", dist.Normal(mu_b, 1.0 / u_b))

        with pyro.plate("ks", self.num_items, device=self.device):
            k = pyro.sample("k", dist.Gamma(
                torch.tensor(2.0, device=self.device),
                torch.tensor(2.0, device=self.device),
            ))

        with pyro.plate("observe_data", obs.size(0)):
            mu = torch.sigmoid(ability[subjects].squeeze(-1) - diff[items].squeeze(-1))
            variance = k[items] * mu * (1.0 - mu) + 1e-8
            pyro.sample("obs", dist.Normal(mu, variance.sqrt()), obs=obs)

    def guide_hierarchical(self, subjects, items, obs):
        loc_mu_b_param = pyro.param("loc_mu_b", torch.zeros(1, device=self.device))
        scale_mu_b_param = pyro.param(
            "scale_mu_b",
            torch.ones(1, device=self.device),
            constraint=constraints.positive,
        )

        loc_mu_theta_param = pyro.param("loc_mu_theta", torch.zeros(1, device=self.device))
        scale_mu_theta_param = pyro.param(
            "scale_mu_theta",
            torch.ones(1, device=self.device),
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
            torch.ones(1, device=self.device),
            constraint=constraints.positive,
        )
        beta_theta_param = pyro.param(
            "beta_theta",
            torch.ones(1, device=self.device),
            constraint=constraints.positive,
        )

        m_theta_param = pyro.param(
            "loc_ability", torch.zeros([self.num_subjects, 1], device=self.device)
        )
        s_theta_param = pyro.param(
            "scale_ability",
            torch.ones([self.num_subjects, 1], device=self.device),
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

        alpha_k_param = pyro.param(
            "alpha_k",
            2.0 * torch.ones(self.num_items, device=self.device),
            constraint=constraints.positive,
        )
        beta_k_param = pyro.param(
            "beta_k",
            2.0 * torch.ones(self.num_items, device=self.device),
            constraint=constraints.positive,
        )
        loc_k_param = pyro.param(
            "loc_k",
            torch.ones(self.num_items, device=self.device),
            constraint=constraints.positive,
        )

        with pyro.plate("mu_b_plate", 1):
            pyro.sample("mu_b", dist.Normal(loc_mu_b_param, scale_mu_b_param))

        with pyro.plate("u_b_plate", 1):
            pyro.sample("u_b", dist.Gamma(alpha_b_param, beta_b_param))

        with pyro.plate("mu_theta_plate", 1):
            pyro.sample("mu_theta", dist.Normal(loc_mu_theta_param, scale_mu_theta_param))

        with pyro.plate("u_theta_plate", 1):
            pyro.sample("u_theta", dist.Gamma(alpha_theta_param, beta_theta_param))

        with pyro.plate("thetas", self.num_subjects, dim=-2, device=self.device):
            with pyro.plate("theta_dims", 1, dim=-1):
                pyro.sample("theta", dist.Normal(m_theta_param, s_theta_param))

        with pyro.plate("bs", self.num_items, dim=-2, device=self.device):
            with pyro.plate("bs_dims", 1, dim=-1):
                pyro.sample("b", dist.Normal(m_b_param, s_b_param))

        with pyro.plate("ks", self.num_items, device=self.device):
            pyro.sample("k", dist.Gamma(alpha_k_param, beta_k_param))
