""""""

import math
import gpytorch
import torch


class DeepKernel(torch.nn.Module):
    def __init__(self, n_dim: int = 6, num_features: int = 10, hidden: int = 20):
        super().__init__()
        self._n_dim = n_dim
        self.num_features = num_features
        self._hidden = hidden

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self._n_dim, self._hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self._hidden, self._hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self._hidden, self.num_features),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim: int, grid_bounds=(-10.0, 10.0), grid_size: int = 64):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )

        variational_strategy = (
            gpytorch.variational.IndependentMultitaskVariationalStrategy(
                gpytorch.variational.GridInterpolationVariationalStrategy(
                    self,
                    grid_size=grid_size,
                    grid_bounds=[grid_bounds],
                    variational_distribution=variational_distribution,
                ),
                num_tasks=num_dim,
            )
        )
        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class DKLModel(gpytorch.Module):
    def __init__(
        self,
        feature_extractor: torch.nn.Module = DeepKernel(),
        grid_bounds=(-10.0, 10.0),
    ):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.num_feat = feature_extractor.num_features
        self.gp_layer = GaussianProcessLayer(
            num_dim=self.num_feat, grid_bounds=grid_bounds
        )
        self.grid_bounds = grid_bounds

        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(
            self.grid_bounds[0], self.grid_bounds[1]
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.scale_to_bounds(features)
        features = features.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_layer(features)
        return res


def main():
    num_classes = 5
    feature_extractor = DeepKernel()

    model = DKLModel()
    likelihood = gpytorch.likelihoods.SoftmaxLikelihood(
        num_features=model.num_feat, num_classes=num_classes
    )

    # test for batch_size=16, input_dim=6
    test_vec = torch.rand(16, 6)
    print(feature_extractor(test_vec))
    test_out = likelihood(model(test_vec))
    print(test_out)
    test_preds = test_out.probs.mean(0).argmax(-1)
    print(test_preds.shape, test_preds[:5])
    print(test_out.probs.mean(0)[:5])
    print(test_out.probs.std(0)[:5])


if __name__ == "__main__":
    main()
