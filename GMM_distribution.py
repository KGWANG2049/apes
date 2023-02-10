import torch
from abc import ABC
import torch.distributions as td
from mp2d.scripts.path_gen import path
# Define multiple multivariate Gaussian distributions


mean1 = torch.tensor(path.solution_path)
cov1 = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
mvn1 = td.MultivariateNormal(mean1, cov1)

mean2 = torch.tensor([3.0, 3.0])
cov2 = torch.tensor([[1.0, -0.5], [-0.5, 1.0]])
mvn2 = td.MultivariateNormal(mean2, cov2)

mean3 = torch.tensor([-3.0, -3.0])
cov3 = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
mvn3 = td.MultivariateNormal(mean3, cov3)

# Define the weights for each multivariate Gaussian
weights = torch.tensor([0.33, 0.33, 0.34])
cat = td.Categorical(weights)


# Define the Gaussian Mixture Model
class GMM(td.Distribution, ABC):
    def __init__(self, mvns, categ):
        super().__init__()
        self.mvns = mvns
        self.categ = categ
        self.num_components = len(mvns)

    def sample(self, sample_shape=torch.Size()):
        component = self.categ.sample(sample_shape).unsqueeze(-1)
        index = component.repeat(1, self.mvns[0].mean.numel())
        samples = torch.stack([mvn.sample() for mvn in self.mvns], dim=-1)
        return samples.gather(-1, index).squeeze(-1)

    def log_prob(self, value):
        log_probs = torch.stack([mvn.log_prob(value) for mvn in self.mvns], dim=-1)
        return torch.logsumexp(log_probs + self.categ.log_prob(torch.arange(self.num_components)), dim=-1)


gmm = GMM([mvn1, mvn2, mvn3], cat)
