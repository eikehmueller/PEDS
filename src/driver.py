import torch

from peds.diffusion_model import DiffusionModel1d, tridiagonal_apply


torch.set_default_dtype(torch.float64)

rng = torch.Generator()
rng.manual_seed(25157)

batchsize = 4
n = 8
alpha = torch.rand((batchsize, 5, n), generator=rng, requires_grad=True)
dalpha = torch.rand((batchsize, 5, n), generator=rng, requires_grad=True)

f_rhs = torch.rand(n, generator=rng)
K = torch.exp(alpha)
model = DiffusionModel1d(f_rhs)
u = model(alpha)
f = tridiagonal_apply(K, u)
error = torch.norm(f - f_rhs)
print(f"error = {error.detach().numpy():8.3e}")
loss = torch.sum(u**2)
loss.backward()
grad = torch.sum(alpha.grad * dalpha)

epsilon = 1.0e-8

uprime = model(alpha + epsilon * dalpha)
lossprime = torch.sum(uprime**2)
grad_numerical = (lossprime - loss) / epsilon

print(grad)
print(grad_numerical)
