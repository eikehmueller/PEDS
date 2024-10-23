import torch

__all__ = ["tridiagonal_apply", "tridiagonal_solve", "DiffusionModel1d"]


def tridiagonal_apply(K_diff, u):
    """Apply tridiagonal matrix to compute v = A(K) u

    K and u can be higher dimensional tensors, the solve is batched
    over all dimensions except the final dimension.

    :arg K_diff: tensor representing diffusion coefficient K(x)
    :arg u: tensor that A(K) is applied to
    """
    n = K_diff.shape[-1]
    v = torch.empty(K_diff.shape)
    v[..., 0] = (2 * K_diff[..., 0] + K_diff[..., 1]) * u[..., 0] - K_diff[..., 1] * u[
        ..., 1
    ]
    for j in range(1, n - 1):
        v[..., j] = (
            -K_diff[..., j] * u[..., j - 1]
            + (K_diff[..., j] + K_diff[..., j + 1]) * u[..., j]
            - K_diff[..., j + 1] * u[..., j + 1]
        )
    v[..., n - 1] = K_diff[..., n - 1] * (u[..., n - 1] - u[..., n - 2])
    return v


def tridiagonal_solve(K_diff, f_rhs):
    """Tridiagonal solve of system A(K) u = f_rhs

    K and f can be higher dimensional tensors, the solve is batched
    over all dimensions except the final dimension.

    :arg K_diff: tensor representing diffusion coefficient K(x)
    :arg f_rhs: tensor representing right hand side f(x)
    """
    n = K_diff.shape[-1]
    c = torch.empty(K_diff.shape)
    u = torch.empty(K_diff.shape)
    c[..., 0] = -K_diff[..., 1] / (2 * K_diff[..., 0] + K_diff[..., 1])
    u[..., 0] = f_rhs[..., 0] / (2 * K_diff[..., 0] + K_diff[..., 1])
    # forward sweep
    for j in range(1, n - 1):
        c[..., j] = -K_diff[..., j + 1] / (
            K_diff[..., j] + K_diff[..., j + 1] + K_diff[..., j] * c[..., j - 1]
        )
        u[..., j] = (f_rhs[..., j] + K_diff[..., j] * u[..., j - 1]) / (
            K_diff[..., j] + K_diff[..., j + 1] + K_diff[..., j] * c[..., j - 1]
        )
    u[..., n - 1] = (f_rhs[..., n - 1] + K_diff[..., n - 1] * u[..., n - 2]) / (
        K_diff[..., n - 1] + K_diff[..., n - 1] * c[..., n - 2]
    )
    # backward sweep
    for j in range(n - 2, -1, -1):
        u[..., j] -= c[..., j] * u[..., j + 1]
    return u


class DiffusionModel1dOperator(torch.autograd.Function):
    """Differentiable function which solves the diffusion equation

    The diffusion equation is

        -div(K(x)grad(u)) = f(x)

    with u(0) = 0, K(1) du/dx(1) = g and K(x) = exp(alpha(x)).

    We consider a finite difference discretisation in 1d which leads to the
    matrix-vector equation

        A(alpha) u = f

    where A(alpha) is tridiagonal. For given f, this class provides the
    differentiable mapping

    alpha -> u = A(alpha)^{-1} f

    """

    def __init__(self):
        """Construct a new instance"""
        super().__init__()

    @staticmethod
    def forward(ctx, metadata, input):
        """Forward pass, compute u = u(alpha)

        :arg ctx: instance of object
        :arg metadata: metadata, contains information on the RHS
        :arg input: tensor containing alpha"""
        f_rhs = metadata["f_rhs"]
        K_diff = torch.exp(input)
        ctx.metadata.update(metadata)
        u = tridiagonal_solve(K_diff, f_rhs)
        ctx.save_for_backward(K_diff, u)
        return u

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass, compute dL/dalpha given dL/du

        :arg grad_output: dL/du
        """
        n = grad_output.shape[-1]
        K_diff, u = ctx.saved_tensors
        # compute w such that A(alpha) w = grad_output
        w = tridiagonal_solve(K_diff, grad_output)
        grad_input = torch.empty(grad_output.shape)
        grad_input[..., 0] = -2 * K_diff[..., 0] * u[..., 0] * w[..., 0]
        for j in range(1, n):
            grad_input[..., j] = -K_diff[..., j] * (
                u[..., j - 1] * w[..., j - 1]
                + u[..., j] * w[..., j]
                - u[..., j - 1] * w[..., j]
                - u[..., j] * w[..., j - 1]
            )
        return None, grad_input


class DiffusionModel1d(torch.nn.Module):

    def __init__(self, f_rhs):
        """Initialise a new instance

        :arg f_rhs: 1d tensor representing the right hand side"""
        super().__init__()
        self.metadata = dict(f_rhs=f_rhs)

    def forward(self, x):
        """Apply model

        Computation is batched over all but the final dimension

        :arg x: diffusion tensor alpha, can be multidimensional"""
        return DiffusionModel1dOperator.apply(self.metadata, x)
