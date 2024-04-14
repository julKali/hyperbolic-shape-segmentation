import torch
import geoopt

class EuclMLR(torch.nn.Module):

    def __init__(self, plane_shape: int, num_planes: int, signed: bool = True):
        self.points = geoopt.ManifoldParameter(
            torch.empty(num_planes, plane_shape), manifold=self.ball
        )
        self.dirs = torch.nn.Parameter(
            torch.empty(num_planes, plane_shape)
        )

    def forward(x):
        pass


class Distance2PoincareHyperplanes(torch.nn.Module):
    n = 0
    # 1D, 2D versions of this class ara available with a one line change
    # class Distance2PoincareHyperplanes2d(Distance2PoincareHyperplanes):
    #     n = 2

    def __init__(
        self,
        plane_shape: int,
        num_planes: int,
        signed=True,
        squared=False,
        *,
        ball,
        std=1.0,
    ):
        super().__init__()
        self.signed = signed
        self.squared = squared
        # Do not forget to save Manifold instance to the Module
        self.ball = ball
        self.plane_shape = geoopt.utils.size2shape(plane_shape)
        self.num_planes = num_planes

        # In a layer we create Manifold Parameters in the same way we do it for
        # regular pytorch Parameters, there is no difference. But geoopt optimizer
        # will recognize the manifold and adjust to it
        self.points = geoopt.ManifoldParameter(
            torch.empty(num_planes, plane_shape), manifold=self.ball
        )
        self.dirs = torch.nn.Parameter(
            torch.empty(num_planes, plane_shape)
        )
        #self.dirs = geoopt.ManifoldParameter(
        #    torch.empty(num_planes, plane_shape), manifold=self.ball
        #)
        self.std = std
        # following best practives, a separate method to reset parameters
        self.reset_parameters()

    # input B Coord P = 16 1024 3
    # output B Class P = 16 1024 50
    def forward(self, input):                   
        #device = self.dirs.device
        z = self.ball.expmap0(input) # 16 1024 3
        #print("in", input.shape, z.shape)
        # z is (b,n,2)
        c = self.ball.c.item() # scalar
        p = self.points
        # see hyperbolic nn paper:
        # (self.ball.lambda_x(torch.zeros_like(p)) / self.ball.lambda_x(p)) is 50,
        w = (self.ball.lambda_x(torch.zeros_like(p)) / self.ball.lambda_x(p)).unsqueeze(-1) * self.dirs # 50, 3
        w = torch.nn.functional.normalize(w)

        #logits = self.ball.dist2plane(z[:,:,None,:], p[None,None,:,:], w[None,None,:,:], signed=True) # b,n,K
        #return logits

        p_hat = -p # 50 3
        z_norm_sq = z.pow(2).sum(dim=-1).unsqueeze(-1) # b,n,1
        p_norm_sq = p_hat.pow(2).sum(dim=-1).unsqueeze(-1) # K,1
        dot = z.inner(p_hat) # b,n,K
        denom = 1 + 2*c*dot + c**2*z_norm_sq.inner(p_norm_sq)
        alph = (1 + 2*c*dot + c*z_norm_sq) / denom
        beta = (1 - c*p_norm_sq).squeeze(-1) / denom

        inner_prod = alph * (p_hat*w).sum(-1) + beta * z.inner(w) # b,n,K
        add_sq = alph**2 * p_norm_sq.squeeze(-1) + 2*alph*beta*dot + beta**2 * z_norm_sq

        norm_w = w.pow(2).sum(dim=-1).sqrt() # K,
        sqrt_c = torch.tensor([c], device=norm_w.device).sqrt() # 1,
        inner = 2*sqrt_c*inner_prod / ((1-c*add_sq) * norm_w)
        logits = inner.asinh() * norm_w * self.ball.lambda_x(p) / sqrt_c # b,n,K

        if not self.signed:
            logits = logits.pow(2)

        return logits # b,n,K

    def extra_repr(self):
        return (
            "plane_shape={plane_shape}, "
            "num_planes={num_planes}, "
            .format(**self.__dict__)
        )

    @torch.no_grad()
    def reset_parameters(self):
        direction = torch.randn_like(self.points)
        direction /= direction.norm(dim=-1, keepdim=True)
        #distance = torch.empty_like(self.points[..., 0]).normal_(std=self.std)
        eps = 1e-3 # from https://arxiv.org/pdf/1705.08039.pdf
        if eps > self.ball.c.pow(-1).sqrt():
            raise NotImplementedError("init eps too small for curvature, maybe resize?")
        self.points.set_(torch.empty_like(self.points).normal_(-eps, eps))
        self.dirs.set_(direction)
