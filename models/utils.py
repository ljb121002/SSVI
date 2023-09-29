import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

class StoLayer(object):
    def sto_init(self, prior_mean, prior_std, posterior_mean_init, posterior_std_init, sigma_parameterization):
        self.posterior_std = nn.Parameter(torch.zeros_like(self.weight), requires_grad=True)
        nn.init.normal_(self.weight, posterior_mean_init[0], posterior_mean_init[1])
        nn.init.normal_(self.posterior_std, posterior_std_init[0], posterior_std_init[1])
        if sigma_parameterization == 'abs':
            self.posterior_std.data.abs_() #.expm1_().log_()
        elif sigma_parameterization == 'softplus':
            self.posterior_std.data.abs_().expm1_().log_()
            while torch.isinf(self.posterior_std).any():
                nn.init.normal_(self.posterior_std, posterior_std_init[0], posterior_std_init[1])
                self.posterior_std.data.abs_().expm1_().log_()
        elif sigma_parameterization == 'exp':
            self.posterior_std.data.abs_().log_()
            while torch.isinf(self.posterior_std).any():
                nn.init.normal_(self.posterior_std, posterior_std_init[0], posterior_std_init[1])
                self.posterior_std.data.abs_().log_()
        else:
            raise NotImplementedError

        self.test_with_mean = False

        self.prior_mean = nn.Parameter(torch.tensor(prior_mean), requires_grad=False)
        self.prior_std = nn.Parameter(torch.tensor(prior_std), requires_grad=False)
        self.sigma_parameterization = sigma_parameterization

    def kl(self):
        prior = D.Normal(self.prior_mean, self.prior_std)
        # assert ((self.weight == 0) != (self.posterior_std == 0)).sum() == 0
        if self.sigma_parameterization == 'abs':
            posterior = D.Normal(self.weight, torch.abs(self.posterior_std) + (self.posterior_std == 0))
        elif self.sigma_parameterization == 'softplus':
            posterior = D.Normal(self.weight, F.softplus(self.posterior_std))
        elif self.sigma_parameterization == 'exp':
            posterior = D.Normal(self.weight, self.posterior_std.exp())
        else:
            raise NotImplementedError
        kl = (D.kl_divergence(posterior, prior) * (self.posterior_std != 0)).sum()
        return kl

class StoModel(object):
    def kl(self):
        kl = 0
        for m in self.modules():
            if isinstance(m, StoLayer):
                kl += m.kl()
        return kl


class StoConv2d(nn.Conv2d, StoLayer):
    def __init__(self, in_planes, out_planes, kernel_size, \
                use_bnn, prior_mean, prior_std, posterior_mean_init, posterior_std_init, same_noise, \
                stride=1, padding=0, groups=1, bias=True, dilation=1, sigma_parameterization='softplus'):
        super(StoConv2d, self).__init__(in_planes, out_planes, kernel_size, stride, padding, dilation, groups, bias)
        self.use_bnn = use_bnn
        self.same_noise = same_noise
        if not self.use_bnn:
            self.same_noise = False
        if self.use_bnn:
            self.sto_init(prior_mean, prior_std, posterior_mean_init, posterior_std_init, sigma_parameterization)
        self.sigma_parameterization = sigma_parameterization
    
    def forward(self, x):
        if not self.same_noise:
            mean = super()._conv_forward(x, self.weight, None)
            if not self.use_bnn or self.test_with_mean:
                return mean
            else:
                if self.sigma_parameterization == 'abs':
                    std = torch.sqrt(super()._conv_forward(x**2, torch.abs(self.posterior_std)**2, None) + 1e-8)
                elif self.sigma_parameterization == 'softplus':
                    std = torch.sqrt(super()._conv_forward(x**2, F.softplus(self.posterior_std)**2 * (self.posterior_std != 0), None) + 1e-8)
                elif self.sigma_parameterization == 'exp':
                    std = torch.sqrt(super()._conv_forward(x**2, self.posterior_std.exp()**2 * (self.posterior_std != 0), None) + 1e-8)
                else:
                    raise NotImplementedError
                
                return mean + std * torch.randn_like(mean)
        else:
            if self.sigma_parameterization == 'abs':
                return super()._conv_forward(x, self.weight + torch.randn_like(self.weight) * torch.abs(self.posterior_std), None)
            elif self.sigma_parameterization == 'softplus':
                return super()._conv_forward(x, self.weight + torch.randn_like(self.weight) * F.softplus(self.posterior_std) * (self.posterior_std != 0), None)
            elif self.sigma_parameterization == 'exp':
                return super()._conv_forward(x, self.weight + torch.randn_like(self.weight) * self.posterior_std.exp() * (self.posterior_std != 0), None)

class StoLinear(nn.Linear, StoLayer):
    def __init__(self, in_features, out_features, \
                 use_bnn, prior_mean, prior_std, posterior_mean_init, posterior_std_init, same_noise, \
                 bias=True, sigma_parameterization='softplus'):
        super(StoLinear, self).__init__(in_features, out_features, bias)
        self.use_bnn = use_bnn
        self.same_noise = same_noise
        if not self.use_bnn:
            self.same_noise = False
        if self.use_bnn:
            self.sto_init(prior_mean, prior_std, posterior_mean_init, posterior_std_init, sigma_parameterization)
        self.sigma_parameterization = sigma_parameterization
    
    def forward(self, x):
        if not self.same_noise:
            mean = super().forward(x)
            if not self.use_bnn or self.test_with_mean:
                return mean
            else:
                # assert ((self.weight == 0) != (self.posterior_std == 0)).sum() == 0
                if self.sigma_parameterization == 'abs':
                    std = torch.sqrt(F.linear(x**2, torch.abs(self.posterior_std)**2) + 1e-8)
                elif self.sigma_parameterization == 'softplus':
                    std = torch.sqrt(F.linear(x**2, F.softplus(self.posterior_std)**2 * (self.posterior_std != 0)) + 1e-8)
                elif self.sigma_parameterization == 'exp':
                    std = torch.sqrt(F.linear(x**2, self.posterior_std.exp()**2 * (self.posterior_std != 0)) + 1e-8)
                else:
                    raise NotImplementedError
                return mean + std * torch.randn_like(mean)
        else:
            # assert ((self.weight == 0) != (self.posterior_std == 0)).sum() == 0
            if self.sigma_parameterization == 'abs':
                return F.linear(x, self.weight + torch.randn_like(self.weight) * torch.abs(self.posterior_std), self.bias)
            elif self.sigma_parameterization == 'softplus':
                return F.linear(x, self.weight + torch.randn_like(self.weight) * F.softplus(self.posterior_std) * (self.posterior_std != 0), self.bias)
            elif self.sigma_parameterization == 'exp':
                return F.linear(x, self.weight + torch.randn_like(self.weight) * self.posterior_std.exp() * (self.posterior_std != 0), self.bias)



def bnn_sample(model, args):
    for n, p in model.named_parameters():
        if n.endswith('weight') and n.replace('weight', 'posterior_std') in model.state_dict():
            posterior_std = model.state_dict()[n.replace('weight', 'posterior_std')].data
            assert ((p == 0) != (posterior_std == 0)).sum() == 0
            if args.sigma_parameterization == 'abs':
                p.data = p.data + torch.randn_like(p.data) * torch.abs(posterior_std) * (posterior_std != 0)
            elif args.sigma_parameterization == 'softplus':
                p.data = p.data + torch.randn_like(p.data) * F.softplus(posterior_std) * (posterior_std != 0)
            elif args.sigma_parameterization == 'exp':
                p.data = p.data + torch.randn_like(p.data) * posterior_std.exp() * (posterior_std != 0)
            else:
                raise NotImplementedError
                # assert ((p == 0) != (posterior_std == 0)).sum() == 0
    return model