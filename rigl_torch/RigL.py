import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import copy

from rigl_torch.util import get_W, get_grad

import wandb

def gaussian_cdf(x):
    return 0.5 * (1 + torch.erf(x / np.sqrt(2)))


EPS = torch.log(torch.exp(torch.tensor(1e-6)) - 1)
# EPS = -100.0

class IndexMaskHook:
    def __init__(self, layer, scheduler):
        self.layer = layer
        self.scheduler = scheduler
        self.dense_grad = None

    def __name__(self):
        return 'IndexMaskHook'

    @torch.no_grad()
    def __call__(self, grad):
        mask = self.scheduler.backward_masks[self.layer]

        # only calculate dense_grads when necessary
        if self.scheduler.check_if_backward_hook_should_accumulate_grad():
            if self.dense_grad is None:
                # initialize as all 0s so we can do a rolling average
                self.dense_grad = torch.zeros_like(grad)
            self.dense_grad += grad / self.scheduler.grad_accumulation_n
        else:
            self.dense_grad = None

        return grad * mask


def _create_step_wrapper(scheduler, optimizer):
    _unwrapped_step = optimizer.step
    def _wrapped_step():
        _unwrapped_step()
        scheduler.reset_momentum()
        scheduler.apply_mask_to_weights()
    optimizer.step = _wrapped_step


class RigLScheduler:

    def __init__(self, model, optimizer, dense_allocation=1, T_end=None, sparsity_distribution='uniform', ignore_linear_layers=True, delta=100, alpha=0.3, static_topo=False, grad_accumulation_n=1, state_dict=None, args=None):
        if dense_allocation <= 0 or dense_allocation > 1:
            raise Exception('Dense allocation must be on the interval (0, 1]. Got: %f' % dense_allocation)

        self.model = model
        self.optimizer = optimizer

        self.args = args

        self.W, self._linear_layers_mask = get_W(model, return_linear_layers_mask=True, use_bnn=args.use_bnn)

        # modify optimizer.step() function to call "reset_momentum" after
        _create_step_wrapper(self, optimizer)
            
        self.dense_allocation = dense_allocation
        self.N = [torch.numel(w) for w in self.W]

        if state_dict is not None:
            self.load_state_dict(state_dict)
            self.apply_mask_to_weights()

        else:
            self.sparsity_distribution = sparsity_distribution
            self.static_topo = static_topo
            self.grad_accumulation_n = grad_accumulation_n
            self.ignore_linear_layers = ignore_linear_layers
            self.backward_masks = None

            # define sparsity allocation
            self.S = []
            for i, (W, is_linear) in enumerate(zip(self.W, self._linear_layers_mask)):
                # when using uniform sparsity, the first layer is always 100% dense
                # UNLESS there is only 1 layer
                if not args.use_bnn:
                    is_first_layer = i == 0
                else:
                    is_first_layer = (i == 0 or i == 1)
                if is_first_layer and self.sparsity_distribution == 'uniform' and len(self.W) > 1:
                    self.S.append(0)

                elif is_linear and self.ignore_linear_layers:
                    # if choosing to ignore linear layers, keep them 100% dense
                    self.S.append(0)

                else:
                    self.S.append(1-dense_allocation)

            # randomly sparsify model according to S
            self.random_sparsify()

            # scheduler keeps a log of how many times it's called. this is how it does its scheduling
            self.step = 0
            self.rigl_steps = 0

            # define the actual schedule
            self.delta_T = delta
            self.alpha = alpha
            self.T_end = T_end

        # also, register backward hook so sparse elements cannot be recovered during normal training
        self.backward_hook_objects = []
        for i, w in enumerate(self.W):
            # if sparsity is 0%, skip
            if self.S[i] <= 0:
                self.backward_hook_objects.append(None)
                continue

            if getattr(w, '_has_rigl_backward_hook', False):
                raise Exception('This model already has been registered to a RigLScheduler.')
        
            self.backward_hook_objects.append(IndexMaskHook(i, self))
            w.register_hook(self.backward_hook_objects[-1])
            setattr(w, '_has_rigl_backward_hook', True)

        assert self.grad_accumulation_n > 0 and self.grad_accumulation_n < delta
        assert self.sparsity_distribution in ('uniform', )




    def state_dict(self):
        obj = {
            'dense_allocation': self.dense_allocation,
            'S': self.S,
            'N': self.N,
            'hyperparams': {
                'delta_T': self.delta_T,
                'alpha': self.alpha,
                'T_end': self.T_end,
                'ignore_linear_layers': self.ignore_linear_layers,
                'static_topo': self.static_topo,
                'sparsity_distribution': self.sparsity_distribution,
                'grad_accumulation_n': self.grad_accumulation_n,
            },
            'step': self.step,
            'rigl_steps': self.rigl_steps,
            'backward_masks': self.backward_masks,
            '_linear_layers_mask': self._linear_layers_mask,
        }

        return obj

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if type(v) == dict:
                self.load_state_dict(v)
            setattr(self, k, v)


    @torch.no_grad()
    def random_sparsify(self):
        is_dist = dist.is_initialized()
        self.backward_masks = []
        for l, w in enumerate(self.W):
            # if sparsity is 0%, skip
            if self.S[l] <= 0:
                self.backward_masks.append(None)
                continue

            n = self.N[l]
            s = int(self.S[l] * n)
            if not self.args.use_bnn or (l%2==0):
                perm = torch.randperm(n)
                perm = perm[:s]
                flat_mask = torch.ones(n, device=w.device)
                flat_mask[perm] = 0
                mask = torch.reshape(flat_mask, w.shape)

                if is_dist:
                    dist.broadcast(mask, 0)

                mask = mask.bool()
            else:
                # use same mask as l-1
                mask = torch.ones(n, device=w.device)
                mask.data = self.backward_masks[-1].data
            w *= mask
            self.backward_masks.append(mask)


    def __str__(self):
        s = 'RigLScheduler(\n'
        s += 'layers=%i,\n' % len(self.N)

        # calculate the number of non-zero elements out of the total number of elements
        N_str = '['
        S_str = '['
        sparsity_percentages = []
        total_params = 0
        total_conv_params = 0
        total_nonzero = 0
        total_conv_nonzero = 0

        for N, S, mask, W, is_linear in zip(self.N, self.S, self.backward_masks, self.W, self._linear_layers_mask):
            actual_S = torch.sum(W[mask == 0] == 0).item()
            N_str += ('%i/%i, ' % (N-actual_S, N))
            sp_p = float(N-actual_S) / float(N) * 100
            S_str += '%.2f%%, ' % sp_p
            sparsity_percentages.append(sp_p)
            total_params += N
            total_nonzero += N-actual_S
            if not is_linear:
                total_conv_nonzero += N-actual_S
                total_conv_params += N

        N_str = N_str[:-2] + ']'
        S_str = S_str[:-2] + ']'
        
        s += 'nonzero_params=' + N_str + ',\n'
        s += 'nonzero_percentages=' + S_str + ',\n'
        s += 'total_nonzero_params=' + ('%i/%i (%.2f%%)' % (total_nonzero, total_params, float(total_nonzero)/float(total_params)*100)) + ',\n'
        s += 'total_CONV_nonzero_params=' + ('%i/%i (%.2f%%)' % (total_conv_nonzero, total_conv_params, float(total_conv_nonzero)/float(total_conv_params)*100)) + ',\n'
        s += 'step=' + str(self.step) + ',\n'
        s += 'num_rigl_steps=' + str(self.rigl_steps) + ',\n'
        s += 'ignoring_linear_layers=' + str(self.ignore_linear_layers) + ',\n'
        s += 'sparsity_distribution=' + str(self.sparsity_distribution) + ',\n'

        return s + ')'


    @torch.no_grad()
    def reset_momentum(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            # if sparsity is 0%, skip
            if s <= 0:
                continue

            param_state = self.optimizer.state[w]
            if 'momentum_buffer' in param_state:
                # mask the momentum matrix
                buf = param_state['momentum_buffer']
                buf *= mask


    @torch.no_grad()
    def apply_mask_to_weights(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            # if sparsity is 0%, skip
            if s <= 0:
                continue
                
            w *= mask

    @torch.no_grad()
    def grow_std_init(self, prev_masks):
        for l, (w, mask, s, prev_mask) in enumerate(zip(self.W, self.backward_masks, self.S, prev_masks)):
            # if sparsity is 0%, skip
            if s <= 0:
                continue

            if self.args.use_bnn and (l%2==0):
                continue
            
            if self.args.use_bnn:
                # ====== abs =====
                # w.data += (mask!=0) * (prev_mask==0) * torch.mean(torch.abs(w.data)).detach() * w.numel() / torch.sum(w!=0)
                # ================
                if self.args.grow_std == 'mean':
                    # ===== softplus =====
                    # # mean after softplus
                    # mean_softplus = torch.mean(F.softplus(w.data)).detach().expm1().log()
                    # mean before softplus
                    mean_softplus = torch.mean(w.data).detach()
                    w.data += (mask!=0) * (prev_mask==0) * mean_softplus * w.numel() / torch.sum(w!=0)
                elif self.args.grow_std == 'eps':
                    # =====================
                    w.data += EPS * (mask!=0) * (prev_mask==0)


    @torch.no_grad()
    def apply_mask_to_gradients(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            # if sparsity is 0%, skip
            if s <= 0:
                continue

            w.grad *= mask

    
    def check_if_backward_hook_should_accumulate_grad(self):
        """
        Used by the backward hooks. Basically just checks how far away the next rigl step is, 
        if it's within `self.grad_accumulation_n` steps, return True.
        """

        if self.step >= self.T_end:
            return False

        steps_til_next_rigl_step = self.delta_T - (self.step % self.delta_T)
        return steps_til_next_rigl_step <= self.grad_accumulation_n


    def cosine_annealing(self):
        return self.alpha / 2 * (1 + np.cos((self.step * np.pi) / self.T_end))

    def check_step_only(self):
        if (self.step+1) % self.delta_T == 0 and self.step<self.T_end:
            return True
        else:
            return False

    def __call__(self):
        self.step += 1
        if self.static_topo:
            return True
        if (self.step % self.delta_T) == 0 and self.step < self.T_end: # check schedule
            self._rigl_step()
            self.rigl_steps += 1
            return False
        return True


    @torch.no_grad()
    def _rigl_step(self):
        drop_fraction = self.cosine_annealing()

        # if distributed these values will be populated
        is_dist = dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else None

        drop_intersect = 0
        drop_union = 0

        prev_masks = copy.deepcopy(self.backward_masks)

        for l, w in enumerate(self.W):
            # if sparsity is 0%, skip
            if self.S[l] <= 0:
                continue

            if self.args.use_bnn and (l%2==1):
                self.backward_masks[l].data = self.backward_masks[l-1].data
                continue


            current_mask = self.backward_masks[l]

            if self.args.use_bnn:
                if self.args.sigma_parameterization == 'abs':
                    sigma = torch.abs(self.W[l+1])
                elif self.args.sigma_parameterization == 'softplus':
                    sigma = F.softplus(self.W[l+1]) * (self.W[l+1] != 0)
                elif self.args.sigma_parameterization == 'exp':
                    sigma = self.W[l+1].exp() * (self.W[l+1] != 0)
                else:
                    raise NotImplementedError()

                if self.args.add_reg_sigma:
                    sigma += 1e-8 * (w!=0)

            # calculate raw scores
            if self.args.drop_criteria == 'mean':
                score_drop = torch.abs(w)
            elif self.args.drop_criteria == 'E_mean_abs' and self.args.use_bnn:
                score_drop = w * torch.erf(w/sigma / np.sqrt(2)) + 2 * sigma * torch.exp(-w**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi)
                # fill the nan values with 0 in score_drop
                score_drop[score_drop != score_drop] = 0
            elif self.args.drop_criteria == 'SNR_mean_abs' and self.args.use_bnn:
                score_drop = w * torch.erf(w/sigma / np.sqrt(2)) + 2 * sigma * torch.exp(-w**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi)
                score_drop = score_drop / torch.sqrt(sigma**2 + w**2 - score_drop**2)
                score_drop[score_drop != score_drop] = 0
            elif self.args.drop_criteria == 'snr' and self.args.use_bnn:
                score_drop = torch.abs(w) / sigma
                score_drop[score_drop != score_drop] = 0
            elif self.args.drop_criteria == 'E_exp_mean_abs' and self.args.use_bnn:
                lam = self.args.lambda_exp
                score_drop = torch.exp(0.5 * (lam ** 2) * (sigma ** 2) + lam * w) * gaussian_cdf(w / sigma + lam * sigma) + \
                             torch.exp(0.5 * (lam ** 2) * (sigma ** 2) - lam * w) * gaussian_cdf(-w / sigma + lam * sigma)
                score_drop[score_drop != score_drop] = 0
            elif self.args.drop_criteria == 'SNR_exp_mean_abs' and self.args.use_bnn:
                lam = self.args.lambda_exp
                E_exp_mean_abs = torch.exp(0.5 * (lam ** 2) * (sigma ** 2) + lam * w) * gaussian_cdf(w / sigma + lam * sigma) + \
                                 torch.exp(0.5 * (lam ** 2) * (sigma ** 2) - lam * w) * gaussian_cdf(-w / sigma + lam * sigma)
                lam = 2 * lam
                E_exp_mean_abs_2 = torch.exp(0.5 * (lam ** 2) * (sigma ** 2) + lam * w) * gaussian_cdf(w / sigma + lam * sigma) + \
                                   torch.exp(0.5 * (lam ** 2) * (sigma ** 2) - lam * w) * gaussian_cdf(-w / sigma + lam * sigma)
                score_drop = E_exp_mean_abs / torch.sqrt(E_exp_mean_abs_2 - E_exp_mean_abs**2)
                score_drop[score_drop != score_drop] = 0                
            else:
                raise NotImplementedError()

            score_grow = torch.abs(self.backward_hook_objects[l].dense_grad)

            # if is distributed, synchronize scores
            if is_dist:
                dist.all_reduce(score_drop)  # get the sum of all drop scores
                score_drop /= world_size     # divide by world size (average the drop scores)

                dist.all_reduce(score_grow)  # get the sum of all grow scores
                score_grow /= world_size     # divide by world size (average the grow scores)

            # calculate drop/grow quantities
            n_total = self.N[l]
            n_ones = torch.sum(current_mask).item()
            n_prune = int(n_ones * drop_fraction)
            n_keep = n_ones - n_prune

            # create drop mask
            _, sorted_indices = torch.topk(score_drop.view(-1), k=n_total)
            new_values = torch.where(
                            torch.arange(n_total, device=w.device) < n_keep,
                            torch.ones_like(sorted_indices),
                            torch.zeros_like(sorted_indices))
            mask1 = new_values.scatter(0, sorted_indices, new_values)

            # calculate similarity drop
            _, sorted_indices_mean = torch.topk(torch.abs(w).view(-1), k=n_total)
            new_values_mean = torch.where(
                            torch.arange(n_total, device=w.device) < n_keep,
                            torch.ones_like(sorted_indices_mean),
                            torch.zeros_like(sorted_indices_mean))
            mask1_mean = new_values_mean.scatter(0, sorted_indices_mean, new_values_mean)
            mask1_drop = (w != 0).float().view(-1) - mask1
            mask1_mean_drop = (w != 0).float().view(-1) - mask1_mean
            drop_intersect += torch.sum(mask1_drop * mask1_mean_drop)
            drop_union += torch.sum(torch.maximum(mask1_drop, mask1_mean_drop))


            # flatten grow scores
            score_grow = score_grow.view(-1)

            # set scores of the enabled connections(ones) to min(s) - 1, so that they have the lowest scores
            score_grow_lifted = torch.where(
                                mask1 == 1, 
                                torch.ones_like(mask1) * (torch.min(score_grow) - 1),
                                score_grow)

            # create grow mask
            _, sorted_indices = torch.topk(score_grow_lifted, k=n_total)
            new_values = torch.where(
                            torch.arange(n_total, device=w.device) < n_prune,
                            torch.ones_like(sorted_indices),
                            torch.zeros_like(sorted_indices))
            mask2 = new_values.scatter(0, sorted_indices, new_values)

            mask2_reshaped = torch.reshape(mask2, current_mask.shape)
            grow_tensor = torch.zeros_like(w)
            
            REINIT_WHEN_SAME = False
            if REINIT_WHEN_SAME:
                raise NotImplementedError()
            else:
                new_connections = ((mask2_reshaped == 1) & (current_mask == 0))

            # update new weights to be initialized as zeros and update the weight tensors
            new_weights = torch.where(new_connections.to(w.device), grow_tensor, w)
            w.data = new_weights

            mask_combined = torch.reshape(mask1 + mask2, current_mask.shape).bool()

            # update the mask
            current_mask.data = mask_combined

        self.reset_momentum()
        self.apply_mask_to_weights()
        self.apply_mask_to_gradients() 

        if self.args.use_bnn:
            self.grow_std_init(prev_masks)

        if drop_union != 0:
            similarity_drop = drop_intersect / drop_union
        else:
            similarity_drop = 1