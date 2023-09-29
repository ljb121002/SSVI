from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import random
import numpy as np
import copy

from rigl_torch.RigL import RigLScheduler

from models.sto_resnet import StoResNet18
from models.utils import bnn_sample

import calibration as cal


import wandb


def train(args, model, device, train_loader, optimizer, epoch, pruner):
    model.train()
    if args.use_bnn:
        model.set_test_mean(False)
    flag = 0
    total_sample, correct, loss_avg, lr_avg = 0., 0., 0., 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.grow_mean_grad:
            if pruner.check_step_only():
                model.set_test_mean(True)
                flag = 1
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        if args.use_bnn:
            loss_kl = model.kl() * args.kl_scale * min(2 * epoch / args.epochs, 1)
            loss += loss_kl
        loss.backward()

        if pruner():
            optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_sample += data.shape[0]
        loss_avg = (loss_avg * batch_idx + loss.item()) / (batch_idx+1)
        lr_avg = (lr_avg * batch_idx + optimizer.param_groups[0]["lr"]) / (batch_idx+1)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        if flag:
            model.set_test_mean(False)
            flag = 0
    if args.use_bnn:
        return {'train_acc': correct/total_sample, 'train_loss': loss_avg, 'lr': lr_avg, 'loss_kl': loss_kl}
    else:
        return {'train_acc': correct/total_sample, 'train_loss': loss_avg, 'lr': lr_avg}




def test(model, device, test_loader, args, return_logit=False):
    model.eval()
    if args.use_bnn:
        model.set_test_mean(True)
    if not return_logit:
        test_loss = 0
        correct = 0
        total = 0
        logit = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                total += data.shape[0]
                correct += pred.eq(target.view_as(pred)).sum().item()
                logit.append(F.softmax(output, dim=1))

        test_loss /= len(test_loader.dataset)
        ece = cal.get_calibration_error(torch.cat(logit, dim=0).cpu(), torch.tensor(test_loader.dataset.targets))

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / total))
        return test_loss, correct / total, ece
    else:
        logit = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                logit.append(F.softmax(output, dim=1))
        logit = torch.cat(logit, dim=0)
        return logit



def ed(param_name, default=None):
    return os.environ.get(param_name, default)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dense_allocation', default=ed('DENSE_ALLOCATION'), type=float,
                        help='percentage of dense parameters allowed. if None, pruning will not be used. must be on the interval (0, 1]')
    parser.add_argument('--delta', default=ed('DELTA', 100), type=int,
                        help='delta param for pruning')
    parser.add_argument('--grad_accumulation_n', default=ed('GRAD_ACCUMULATION_N', 1), type=int)
    parser.add_argument('--alpha', default=ed('ALPHA', 0.3), type=float,
                        help='alpha param for pruning')
    parser.add_argument('--static_topo', default=ed('STATIC_TOPO', 0), type=int, help='if 1, use random sparsity topo and remain static')
    parser.add_argument('--batch_size', type=int, default=ed('BATCH_SIZE', 128), metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=ed('TEST_BATCH_SIZE', 1000), metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=ed('EPOCHS', 250), metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=ed('LR', 0.1), metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step_size', type=float, default=ed('DECAY_STEP', 80), metavar='DS')
    parser.add_argument('--gamma', type=float, default=ed('GAMMA', 0.2), metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--wd', '--weight_decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', default=1, type=bool,
                        help='For Saving the current Model')
    parser.add_argument('--exp_name', default='name', type=str)

    parser.add_argument('--model', default='resnet20frn', type=str)
    parser.add_argument('--load_ckpt', default='', type=str)
    parser.add_argument('--eval_only', default=False, action='store_true')
    parser.add_argument('--nowandb', default=False, action='store_true')
    parser.add_argument('--data', default='cifar10', type=str)

    # bnn parameter
    parser.add_argument('--use_bnn', default=False, action='store_true')
    parser.add_argument('--prior_mean', default=0, type=float)
    parser.add_argument('--prior_std', default=0.005, type=float)
    parser.add_argument('--posterior_mean_init', default=(0.0, 0.01), type=tuple)
    parser.add_argument('--posterior_std_init', default=(0.001, 0.0005), type=tuple)
    parser.add_argument('--posterior_std_init_mean', default=0.0006, type=float)
    parser.add_argument('--kl_scale', default=0.01, type=float)
    parser.add_argument('--eval_bnn', default=0, type=int, help='if 0, eval normal nn; if 1, eval bnn with mean; if >1, eval bnn with mean and sample eval_bnn times')
    parser.add_argument('--same_noise', default=False, action='store_true')

    parser.add_argument('--drop_criteria', default='SNR_mean_abs', type=str, choices=['mean', 'E_mean_abs', 'snr', 'E_exp_mean_abs', 'SNR_mean_abs', 'SNR_exp_mean_abs'])
    parser.add_argument('--lambda_exp', default=1.0, type=float)
    parser.add_argument('--add_reg_sigma', default=False, action='store_true', help='if true, add regularization term for sigma to prevent zeros')
    parser.add_argument('--grow_std', default='mean', type=str, choices=['mean', 'eps'])
    parser.add_argument('--grow_mean_grad', default=False, action='store_true', help='if true, grow mean grad')
    parser.add_argument('--lr_std', default=0.01, type=float, help='lr for std')
    parser.add_argument('--sigma_parameterization', default='softplus', type=str, choices=['softplus', 'exp', 'abs'])

    args = parser.parse_args()
    
    args.posterior_std_init = (args.posterior_std_init_mean, args.posterior_std_init[1])


    if args.grow_mean_grad:
        assert 'growgrad_mean' in args.exp_name

    if not args.nowandb:
        wandb.init(entity='entity', project='ssvi', name=args.exp_name, config=args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")


    if args.data == 'cifar10':
        dataset1 = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([
                                                                                    transforms.RandomCrop(32, 4),
                                                                                    transforms.RandomHorizontalFlip(),
                                                                                    transforms.ToTensor(),
                                                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                                                    ]))
        dataset2 = datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                                                                                    transforms.ToTensor(),
                                                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                                                    ]))
        num_classes = 10
    elif args.data == 'cifar100':
        dataset1 = datasets.CIFAR100('./data', train=True, download=True, transform=transforms.Compose([
                                                                                    transforms.RandomHorizontalFlip(),
                                                                                    transforms.RandomCrop(32, 4),
                                                                                    transforms.ToTensor(),
                                                                                    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
                                                                                    ]))
        dataset2 = datasets.CIFAR100('./data', train=False, transform=transforms.Compose([
                                                                                    transforms.ToTensor(),
                                                                                    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
                                                                                    ]))
        num_classes = 100


    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=args.test_batch_size, shuffle=False)

    args.kl_scale /= len(train_loader.dataset)


    model = StoResNet18(num_classes=num_classes, use_bnn=args.use_bnn, prior_mean=args.prior_mean, prior_std=args.prior_std, \
                        posterior_mean_init=args.posterior_mean_init, posterior_std_init=args.posterior_std_init, same_noise=args.same_noise, \
                            sigma_parameterization=args.sigma_parameterization)
    # model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)

    if args.load_ckpt:
        ckpt = torch.load(args.load_ckpt, map_location='cpu')
        if 'model' in ckpt:
            ckpt = ckpt['model']
        model.load_state_dict(ckpt, strict=True)
    model.to(device)

    if args.eval_only:
        print(args.load_ckpt)
        if args.eval_bnn == 0 or args.eval_bnn == 1:
            # eval normal nn or eval bnn with mean
            loss, acc, ece = test(model, device, test_loader, args)
            print('acc: ', acc)
            print('loss: ', loss)
            print('ece: ', ece)
            exit(0)
        else:
            logit_avg = None
            for i in range(args.eval_bnn):
                model_i = bnn_sample(copy.deepcopy(model), args)
                logit = test(model_i, device, test_loader, args, return_logit=True)
                if logit_avg is None:
                    logit_avg = logit
                else:
                    logit_avg = logit_avg * i / (i+1) + logit / (i+1)
                if i % 10 == 0:
                    print('finish eval ', str(i))
            labels = torch.tensor(test_loader.dataset.targets).cuda()
            acc = (logit_avg.argmax(dim=1) == labels).float().mean().item()
            loss = -torch.log(logit_avg[range(len(test_loader.dataset.targets)), labels]).mean().item()
            ece = cal.get_calibration_error(logit_avg.cpu(), labels.cpu())
            print('eval bnn with {} times'.format(args.eval_bnn))
            print('acc: ', acc)
            print('loss: ', loss)
            print('ece: ', ece)
            exit(0)
        
    optimizer = optim.SGD([{'params': [param for name, param in model.named_parameters() if not name.endswith('posterior_std')], 
                            'weight_decay': args.weight_decay,
                            'lr': args.lr},
                           {'params': [param for name, param in model.named_parameters() if name.endswith('posterior_std')], 
                            'weight_decay': 0, 
                            'lr': args.lr_std}],
                           momentum=args.momentum)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    pruner = lambda: True
    if args.dense_allocation is not None:
        T_end = int(0.75 * args.epochs * len(train_loader))
        pruner = RigLScheduler(model, optimizer, dense_allocation=args.dense_allocation, alpha=args.alpha, delta=args.delta, static_topo=args.static_topo, T_end=T_end, ignore_linear_layers=False, grad_accumulation_n=args.grad_accumulation_n, args=args)

    writer = SummaryWriter(log_dir='./graphs')

    # print(model)

    acc_best = 0
    for epoch in range(1, args.epochs + 1):
        print(pruner)
        train_log = train(args, model, device, train_loader, optimizer, epoch, pruner=pruner)
        loss, acc, _ = test(model, device, test_loader, args)

        train_log.update({'test_loss': loss, 'test_acc': acc})
        if not args.nowandb:
            wandb.log(train_log)

        if epoch == 1:
            with open('./log/' + args.exp_name + '.txt', 'a') as f:
                f.write('\nepoch ')
                for k in train_log.keys():
                    f.write(str(k) + ' ')
                f.write('test_loss ')
                f.write('test_acc\n')
                
        with open('./log/' + args.exp_name + '.txt', 'a') as f:
            f.write(str(epoch) + ' ')
            for v in train_log.values():
                f.write(str(v) + ' ')
            f.write(str(loss) + ' ')
            f.write(str(acc) + '\n')

        scheduler.step()

        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar('accuracy', acc, epoch)

        if args.save_model:
            if not os.path.exists('./ckpts'):
                os.makedirs('./ckpts')
            # torch.save(model.state_dict(), "./ckpts/" + args.exp_name + ".pt")
            if acc > acc_best:
                acc_best = acc
                # save model with epoch in a dict
                torch.save({'epoch': epoch, 'model': model.state_dict()}, "./ckpts/" + args.exp_name + "_best.pt")




if __name__ == '__main__':
    main()
