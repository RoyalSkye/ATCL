import numpy as np
from torch.autograd import Variable
from utils_algo import *
from utils_mcl_loss import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trades_loss(adv_logits, natural_logits, target, beta):
    """
        Based on the repo TREADES: https://github.com/yaodongyu/TRADES
    """
    batch_size = len(target)
    criterion_kl = nn.KLDivLoss(size_average=False).cuda()
    loss_natural = nn.CrossEntropyLoss(reduction='mean')(natural_logits, target)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_logits, dim=1), F.softmax(natural_logits, dim=1))
    loss = loss_natural + beta * loss_robust
    return loss


def cwloss(output, target, confidence=50, num_classes=10):
    # Compute the probability of the label class versus the maximum other
    # The same implementation as in repo CAT https://github.com/sunblaze-ucb/curriculum-adversarial-training-CAT
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.to(device)
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss


def cl_adv(args, model, data, target, epsilon, step_size, num_steps, id, ccp, partialY, pseudo_labels, alpha, loss_fn, category="Madry", rand_init=True):
    model.eval()
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).to(device).detach() if rand_init else data.detach()
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().to(device) if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    # generate adversarial examples
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        model.zero_grad()
        with torch.enable_grad():
            if args.method in ['exp', 'log']:
                loss_adv = loss_fn(output, partialY[id].float())
            elif args.method in ['mae', 'mse', 'ce', 'gce', 'phuber_ce']:
                loss_adv = unbiased_estimator(loss_fn, output, partialY[id].float())
            elif args.method in ['free', 'nn', 'ga', 'pc', 'forward', 'scl_exp', 'scl_nl', 'l_uw', 'l_w']:
                assert args.cl_num == 1
                loss_adv, _ = chosen_loss_c(f=output, K=output.size(-1), labels=target, ccp=ccp, meta_method=args.method)
            elif args.method in ["log_ce", "exp_ce"]:
                loss_adv = loss_fn(output, partialY[id].float(), pseudo_labels, alpha)
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv = Variable(x_adv, requires_grad=False)

    return x_adv


def pgd(model, data, target, epsilon, step_size, num_steps, loss_fn, category, rand_init, num_classes=10):
    model.eval()
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).to(device).detach() if rand_init else data.detach()
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().to(device) if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "cw":
                loss_adv = cwloss(output, target, num_classes=num_classes)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(reduction="batchmean").cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1), F.softmax(model(data), dim=1))
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv = Variable(x_adv, requires_grad=False)
    return x_adv


def eval_robust(model, test_loader, perturb_steps, epsilon, step_size, loss_fn, category, random, num_classes=10):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            x_adv = pgd(model, data, target, epsilon, step_size, perturb_steps, loss_fn, category, rand_init=random, num_classes=num_classes)
            output = model(x_adv)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100 * correct / len(test_loader.dataset)
    return test_loss, test_accuracy
