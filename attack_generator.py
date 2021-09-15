import numpy as np
from models import *
from torch.autograd import Variable
from utils_algo import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def pgd(model, data, target, true_labels, epsilon, step_size, num_steps, K, ccp, meta_method="nn", loss_fn="unbiased", category="Madry", rand_init=True):
    model.eval()
    y_adv, bs = true_labels, true_labels.size(0)
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).to(device).detach() if rand_init else data.detach()
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().to(device) if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        predict = torch.max(output, dim=1)[1]
        y_adv = torch.cat((y_adv, predict))
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "unbiased":
                loss_adv, _ = chosen_loss_c(f=output, K=K, labels=target, ccp=ccp, meta_method=meta_method)
            elif loss_fn == "biased":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        if loss_fn == "biased":
            eta *= -1
        # Update adversarial data
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv = Variable(x_adv, requires_grad=False)
    y_adv = torch.cat((y_adv, target))
    y_adv = y_adv.view(-1, bs).transpose(0, 1)
    return x_adv, y_adv


def eval_clean(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, index in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def eval_robust(model, test_loader, perturb_steps, epsilon, step_size, loss_fn, category, random):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.enable_grad():
        for data, target, index in test_loader:
            data, target = data.to(device), target.to(device)
            x_adv, _ = pgd(model, data, target, epsilon, step_size, perturb_steps, loss_fn, category, rand_init=random)
            output = model(x_adv)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy
