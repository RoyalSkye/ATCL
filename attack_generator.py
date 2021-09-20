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


def adv_cl(model, data, target, true_labels, id, epsilon, step_size, num_steps, K, ccp, x_to_mcls, generate_cl_steps=100, meta_method="nn", category="Madry", rand_init=True):
    """
        min---max \bar{l}(\bar{y}, g(x))
           |--min cross-entropy(\bar{y}, g(x)) -> mcls
    """
    # TODO: 1. how to discriminate between new_cls and true_labels, it's possible that new_cls == true_labels, e.g. prob margin
    #  2. how to use the new (m)cls - "learning with mcls"
    #  3. For CIFAR10, PGD num_steps can gradually increase to avoid failing
    #  4. Co-teaching e.g., two models with different losses to generate mcls / gradient ascent
    model.eval()
    y_adv, bs = true_labels, true_labels.size(0)
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).to(device).detach() if rand_init else data.detach()
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().to(device) if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    # generate multiple complementary labels friendly: for each datapoint, stop when predict = (cl) target
    if generate_cl_steps > 0:
        iter_adv = x_adv.clone().detach()
        iter_target = target.clone().detach()
        remain_index = [i for i in range(bs)]
        for k in range(generate_cl_steps):
            iter_adv.requires_grad_()
            output_index = []
            iter_index = []
            output = model(iter_adv)
            # prob_ = torch.softmax(output.detach(), dim=1)
            predict = torch.max(output.detach(), dim=1)[1]
            predict_ext = torch.full((bs,), -1).to(device)
            predict_ext[remain_index] = predict
            y_adv = torch.cat((y_adv, predict_ext))
            for idx in range(len(predict)):
                if predict[idx] == iter_target[idx]:
                    output_index.append(idx)
                else:
                    iter_index.append(idx)
            remain_index = [remain_index[i] for i in range(len(remain_index)) if i not in output_index]
            model.zero_grad()
            with torch.enable_grad():
                # loss_adv, _ = chosen_loss_c(f=output, K=K, labels=iter_target, ccp=ccp, meta_method=meta_method)
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, iter_target)
                # one_hot = torch.zeros(iter_target.size(0), K).to(device).scatter_(1, iter_target.view(-1, 1), 1)
                # loss_adv = nn.L1Loss(reduction="mean")(output, one_hot)
            loss_adv.backward()
            grad = iter_adv.grad
            if len(iter_index) != 0:
                iter_adv = iter_adv[iter_index]
                iter_target = iter_target[iter_index]
                grad = grad[iter_index]
                eta = step_size * grad.sign()
                iter_adv = iter_adv.detach() - eta + 0.001 * torch.randn(iter_adv.shape).detach().to(device)
                iter_adv = torch.clamp(iter_adv, 0.0, 1.0)
            else:
                break
        iter_adv = Variable(iter_adv, requires_grad=False)

    # generate adversarial examples
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        predict = torch.max(output.detach(), dim=1)[1]
        if generate_cl_steps <= 0: y_adv = torch.cat((y_adv, predict))
        model.zero_grad()
        with torch.enable_grad():
            loss_adv, _ = chosen_loss_c(f=output, K=K, labels=target, ccp=ccp, meta_method=meta_method)
            # loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv = Variable(x_adv, requires_grad=False)
    y_adv = torch.cat((y_adv, target))
    y_adv = y_adv.view(-1, bs).transpose(0, 1)
    # updata x_to_mcls
    for i, y in enumerate(y_adv):
        new_cls = torch.unique(y[1:]).tolist()
        if -1 in new_cls: new_cls.remove(-1)
        # remove the potential true labels
        if y[1] in new_cls: new_cls.remove(y[1])
        if y[-1] in new_cls: new_cls.remove(y[-1])
        x_to_mcls[id[i].item()] = x_to_mcls[id[i].item()] | set(new_cls)
    return x_adv, y_adv


def pgd(model, data, target, epsilon, step_size, num_steps, loss_fn, category, rand_init, num_classes=10):
    model.eval()
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        nat_output = model(data)
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
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
                criterion_kl = nn.KLDivLoss(reduction="mean").cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1), F.softmax(nat_output, dim=1))
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
