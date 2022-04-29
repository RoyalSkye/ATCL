import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assump_free_loss(f, K, labels, ccp):
    """Assumption free loss (based on Thm 1) is equivalent to non_negative_loss if the max operator's threshold is negative inf."""
    return non_negative_loss(f=f, K=K, labels=labels, ccp=ccp, beta=np.inf)


def non_negative_loss(f, K, labels, ccp, beta):
    ccp = torch.from_numpy(ccp).float().to(device)
    neglog = -F.log_softmax(f, dim=1)  # (bs, K)
    loss_vector = torch.zeros(K, requires_grad=True).to(device)
    temp_loss_vector = torch.zeros(K).to(device)
    for k in range(K):
        idx = (labels == k)
        if torch.sum(idx).item() > 0:
            idxs = idx.view(-1, 1).repeat(1, K)  # (bs, K)
            neglog_k = torch.masked_select(neglog, idxs).view(-1, K)
            temp_loss_vector[k] = -(K-1) * ccp[k] * torch.mean(neglog_k, dim=0)[k]  # average of k-th class loss for k-th comp class samples
            # ccp[k] or ccp, refer to https://github.com/takashiishida/comp/issues/2
            loss_vector = loss_vector + torch.mul(ccp[k], torch.mean(neglog_k, dim=0))  # only k-th in the summation of the second term inside max
    loss_vector = loss_vector + temp_loss_vector
    count = np.bincount(labels.data.cpu()).astype('float')
    while len(count) < K:
        count = np.append(count, 0)  # when largest label is below K, bincount will not take care of them
    loss_vector_with_zeros = torch.cat((loss_vector.view(-1, 1), torch.zeros(K, requires_grad=True).view(-1, 1).to(device)-beta), 1)
    max_loss_vector, _ = torch.max(loss_vector_with_zeros, dim=1)
    final_loss = torch.sum(max_loss_vector)
    return final_loss, torch.mul(torch.from_numpy(count).float().to(device), loss_vector)


def fast_free_loss(f, K, cl_label):
    loss = -(K - 1) * nn.CrossEntropyLoss(reduction="none")(f, cl_label)
    for k in range(K):
        ll = torch.LongTensor([k] * cl_label.size(0)).to(device)
        loss += nn.CrossEntropyLoss(reduction="none")(f, ll)

    return loss


def forward_loss(f, K, labels, reduction='mean'):
    Q = torch.ones(K, K) * 1/(K-1)  # uniform assumption
    Q = Q.to(device)
    for k in range(K):
        Q[k, k] = 0
    q = torch.mm(F.softmax(f, 1), Q)
    return F.nll_loss(q.log(), labels.long(), reduction=reduction)


def pc_loss(f, K, labels):
    sigmoid = nn.Sigmoid()
    fbar = f.gather(1, labels.long().view(-1, 1)).repeat(1, K)
    loss_matrix = sigmoid(-1. * (f - fbar))  # multiply -1 for "complementary"
    M1, M2 = K*(K-1)/2, K-1
    pc_loss = torch.sum(loss_matrix)*(K-1)/len(labels) - M1 + M2
    return pc_loss


def phi_loss(phi, logits, target, reduction='mean'):
    """
        Official implementation of "Unbiased Risk Estimators Can Mislead: A Case Study of Learning with Complementary Labels"
        by Yu-Ting Chou et al.
    """
    if phi == 'lin':
        activated_prob = F.softmax(logits, dim=1)
    elif phi == 'quad':
        activated_prob = torch.pow(F.softmax(logits, dim=1), 2)
    elif phi == 'exp':
        activated_prob = torch.exp(F.softmax(logits, dim=1))
    elif phi == 'log':
        activated_prob = torch.log(F.softmax(logits, dim=1))
    elif phi == 'nl':
        activated_prob = -torch.log(1 - F.softmax(logits, dim=1) + 1e-5)
    elif phi == 'hinge':
        activated_prob = F.softmax(logits, dim=1) - (1 / 10)
        activated_prob[activated_prob < 0] = 0
    else:
        raise ValueError('Invalid phi function')

    loss = -F.nll_loss(activated_prob, target, reduction=reduction)
    return loss


"""
    Below is the official implementation of ICML 2021 "Discriminative Complementary-Label Learning with Weighted Loss"
    by Yi Gao et al.
"""


def non_k_softmax_loss(f, K, labels):
    Q_1 = 1 - F.softmax(f, 1)
    Q_1 = F.softmax(Q_1, 1)
    labels = labels.long()
    return F.nll_loss(Q_1.log(), labels.long())  # Equation(8) in paper


def w_loss(f, K, labels):
    loss_class = non_k_softmax_loss(f=f, K=K, labels=labels)
    loss_w = w_loss_p(f=f, K=K, labels=labels)
    final_loss = loss_class + loss_w  # Equation(11) in paper
    return final_loss


# weighted loss
def w_loss_p(f, K, labels):
    Q_1 = 1-F.softmax(f, 1)
    Q = F.softmax(Q_1, 1)
    q = torch.tensor(1.0) / torch.sum(Q_1, dim=1)
    q = q.view(-1, 1).repeat(1, K)
    w = torch.mul(Q_1, q)  # weight
    w_1 = torch.mul(w, Q.log())
    return F.nll_loss(w_1, labels.long())  # Equation(14) in paper


def chosen_loss_c(f, K, labels, ccp, meta_method, reduction='mean'):
    class_loss_torch = None
    if meta_method == 'free':
        final_loss, class_loss_torch = assump_free_loss(f=f, K=K, labels=labels, ccp=ccp)
    elif meta_method == 'ga':
        final_loss, class_loss_torch = assump_free_loss(f=f, K=K, labels=labels, ccp=ccp)
    elif meta_method == 'nn':
        final_loss, class_loss_torch = non_negative_loss(f=f, K=K, labels=labels, beta=0, ccp=ccp)
    elif meta_method == 'forward':
        final_loss = forward_loss(f=f, K=K, labels=labels, reduction=reduction)
    elif meta_method == 'pc':
        final_loss = pc_loss(f=f, K=K, labels=labels)
    elif meta_method[:3] == "scl":
        final_loss = phi_loss(meta_method[4:], f, labels, reduction=reduction)
    elif meta_method == 'l_uw':
        final_loss = non_k_softmax_loss(f=f, K=K, labels=labels)
    elif meta_method == 'l_w':
        final_loss = w_loss(f=f, K=K, labels=labels)

    return final_loss, class_loss_torch
