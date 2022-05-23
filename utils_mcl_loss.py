"""
    Official implementation from "Learning with Multiple Complementary Labels"
    by Lei Feng et al.
"""
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mae_loss(outputs, partialY):
    sm_outputs = F.softmax(outputs, dim=1)
    loss_fn = nn.L1Loss(reduction='none')
    loss_matrix = loss_fn(sm_outputs, partialY.float())
    sample_loss = loss_matrix.sum(dim=-1)
    return sample_loss


def mse_loss(outputs, Y):
    sm_outputs = F.softmax(outputs, dim=1)
    loss_fn = nn.MSELoss(reduction='none')
    loss_matrix = loss_fn(sm_outputs, Y.float())
    sample_loss = loss_matrix.sum(dim=-1)
    return sample_loss


def gce_loss(outputs, Y):
    q = 0.7
    sm_outputs = F.softmax(outputs, dim=1)
    pow_outputs = torch.pow(sm_outputs, q)
    sample_loss = (1-(pow_outputs*Y).sum(dim=1))/q  # n
    return sample_loss


def phuber_ce_loss(outputs, Y):
    trunc_point = 0.1
    n = Y.shape[0]
    soft_max = nn.Softmax(dim=1)
    sm_outputs = soft_max(outputs)
    final_outputs = sm_outputs * Y
    final_confidence = final_outputs.sum(dim=1)
   
    ce_index = (final_confidence > trunc_point)
    sample_loss = torch.zeros(n).to(device)

    if ce_index.sum() > 0:
        ce_outputs = outputs[ce_index,:]
        logsm = nn.LogSoftmax(dim=-1)
        logsm_outputs = logsm(ce_outputs)
        final_ce_outputs = logsm_outputs * Y[ce_index,:]
        sample_loss[ce_index] = - final_ce_outputs.sum(dim=-1)

    linear_index = (final_confidence <= trunc_point)

    if linear_index.sum() > 0:
        sample_loss[linear_index] = -math.log(trunc_point) + (-1/trunc_point)*final_confidence[linear_index] + 1

    return sample_loss


def ce_loss(outputs, Y):
    logsm = nn.LogSoftmax(dim=1)
    logsm_outputs = logsm(outputs)
    final_outputs = logsm_outputs * Y
    sample_loss = - final_outputs.sum(dim=1)
    return sample_loss


def unbiased_estimator(loss_fn, outputs, partialY):
    n, k = partialY.shape[0], partialY.shape[1]
    comp_num = k - partialY.sum(dim=1)
    temp_loss = torch.zeros(n, k).to(device)
    for i in range(k):
        tempY = torch.zeros(n, k).to(device)
        tempY[:, i] = 1.0
        temp_loss[:, i] = loss_fn(outputs, tempY)

    candidate_loss = (temp_loss * partialY).sum(dim=1)  # for true label
    noncandidate_loss = (temp_loss * (1-partialY)).sum(dim=1)  # for complementary label
    total_loss = candidate_loss - (k-comp_num-1.0)/comp_num * noncandidate_loss
    average_loss = total_loss.mean()
    return average_loss


def log_ce_loss(outputs, partialY, pseudo_labels, alpha):
    k = partialY.shape[1]
    can_num = partialY.sum(dim=1).float()  # n

    soft_max = nn.Softmax(dim=1)
    sm_outputs = soft_max(outputs)
    final_outputs = sm_outputs * partialY  # \sum_{j\notin \bar{Y}} [p(j|x)]

    pred_outputs = sm_outputs[torch.arange(sm_outputs.size(0)), pseudo_labels]  # p(pl|x)
    # pred_outputs, _ = torch.max(final_outputs, dim=1)  # \max \sum_{j\notin \bar{Y}} [p(j|x)]

    average_loss = - ((k - 1) / (k - can_num) * torch.log(alpha * final_outputs.sum(dim=1) + (1 - alpha) * pred_outputs)).mean()

    return average_loss


def log_loss(outputs, partialY):
    k = partialY.shape[1]
    can_num = partialY.sum(dim=1).float()  # n
    
    soft_max = nn.Softmax(dim=1)
    sm_outputs = soft_max(outputs)
    final_outputs = sm_outputs * partialY
    
    average_loss = - ((k-1)/(k-can_num) * torch.log(final_outputs.sum(dim=1))).mean()
    return average_loss


def exp_ce_loss(outputs, partialY, pseudo_labels, alpha):
    k = partialY.shape[1]
    can_num = partialY.sum(dim=1).float()  # n

    soft_max = nn.Softmax(dim=1)
    sm_outputs = soft_max(outputs)
    final_outputs = sm_outputs * partialY  # \sum_{j\notin \bar{Y}} [p(j|x)]

    pred_outputs = sm_outputs[torch.arange(sm_outputs.size(0)), pseudo_labels]  # p(pl|x)
    # pred_outputs, _ = torch.max(final_outputs, dim=1)  # \max \sum_{j\notin \bar{Y}} [p(j|x)]

    average_loss = ((k - 1) / (k - can_num) * torch.exp(-alpha * final_outputs.sum(dim=1) - (1 - alpha) * pred_outputs)).mean()

    return average_loss


def exp_loss(outputs, partialY):
    k = partialY.shape[1]
    can_num = partialY.sum(dim=1).float()  # n
    
    soft_max = nn.Softmax(dim=1)
    sm_outputs = soft_max(outputs)
    final_outputs = sm_outputs * partialY

    average_loss = ((k-1)/(k-can_num) * torch.exp(-final_outputs.sum(dim=1))).mean()
    return average_loss
