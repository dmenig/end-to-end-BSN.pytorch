# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F


def bi_loss(scores, anchors, tem_match_thres=0.5, use_gpu=True):
    scores = scores.view(-1).to("cuda" if use_gpu else "cpu")
    anchors = anchors.contiguous().view(-1)  # anchors is an array of booleans

    pmask = (scores > tem_match_thres).float().to("cuda" if use_gpu else "cpu")
    num_positive = torch.sum(pmask)
    num_entries = len(scores)
    ## I accounted for cases with no action in the video, and cases with only
    ## action in the video, which cause nans without securities
    ratio = max((num_entries, num_positive + 1)) / max((num_positive, 1))
    coef_0 = 0.5 * (ratio) / (ratio - 1)
    coef_1 = coef_0 * (ratio - 1)
    loss = coef_1 * pmask * torch.log(anchors + 0.00001) + coef_0 * (
        1.0 - pmask
    ) * torch.log(1.0 - anchors + 0.00001)
    loss = -torch.mean(loss)
    num_sample = [torch.sum(pmask), ratio]
    return loss, num_sample


def TEM_loss_calc(
    anchors_action,
    anchors_start,
    anchors_end,
    match_scores_action,
    match_scores_start,
    match_scores_end,
    use_gpu=True,
):

    loss_action, num_sample_action = bi_loss(
        match_scores_action, anchors_action, use_gpu=use_gpu
    )
    loss_start_small, num_sample_start_small = bi_loss(
        match_scores_start, anchors_start, use_gpu=use_gpu
    )
    loss_end_small, num_sample_end_small = bi_loss(
        match_scores_end, anchors_end, use_gpu=use_gpu
    )

    loss_dict = {
        "loss_action": loss_action,
        "num_sample_action": num_sample_action,
        "loss_start": loss_start_small,
        "num_sample_start": num_sample_start_small,
        "loss_end": loss_end_small,
        "num_sample_end": num_sample_end_small,
    }
    # print loss_dict
    return loss_dict


def TEM_loss_function(y_action, y_start, y_end, TEM_output, use_gpu=True):
    anchors_action = TEM_output[:, 0, :]
    anchors_start = TEM_output[:, 1, :]
    anchors_end = TEM_output[:, 2, :]
    loss_dict = TEM_loss_calc(
        anchors_action,
        anchors_start,
        anchors_end,
        y_action,
        y_start,
        y_end,
        use_gpu=use_gpu,
    )

    cost = (
        2 * loss_dict["loss_action"] + loss_dict["loss_start"] + loss_dict["loss_end"]
    )
    loss_dict["cost"] = cost
    return loss_dict


def PEM_loss_function(
    anchors_iou,
    match_iou,
    pem_high_iou_thres=0.6,
    pem_low_iou_thres=2.2,
    u_ratio_m=1.0,
    u_ratio_l=2.0,
    use_gpu=True,
):
    match_iou = match_iou.to("cuda" if use_gpu else "cpu")
    anchors_iou = anchors_iou.view(-1)
    u_hmask = (match_iou > pem_high_iou_thres).float()
    u_mmask = (
        (match_iou <= pem_high_iou_thres) & (match_iou > pem_low_iou_thres)
    ).float()
    u_lmask = (match_iou < pem_low_iou_thres).float()

    num_h = torch.sum(u_hmask)
    num_m = torch.sum(u_mmask)
    num_l = torch.sum(u_lmask)

    r_m = u_ratio_m * num_h / (num_m)
    r_m = torch.min(r_m, torch.Tensor([1.0]).to("cuda" if use_gpu else "cpu"))[0]
    u_smmask = torch.Tensor(np.random.rand(u_hmask.size()[0])).to(
        "cuda" if use_gpu else "cpu"
    )
    u_smmask = u_smmask * u_mmask
    u_smmask = (u_smmask > (1.0 - r_m)).float()

    r_l = u_ratio_l * num_h / (num_l)
    r_l = torch.min(r_l, torch.Tensor([1.0]).to("cuda" if use_gpu else "cpu"))[0]
    u_slmask = torch.Tensor(np.random.rand(u_hmask.size()[0])).to(
        "cuda" if use_gpu else "cpu"
    )
    u_slmask = u_slmask * u_lmask
    u_slmask = (u_slmask > (1.0 - r_l)).float()

    iou_weights = u_hmask + u_smmask + u_slmask
    iou_loss = F.smooth_l1_loss(anchors_iou, match_iou)
    iou_loss = torch.sum(iou_loss * iou_weights) / (
        torch.Tensor([1.0]).to("cuda" if use_gpu else "cpu") + torch.sum(iou_weights)
    )

    return iou_loss


def correctLogits(logits, matches, use_gpu):
    new_logits = []
    for i in range(len(logits)):
        new_logit = [logits[i][j] for j in matches[i]]
        if len(new_logit) != 0:
            new_logits.append(torch.stack(new_logit))
    return (
        torch.cat(new_logits)
        if len(new_logits) != 0
        else torch.Tensor([]).to("cuda" if use_gpu else "cpu")
    )
