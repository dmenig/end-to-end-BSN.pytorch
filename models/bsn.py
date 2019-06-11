# -*- coding: utf-8 -*-
"""

@author :  Damien MENIGAUX

        6th June 2019
        I reworked the official BSN implementation in pytorch so that it could
        be easily trained on any custom dataset, in an end-to-end fashion
        and that it could accept videos with no action, which I find the previous
        work lacked.

        This network works with a contextual 2D backbone, but it is easy to
        replace the feature extractor by anything else (such as 3d models...)

"""


import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from models.backbone import CNNEncoder, CNN_embed_dim


def getMax(scores_list, temporal_scale, peak_thres, temporal_step):
    """
    In contrary to the paper, I didn't choose to select all local maxima and all
    probas above threshold, but rather all local maxima above threshold.
    """

    return (
        [[0, scores_list[0]]]
        + [
            (i, scores_list[i])
            for i in range(1, temporal_scale - 1)
            if scores_list[i] > scores_list[i - 1]
            and scores_list[i] > scores_list[i + 1]
            and scores_list[i] > peak_thres
        ]
        + [[temporal_scale - 1, scores_list[-1]]]
    )


def generateProposals(tem_output, temporal_scale=100, peak_thres=0.5):
    """
    We focus on having a fully differentiable forward pass. The proposals
    generated must be indexes, but we'll keep using the tem_output tensor.
    """
    temporal_step = 1.0 / temporal_scale

    # extract start and end proposals
    starts = getMax(tem_output[1, :], temporal_scale, peak_thres, temporal_step)
    ends = getMax(tem_output[2, :], temporal_scale, peak_thres, temporal_step)

    proposals = []
    logits = []
    for i in range(len(ends)):
        end_index, end_score = ends[i]

        for j in range(len(starts)):
            start_index, start_score = starts[j]
            if start_index >= end_index:
                break
            proposals.append(
                [
                    end_index,
                    start_index,
                    end_score,
                    start_score,
                    end_score * start_score,
                ]
            )
            logits.append(torch.mean(tem_output[3:, start_index:end_index], 1))
    if len(proposals) != 0:
        proposals = np.stack(proposals)
        # sort according to compounded score
        reorder = np.argsort(proposals[:, -1])[::-1]
        proposals = proposals[reorder]
        logits = torch.stack([logits[i] for i in reorder])
    else:
        proposals = np.array(proposals)
        logits = torch.Tensor([])
    return proposals, logits


def interp1d(indexes, values, use_gpu):
    """
    Differentiable 1d interpolation
    """
    input = values.float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
    grid = (
        torch.from_numpy(np.stack((indexes, np.zeros(len(indexes))), 1))
        .to("cuda" if use_gpu else "cpu")
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .div((input.size(-1) - 1) * 0.5)
        .add(-1)
    )

    return torch.nn.functional.grid_sample(input, grid)[0, 0, 0]


def generateFeature(
    tem_output,
    pgm_proposals,
    num_sample_start=8,
    num_sample_end=8,
    num_sample_action=16,
    num_sample_interpld=3,
    temporal_scale=100,
    pem_top_K=500,
    pem_top_K_inference=1000,
    bsp_boundary_ratio=0.2,
    mode="train",
    use_gpu=True,
):
    temporal_step = 1.0 / temporal_scale
    action_scores = tem_output[0, :]
    scores_length = len(action_scores)
    video_extend = int(scores_length / 4 + 10)
    if mode == "train":
        pgm_proposals = pgm_proposals[:pem_top_K]
    else:
        pgm_proposals = pgm_proposals[:pem_top_K_inference]

    action_scores = F.pad(
        action_scores, (video_extend, video_extend), mode="constant", value=0
    )  ### padding

    feature_bsp = []

    for idx in range(len(pgm_proposals)):
        xmin, xmax, xmin_score, xmax_score, compound_score = pgm_proposals[idx]
        xlen = xmax - xmin
        xmin_0 = xmin - xlen * bsp_boundary_ratio
        xmin_1 = xmin + xlen * bsp_boundary_ratio
        xmax_0 = xmax - xlen * bsp_boundary_ratio
        xmax_1 = xmax + xlen * bsp_boundary_ratio
        # start
        plen_start = (xmin_1 - xmin_0) / (num_sample_start - 1)
        plen_sample = plen_start / num_sample_interpld
        tmp_x_new = [
            xmin_0 - plen_start / 2 + plen_sample * i
            for i in range(num_sample_start * num_sample_interpld + 1)
        ]

        tmp_y_new_start_action = interp1d(tmp_x_new, action_scores, use_gpu)

        tmp_y_new_start = torch.tensor(
            [
                torch.mean(
                    tmp_y_new_start_action[
                        i * num_sample_interpld : (i + 1) * num_sample_interpld + 1
                    ]
                )
                for i in range(num_sample_start)
            ]
        )
        # end
        plen_end = (xmax_1 - xmax_0) / (num_sample_end - 1)
        plen_sample = plen_end / num_sample_interpld
        tmp_x_new = [
            xmax_0 - plen_end / 2 + plen_sample * i
            for i in range(num_sample_end * num_sample_interpld + 1)
        ]
        tmp_y_new_end_action = interp1d(tmp_x_new, action_scores, use_gpu)

        tmp_y_new_end = torch.tensor(
            [
                torch.mean(
                    tmp_y_new_end_action[
                        i * num_sample_interpld : (i + 1) * num_sample_interpld + 1
                    ]
                )
                for i in range(num_sample_end)
            ]
        )
        # action
        plen_action = (xmax - xmin) / (num_sample_action - 1)
        plen_sample = plen_action / num_sample_interpld
        tmp_x_new = [
            xmin - plen_action / 2 + plen_sample * i
            for i in range(num_sample_action * num_sample_interpld + 1)
        ]
        tmp_y_new_action = interp1d(tmp_x_new, action_scores, use_gpu)

        tmp_y_new_action = torch.tensor(
            [
                torch.mean(
                    tmp_y_new_action[
                        i * num_sample_interpld : (i + 1) * num_sample_interpld + 1
                    ]
                )
                for i in range(num_sample_action)
            ]
        )
        tmp_feature = torch.cat([tmp_y_new_action, tmp_y_new_start, tmp_y_new_end])
        feature_bsp.append(tmp_feature)

    if len(feature_bsp) != 0:
        return torch.stack(feature_bsp)
    return torch.Tensor(feature_bsp)


def binarize_sequences(raw_targets, sample_duration, boundary_ratio=0.1):
    match_score_actions, match_score_starts, match_score_ends = [], [], []
    for raw_target in raw_targets:
        anchor_xmin = [i / sample_duration for i in range(sample_duration)]
        anchor_xmax = [i / sample_duration for i in range(1, sample_duration + 1)]
        gt_bbox = []
        for action in raw_target:
            tmp_start = max(min(1, action[0]), 0)
            tmp_end = max(min(1, action[1]), 0)
            gt_bbox.append([tmp_start, tmp_end])
        if len(gt_bbox) == 0:
            match_score_action = [0] * len(anchor_xmin)
            match_score_start = [0] * len(anchor_xmin)
            match_score_end = [0] * len(anchor_xmin)
        else:

            gt_bbox = np.array(gt_bbox)
            gt_xmins = gt_bbox[:, 0]
            gt_xmaxs = gt_bbox[:, 1]

            gt_lens = gt_xmaxs - gt_xmins
            gt_len_small = np.maximum(1.0 / sample_duration, boundary_ratio * gt_lens)
            gt_start_bboxs = np.stack(
                (gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1
            )
            gt_end_bboxs = np.stack(
                (gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1
            )

            match_score_action = []
            for jdx in range(len(anchor_xmin)):
                match_score_action.append(
                    np.max(
                        ioa_with_anchors(
                            anchor_xmin[jdx], anchor_xmax[jdx], gt_xmins, gt_xmaxs
                        )
                    )
                )
            match_score_start = []
            for jdx in range(len(anchor_xmin)):
                match_score_start.append(
                    np.max(
                        ioa_with_anchors(
                            anchor_xmin[jdx],
                            anchor_xmax[jdx],
                            gt_start_bboxs[:, 0],
                            gt_start_bboxs[:, 1],
                        )
                    )
                )
            match_score_end = []
            for jdx in range(len(anchor_xmin)):
                match_score_end.append(
                    np.max(
                        ioa_with_anchors(
                            anchor_xmin[jdx],
                            anchor_xmax[jdx],
                            gt_end_bboxs[:, 0],
                            gt_end_bboxs[:, 1],
                        )
                    )
                )
        match_score_actions.append(match_score_action)
        match_score_starts.append(match_score_start)
        match_score_ends.append(match_score_end)
    return (
        torch.Tensor(match_score_actions),
        torch.Tensor(match_score_starts),
        torch.Tensor(match_score_ends),
    )


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.0)
    union_len = len_anchors - inter_len + box_max - box_min
    jaccard = np.divide(inter_len, union_len)
    return jaccard


def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute intersection between score a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.0)
    scores = np.divide(inter_len, len_anchors)
    return scores


def getIou(proposals, raw_targets, use_gpu):
    ## the original paper uses a "feature_frame" variable, which is meant to
    ## store how many frame were actually passed through the feature extractor
    ## I don't think it's needed so I dropped it.
    match_ious = []
    match_labels = []
    labels = []
    matches = []
    for i in range(len(proposals)):
        gt_xmins = []
        gt_xmaxs = []
        gt_labels = []
        match = []
        for target in raw_targets[i]:
            gt_xmins.append(target[1])
            gt_xmaxs.append(target[2])
            gt_labels.append(target[0])
        match_iou = []
        match_label = []
        for j in range(len(proposals[i])):
            ious = iou_with_anchors(
                proposals[i][j][1].data.cpu().numpy(),
                proposals[i][j][0].data.cpu().numpy(),
                gt_xmins,
                gt_xmaxs,
            )
            if len(ious) != 0 and max(ious) > 0.5:
                tmp_new_iou = max(ious)
                match_label.append(gt_labels[np.argmax(ious)])
                match_iou.append(tmp_new_iou)
                match.append(j)
            else:
                match_iou.append(0.0 if len(ious) == 0 else max(ious))
        match_ious.append(torch.Tensor(match_iou).to("cuda" if use_gpu else "cpu"))
        match_labels.append(
            torch.LongTensor(match_label).to("cuda" if use_gpu else "cpu")
        )
        matches.append(match)
    return torch.cat(match_ious), torch.cat(match_labels), matches


class TEM(torch.nn.Module):
    def __init__(self, num_classes, feat_dim=400, temporal_dim=100, c_hidden=512):
        super(TEM, self).__init__()

        self.conv1 = torch.nn.Conv1d(
            in_channels=feat_dim,
            out_channels=c_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=c_hidden,
            out_channels=c_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
        )
        self.conv3 = torch.nn.Conv1d(
            in_channels=c_hidden,
            out_channels=3 + num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        # B, T, C  = x.size()
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(0.01 * self.conv3(x))
        return x


class PEM(torch.nn.Module):
    def __init__(self, feat_dim=32, hidden_dim=512, output_dim=1):  # paper parameters
        super(PEM, self).__init__()

        self.fc1 = torch.nn.Linear(
            in_features=feat_dim, out_features=hidden_dim, bias=True
        )
        self.fc2 = torch.nn.Linear(
            in_features=hidden_dim, out_features=output_dim, bias=True
        )

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        # batch_num_proposals, C = x.size()
        x = F.relu(0.1 * self.fc1(x))
        x = torch.sigmoid(0.1 * self.fc2(x))
        return x


class BSNNet(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        sample_duration=100,
        backbone="inception",
        learnable_proportion=0.1,
        bsp_boundary_ratio=0.2,
        **kwargs
    ):
        super(BSNNet, self).__init__()

        self.bsp_boundary_ratio = bsp_boundary_ratio
        self.sample_duration = sample_duration
        self.encoder = CNNEncoder(
            learnable_proportion=learnable_proportion, archi=backbone
        )

        self.tem = TEM(num_classes=num_classes, feat_dim=CNN_embed_dim[backbone])

        self.pem = PEM()

    def forward(self, x):
        ## extract features from images
        x = self.encoder(x)
        ## extract actionness, start probability, end probability, classes probability
        tem_output = self.tem(x)
        ## generate proposals from start and end probabilities
        pgm_proposals, logits = zip(
            *[
                generateProposals(tem_output_i, temporal_scale=self.sample_duration)
                for tem_output_i in tem_output
            ]
        )

        bsp_features = [
            generateFeature(
                tem_output_i,
                pgm_i,
                temporal_scale=self.sample_duration,
                bsp_boundary_ratio=self.bsp_boundary_ratio,
                use_gpu=next(self.parameters()).is_cuda,
            )
            for pgm_i, tem_output_i in zip(pgm_proposals, tem_output)
        ]
        ## keep track of the positions of proposals when concatenating the tensor
        lens = [0] + list(map(lambda x: x.size(0), bsp_features))
        index_to_retrieve = np.cumsum(lens)
        bsp_features = torch.cat(bsp_features).to(
            "cuda" if next(self.parameters()).is_cuda else "cpu"
        )
        if bsp_features.size(0) == 0:
            return tem_output, torch.Tensor([]), logits

        pem_output = self.pem(bsp_features)
        pem_output = [
            pem_output[index_to_retrieve[i] : index_to_retrieve[i + 1]]
            for i in range(len(index_to_retrieve) - 1)
        ]
        pgm_proposals = list(
            map(
                lambda proposals: np.array([x[:2] for x in proposals]).astype(float),
                pgm_proposals,
            )
        )
        return (
            tem_output,
            torch.cat(pem_output),
            logits,
            list(
                map(
                    lambda x: torch.Tensor(x).to(
                        "cuda" if next(self.parameters()).is_cuda else "cpu"
                    ),
                    pgm_proposals,
                )
            ),
        )
