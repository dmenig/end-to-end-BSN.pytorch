from models.bsn import BSNNet, getIou, binarize_sequences
from losses import TEM_loss_function, PEM_loss_function, correctLogits


sample_duration = 10
use_gpu = False
batch_size = 2
image_size = 224


### Define Model
model = BSNNet(
    num_classes=1000,
    sample_duration=sample_duration,
    backbone="inception",
    learnable_proportion=0.1,
    bsp_boundary_ratio=0.2,
)

model.to("cuda" if use_gpu else "cpu")

### Define inputs

import torch
import numpy as np

# inputs would be a 5d batch of videos
inputs = (
    torch.from_numpy(
        np.random.choice(
            256, size=(batch_size, 3, sample_duration, image_size, image_size)
        )
    )
    .float()
    .add(-110)
    .div(40.0)
)
# raw_targets contains the detections for each video of the batch
# raw_targets = [[[class, t_start, t_end] for (class_ t_start, t_end) in video.metadata] for video in batch]
# where t_start, t_end are in [0, 1] (normalized by the sample_duration)

raw_targets = [[[666, 0.2, 0.7]], []]



### Forward pass and loss computation

label_action, label_start, label_end = binarize_sequences(raw_targets, sample_duration)
tem_output, pem_output, logits, proposals = model(inputs)
match_ious, match_labels, matches = getIou(proposals, raw_targets, use_gpu=use_gpu)
# correct proposals
loss_tem = TEM_loss_function(
    label_action, label_start, label_end, tem_output, use_gpu=use_gpu
)

## loss for optimizer
loss_total = loss_tem["cost"] + loss_iou


# correct iou predictions of proposals
loss_iou = PEM_loss_function(pem_output, match_ious, use_gpu=use_gpu)
# only correct labels of "right" proposals
new_logits = correctLogits(logits, matches, use_gpu=use_gpu)
if new_logits.size(0) != 0:
    loss_classif = criterions["classif"](new_logits, match_labels)
    total_loss += classif_loss

loss_total_untrimmed = loss_tem["cost"]
loss_action = loss_tem["loss_action"]
loss_start = loss_tem["loss_start"]
loss_end = loss_tem["loss_end"]
