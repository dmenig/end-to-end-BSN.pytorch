# bsn.pytorch
End-to-End Boundary Sensitive Network in Pytorch


## Requirements

```
torch
torchvision
numpy
pretrainedmodels (https://github.com/Cadene/pretrained-models.pytorch)
```

## Data format

There's no DataLoader in this. You just have to be able to carry your detection labels to the model with the form exposed in the `example.py` : 
```
# raw_targets contains the detections for each video of the batch
# raw_targets = [[[class, t_start, t_end] for (class_ t_start, t_end) in video.metadata] for video in batch]
# where t_start, t_end are in [0, 1] (normalized by the sample_duration)
```


## Motivation

It is meant to provide a real end-to-end training fashion, as advertised in the paper (although I don't recommend training the full unfrozen feature extractor by this manner).
My goal was also to be able to use samples with no action, which are still useful to learn good features. So I modified the places in the code that assumed there was some action. In particular : for proposal generation, I follow the rule of local maxima extraction, with the condition that those are higher than a threshold.
I also wanted to add a classification part, which you can remove to retrieve the exact results of the paper if you want. I simply computed it alongside the TEM outputs, and obtained the class predicted by averaging logits along a proposal.

## Credits

This git holds the architecture of the network exposed in this paper: https://arxiv.org/pdf/1806.02964.pdf

There was already some pytorch code release, but I didn't feel like it could be easily usable for any other dataset the THUMOS14 and ActivityNet https://github.com/wzmsltw/BSN-boundary-sensitive-network. Thank you anyway for the great work.

So I took it, reworked it and here you have it.
