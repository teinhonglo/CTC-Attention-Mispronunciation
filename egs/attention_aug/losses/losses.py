import torch
from torch import nn


class CTCLoss(nn.Module):
    def __init__(self, opts):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')
        self.use_focal_loss = opts.use_focal_loss
        print("use focal loss", self.use_focal_loss)
        self.gamma = 2.
        self.alpha = 1.

    def forward(self, predicts, labels, ref_labels, preds_lengths, label_lengths, ref_length):
        ctc_loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        if self.use_focal_loss:
            weight = 1. - torch.exp(-ctc_loss)
            #weight = torch.subtract(torch.tensor([1.0]), weight)
            weight = self.alpha * (weight ** self.gamma)
            focal_loss = torch.multiply(ctc_loss, weight)
            loss = focal_loss
        else:
            loss = ctc_loss
        loss = loss.sum()
        return loss

class CTCContrastLoss(nn.Module):
    def __init__(self, opts):
        super(CTCContrastLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')
        self.use_focal_loss = opts.use_focal_loss
        self.triplet_loss = nn.TripletMarginLoss(margin=16, p=2)

    def forward(self, predicts, labels, ref_labels, preds_lengths, label_lengths, ref_lengths):
        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)
            ref_loss = self.loss_func(predicts, ref_labels, preds_lengths, ref_lengths)
            anchor = torch.zeros(ctc_loss.shape).to(ctc_loss.device)

            margin_loss = self.triplet_loss(anchor.unsqueeze(1), ctc_loss.unsqueeze(1), ref_loss.unsqueeze(1))
            loss = ctc_loss.sum() + margin_loss.sum()

        return loss
