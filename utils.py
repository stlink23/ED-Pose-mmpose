# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmengine.model import BaseModule, ModuleList
from torch import Tensor
import numpy as np

class FFN(BaseModule):
    """Very simple multi-layer perceptron with relu. Mostly used in DETR series
    detectors.

    Args:
        input_dim (int): Feature dim of the input tensor.
        hidden_dim (int): Feature dim of the hidden layer.
        output_dim (int): Feature dim of the output tensor.
        num_layers (int): Number of FFN layers..
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int) -> None:
        super().__init__()

        self.num_layers = num_layers

        self.layers = ModuleList()
        self.layers.append(Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(Linear(hidden_dim, hidden_dim))
        self.layers.append(Linear(hidden_dim, output_dim))

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of FFN.
        Args:
            x (Tensor): The input feature, has shape
                (num_queries, bs, input_dim).
        Returns:
            Tensor: The output feature, has shape
                (num_queries, bs, output_dim).
        """
        # 确保输入张量的数据类型与模型参数一致
        if x.dtype != self.layers[0].weight.dtype:
            x = x.to(self.layers[0].weight.dtype)
            
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
        return x


class PositionEmbeddingSineHW(BaseModule):
    """This is a more standard version of the position embedding, very similar
    to the one used by the Attention is all you need paper, generalized to work
    on images."""

    def __init__(self,
                 num_pos_feats=64,
                 temperatureH=10000,
                 temperatureW=10000,
                 normalize=False,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperatureH = temperatureH
        self.temperatureW = temperatureW
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask: Tensor):
        #直接传入mask，作者方法额外传如img
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_tx = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_tx = self.temperatureW**(2 * (dim_tx // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_tx

        dim_ty = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_ty = self.temperatureH**(2 * (dim_ty // 2) / self.num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_ty

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_boxes

def oks_overlaps(kpt_preds, kpt_gts, kpt_valids, kpt_areas, sigmas):
    sigmas = kpt_preds.new_tensor(sigmas)
    variances = (sigmas * 2)**2

    assert kpt_preds.size(0) == kpt_gts.size(0)
    kpt_preds = kpt_preds.reshape(-1, kpt_preds.size(-1) // 2, 2)
    kpt_gts = kpt_gts.reshape(-1, kpt_gts.size(-1) // 2, 2)

    squared_distance = (kpt_preds[:, :, 0] - kpt_gts[:, :, 0]) ** 2 + \
        (kpt_preds[:, :, 1] - kpt_gts[:, :, 1]) ** 2
    # import pdb
    # pdb.set_trace()
    # assert (kpt_valids.sum(-1) > 0).all()
    squared_distance0 = squared_distance / (kpt_areas[:, None] * variances[None, :] * 2)
    squared_distance1 = torch.exp(-squared_distance0)
    squared_distance1 = squared_distance1 * kpt_valids
    oks = squared_distance1.sum(dim=1) / (kpt_valids.sum(dim=1)+1e-6)

    return oks

def oks_loss(pred,
             target,
             valid=None,
             area=None,
             linear=False,
             sigmas=None,
             eps=1e-6):
    """Oks loss.
    Computing the oks loss between a set of predicted poses and target poses.
    The loss is calculated as negative log of oks.
    Args:
        pred (torch.Tensor): Predicted poses of format (x1, y1, x2, y2, ...),
            shape (n, 2K).
        target (torch.Tensor): Corresponding gt poses, shape (n, 2K).
        linear (bool, optional): If True, use linear scale of loss instead of
            log scale. Default: False.
        eps (float): Eps to avoid log(0).
    Return:
        torch.Tensor: Loss tensor.
    """
    oks = oks_overlaps(pred, target, valid, area, sigmas).clamp(min=eps)
    if linear:
        loss = 1 - oks
    else:
        loss = -oks.log()
    return loss

class OKSLoss(nn.Module):
    """IoULoss.
    Computing the oks loss between a set of predicted poses and target poses.
    Args:
        linear (bool): If True, use linear scale of loss instead of log scale.
            Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """
    def __init__(self,
                 linear=False,
                 num_keypoints=17,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0):
        super(OKSLoss, self).__init__()
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        if num_keypoints == 17:
            self.sigmas = np.array([
                .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07,
                1.07, .87, .87, .89, .89
            ], dtype=np.float32) / 10.0
        elif num_keypoints == 14:
            self.sigmas = np.array([
                .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89,
                .79, .79
            ]) / 10.0
        else:
            raise ValueError(f'Unsupported keypoints number {num_keypoints}')

    def forward(self,
                pred,
                target,
                valid,
                area,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            valid (torch.Tensor): The visible flag of the target pose.
            area (torch.Tensor): The area of the target pose.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * oks_loss(
            pred,
            target,
            valid=valid,
            area=area,
            linear=self.linear,
            sigmas=self.sigmas,
            eps=self.eps)
        return loss
