import torch
import torch.nn as nn
import ipdb
from typing import Optional, Tuple, Union


def weighted_procrustes(
    src_points: torch.Tensor,
    ref_points: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    weight_thresh: float = 0.0,
    eps: float = 1e-5,
    return_transform: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Compute rigid transformation from `src_points` to `ref_points` using weighted SVD.

    Modified from [PointDSC](https://github.com/XuyangBai/PointDSC/blob/master/models/common.py).

    Args:
        src_points: torch.Tensor (B, N, 3) or (N, 3)
        ref_points: torch.Tensor (B, N, 3) or (N, 3)
        weights: torch.Tensor (B, N) or (N,) (default: None)
        weight_thresh: float (default: 0.)
        eps: float (default: 1e-5)
        return_transform: bool (default: False)

    Returns:
        R: torch.Tensor (B, 3, 3) or (3, 3)
        t: torch.Tensor (B, 3) or (3,)
        transform: torch.Tensor (B, 4, 4) or (4, 4)
    """
    if src_points.ndim == 2:
        src_points = src_points.unsqueeze(0)
        ref_points = ref_points.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    device = src_points.device
    batch_size = src_points.shape[0]

    src_points_f = src_points.float()
    ref_points_f = ref_points.float()
    if weights is None:
        weights = torch.ones_like(src_points_f[:, :, 0])
    else:
        weights = weights.float()
    weights = torch.where(torch.lt(weights, weight_thresh), torch.zeros_like(weights), weights)
    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)
    weights = weights.unsqueeze(2)  # (B, N, 1)

    src_centroid = torch.sum(src_points_f * weights, dim=1, keepdim=True)  # (B, 1, 3)
    ref_centroid = torch.sum(ref_points_f * weights, dim=1, keepdim=True)  # (B, 1, 3)
    src_points_centered = src_points_f - src_centroid  # (B, N, 3)
    ref_points_centered = ref_points_f - ref_centroid  # (B, N, 3)

    H = src_points_centered.permute(0, 2, 1) @ (weights * ref_points_centered)
    H_cpu = H.float().cpu()
    U, _, V = torch.svd(H_cpu)  # H = USV^T
    Ut = U.transpose(1, 2).to(device=device, dtype=torch.float32)
    V = V.to(device=device, dtype=torch.float32)
    eye = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
    eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
    R = V @ eye @ Ut

    t = ref_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
    t = t.squeeze(2)

    if return_transform:
        transform = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).repeat(
            batch_size, 1, 1
        )
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t
        if squeeze_first:
            transform = transform.squeeze(0)
        return transform
    else:
        if squeeze_first:
            R = R.squeeze(0)
            t = t.squeeze(0)
        return R, t


class WeightedProcrustes(nn.Module):
    def __init__(
        self,
        weight_thresh: float = 0.0,
        eps: float = 1e-5,
        return_transform: bool = False,
    ) -> None:
        super(WeightedProcrustes, self).__init__()
        self.weight_thresh = weight_thresh
        self.eps = eps
        self.return_transform = return_transform

    def forward(
        self,
        src_points: torch.Tensor,
        tgt_points: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return weighted_procrustes(
            src_points,
            tgt_points,
            weights=weights,
            weight_thresh=self.weight_thresh,
            eps=self.eps,
            return_transform=self.return_transform,
        )
