import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


def bbox_deduplicate(bbox, padding="zero", dedup_repeat=False):
    """
    Args:
        bbox:  Nx10     [[3dbbox]+[2dbbox], ...]
        padding:        zero / repeat
        dedup_repeat:   dedup repeat bbox when padding=="zero".
    Returns:
    """

    if isinstance(bbox, torch.Tensor):
        pass
    elif isinstance(bbox, np.ndarray):
        bbox = torch.from_numpy(bbox)
    else:
        raise NotImplementedError()

    max_surf = bbox.shape[-2]
    dedup_mask = np.zeros((max_surf), dtype=np.bool)

    if padding == "repeat":
        bbox_threshold = 0.08
    elif padding == "zero":
        bbox_threshold = 2e-4

    bboxes = torch.concatenate(
        [bbox[0][:, :6].unflatten(-1, torch.Size([2, 3])), bbox[0][:, 6:].unflatten(-1, torch.Size([2, 2]))]
        , dim=-1).detach().cpu().numpy()

    non_repeat = None
    for bbox_idx, bbox in enumerate(bboxes):
        if padding == "repeat":
            if non_repeat is None:
                non_repeat = bbox[np.newaxis, :, :]
                is_deduped = False
            else:
                diff = np.max(np.max(np.abs(non_repeat - bbox)[..., -2:], -1), -1)  #
                same = diff < bbox_threshold
                bbox_rev = bbox[::-1]  # also test reverse bbox for matching
                diff_rev = np.max(np.max(np.abs(non_repeat - bbox_rev)[..., -2:], -1), -1)  # [...,-2:]
                same_rev = diff_rev < bbox_threshold
                if same.sum() >= 1 or same_rev.sum() >= 1:
                    is_deduped = True
                else:
                    is_deduped = False
                    non_repeat = np.concatenate([non_repeat, bbox[np.newaxis, :, :]], 0)

        if padding == "zero":
            is_deduped = False
            v = 1
            for h in (bbox[1] - bbox[0])[3:]:
                v *= h
            if v < bbox_threshold:
                is_deduped = True
            elif dedup_repeat and non_repeat is not None:  # remove repeated bbox
                bbox_threshold_2 = 0.02
                diff = np.max(np.max(np.abs(non_repeat - bbox)[..., :3], -1), -1)  #
                same = diff < bbox_threshold_2
                bbox_rev = bbox[::-1]  # also test reverse bbox for matching
                diff_rev = np.max(np.max(np.abs(non_repeat - bbox_rev)[..., :3], -1), -1)  # [...,-2:]
                same_rev = diff_rev < bbox_threshold_2
                if same.sum() >= 1 or same_rev.sum() >= 1:
                    is_deduped = True
            if is_deduped == False:
                if non_repeat is None:
                    non_repeat = bbox[np.newaxis, :, :]
                else:
                    non_repeat = np.concatenate([non_repeat, bbox[np.newaxis, :, :]], 0)

        dedup_mask[bbox_idx] = is_deduped

    bboxes = np.concatenate([non_repeat[:, :, :3].reshape(len(non_repeat), -1), non_repeat[:, :, 3:].reshape(len(non_repeat), -1)], axis=-1)

    return bboxes, dedup_mask




def get_diff_map(A, B):
    """
    matching panels by bbox
    Args:
        A: NxD
        B: NxD
    Returns:
    """
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    if isinstance(B, torch.Tensor):
        B = B.detach().cpu().numpy()

    diff_map = abs(A[:, None, :] - B[None, :, :])

    diff_map = np.abs(np.mean(diff_map, axis=-1))
    # _, min_indices = torch.min(diff_map, dim=1)

    row_ind, col_ind = linear_sum_assignment(diff_map)

    cost = diff_map[row_ind, col_ind]
    cost_total = cost.sum()

    return diff_map, cost, cost_total, row_ind, col_ind


def bbox_3d_iou(pred_bboxes, gt_bboxes):
    pred_3d = pred_bboxes[:, :6]  # [N, 6]
    gt_3d = gt_bboxes[:, :6]      # [N, 6]

    x_min_inter = np.maximum(pred_3d[:, 0], gt_3d[:, 0])
    y_min_inter = np.maximum(pred_3d[:, 1], gt_3d[:, 1])
    z_min_inter = np.maximum(pred_3d[:, 2], gt_3d[:, 2])
    x_max_inter = np.minimum(pred_3d[:, 3], gt_3d[:, 3])
    y_max_inter = np.minimum(pred_3d[:, 4], gt_3d[:, 4])
    z_max_inter = np.minimum(pred_3d[:, 5], gt_3d[:, 5])

    inter_lengths = np.stack([
        x_max_inter - x_min_inter,
        y_max_inter - y_min_inter,
        z_max_inter - z_min_inter
    ], axis=1)
    inter_lengths = np.maximum(inter_lengths, 0)
    inter_volume = np.prod(inter_lengths, axis=1)

    pred_lengths = pred_3d[:, 3:6] - pred_3d[:, 0:3]
    gt_lengths = gt_3d[:, 3:6] - gt_3d[:, 0:3]
    pred_lengths = np.maximum(pred_lengths, 0)
    gt_lengths = np.maximum(gt_lengths, 0)
    pred_volume = np.prod(pred_lengths, axis=1)
    gt_volume = np.prod(gt_lengths, axis=1)

    union_volume = pred_volume + gt_volume - inter_volume

    iou_3d = np.where(union_volume > 0, inter_volume / union_volume, 0.0)
    return iou_3d


def bbox_2d_iou(pred_bboxes, gt_bboxes):
    pred_2d = pred_bboxes  # [N, 4]
    gt_2d = gt_bboxes      # [N, 4]

    x_min_inter = np.maximum(pred_2d[:, 0], gt_2d[:, 0])
    y_min_inter = np.maximum(pred_2d[:, 1], gt_2d[:, 1])
    x_max_inter = np.minimum(pred_2d[:, 2], gt_2d[:, 2])
    y_max_inter = np.minimum(pred_2d[:, 3], gt_2d[:, 3])

    inter_lengths = np.stack([
        x_max_inter - x_min_inter,
        y_max_inter - y_min_inter
    ], axis=1)
    inter_lengths = np.maximum(inter_lengths, 0)
    inter_area = np.prod(inter_lengths, axis=1)

    pred_lengths = pred_2d[:, 2:4] - pred_2d[:, 0:2]
    gt_lengths = gt_2d[:, 2:4] - gt_2d[:, 0:2]
    pred_lengths = np.maximum(pred_lengths, 0)
    gt_lengths = np.maximum(gt_lengths, 0)
    pred_area = np.prod(pred_lengths, axis=1)
    gt_area = np.prod(gt_lengths, axis=1)

    union_area = pred_area + gt_area - inter_area

    iou_2d = np.where(union_area > 0, inter_area / union_area, 0.0)
    return iou_2d


def evaluate_bboxes_iou(pred_bboxes, gt_bboxes, indices_2d=[6, 7, 8, 9]):
    """
    Example of usage

    pred_bboxes Nx10
    gt_bboxes   Nx10
    """
    raise NotImplementedError

    ious_3d = [bbox_3d_iou(pred_bboxes[i], gt_bboxes[i]) for i in range(len(pred_bboxes))]
    ious_2d = [bbox_2d_iou(pred_bboxes[i][indices_2d], gt_bboxes[i][indices_2d]) for i in range(len(pred_bboxes))]

    mIoU_3d = sum(ious_3d) / len(ious_3d) if ious_3d else 0.0
    mIoU_2d = sum(ious_2d) / len(ious_2d) if ious_2d else 0.0

    return mIoU_3d, mIoU_2d


def bbox_l2_distance(pred_bbox, gt_bbox):
    return np.sqrt(np.sum((pred_bbox - gt_bbox) ** 2))


def get_bbox(point_cloud):
    """
    Get the tighest fitting 3D bounding box giving a set of points (axis-aligned)
    """
    # Find the minimum and maximum coordinates along each axis
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])

    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])

    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])

    # Create the 3D bounding box using the min and max values
    min_point = np.array([min_x, min_y, min_z])
    max_point = np.array([max_x, max_y, max_z])
    return min_point, max_point
