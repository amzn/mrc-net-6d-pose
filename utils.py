import torch
import pytorch3d
import numpy as np
import config as cfg
import torch.nn.functional as F
import config as bop_cfg
from collections import defaultdict


PI = 3.141592653589793238462643383279502884
INF = 1e20
EPS = 1e-10


MAX_EXPERIMENTS = 100


def transform_bounding_box(bbox_xyxy, T):
    """
    Apply affine transform to bounding box coordinates
    """
    x1, y1, x2, y2 = bbox_xyxy
    verts = np.array([
        [x1, y1, 1.0],
        [x1, y2, 1.0],
        [x2, y1, 1.0],
        [x2, y2, 1.0]])
    verts_T = verts @ np.transpose(T)
    bbox_T = np.array([verts_T[:, 0].min(), verts_T[:, 1].min(),
                       verts_T[:, 0].max(), verts_T[:, 1].max()])
    return bbox_T


def make_quaternion_prototypes(device='cpu', n_side=4):
    """
    Generate quaternions uniformly on SO(3) using Hopf coordinates
    and HEALPix grids

    Outputs:
        prototypes: generated quaternion prototypes (n_grid, 4)
        where n_grid == 12*n_side**3*sqrt(12*PI)
    """
    # Uniformly sample the 3D sphere
    n_pix = 12 * n_side**2
    p = torch.arange(12*n_side**2, device=device)
    ph = (p + 1) / 2
    i1 = torch.floor(torch.sqrt(ph - torch.sqrt(torch.floor(ph)))) + 1
    j1 = p + 1 - 2*i1*(i1 - 1)
    valid = torch.logical_and(i1 < n_side, j1 <= 4*i1)
    i1 = i1[valid]
    j1 = j1[valid]
    z1 = 1 - i1**2 / 3 / n_side**2
    s = 1
    phi1 = PI / 2 / i1 * (j1 - s / 2)
    theta1 = torch.acos(z1)

    ph = p - 2*n_side*(n_side - 1)
    i2 = torch.floor(ph / 4 / n_side) + n_side
    j2 = ph % (4*n_side) + 1
    j0 = j2[i2 == 2*n_side]
    s = (n_side + 1) % 2
    phi0 = PI / 2 / n_side * (j0 - s / 2)
    theta0 = PI / 2 * torch.ones_like(phi0)
    valid = torch.logical_and(n_side <= i2, i2 < 2*n_side)
    i2 = i2[valid]
    j2 = j2[valid]
    z2 = 4 / 3 - 2*i2 / 3 / n_side
    s = (i2 - n_side + 1) % 2
    phi2 = PI / 2 / n_side * (j2 - s / 2)
    theta2 = torch.acos(z2)

    theta = torch.concat([
        theta1, PI - theta1, theta2,
        PI - theta2, theta0], dim=0).reshape(1, -1)
    phi = torch.concat([
        phi1, phi1, phi2, phi2, phi0], dim=0).reshape(1, -1)

    # Generate prototypes using Hopf fibration
    n1 = torch.floor(torch.sqrt(torch.tensor(PI * n_pix, device=device)))
    psi = torch.arange(0, 2*PI, step=2*PI / n1, device=device).reshape(-1, 1)
    x1 = torch.cos(theta/2) * torch.cos(psi/2)
    x2 = torch.sin(theta/2) * torch.sin(phi - psi/2)
    x3 = torch.sin(theta/2) * torch.cos(phi - psi/2)
    x4 = torch.cos(theta/2) * torch.sin(psi/2)

    prototypes = torch.stack([x1, x2, x3, x4], dim=-1).reshape(-1, 4)
    prototypes = prototypes / torch.norm(prototypes, dim=-1, keepdim=True)
    return prototypes


@torch.no_grad()
def quantize_quaternion_vertex(quaternions, vertices, vertices_mask, diameter,
                               v_corr, q_sym, t_sym, mask_sym,
                               sigma=cfg.POSE_SIGMA):
    """
    Quantize quaternion into bins based on vertex errors

    Inputs:
      quaternions: quaternions to quantize against (N, 4)
      vertices: vertices of object model (N, V, 3)
      vertices_mask: takes value 1 wherever the vertex is valid (N, V)
      diameter: object diameter provided by CAD models (?)
      v_corr: correlation matrix of the object vertices (N, 3, 3)
      q_sym: rotational symmetry group (N, S, 4)
      t_sym: translation symmetry group (N, S, 3)
      mask_sym: indicates whether each symmetric transformation is valid (N, S)
      sigma: standard deviation of the Gaussian kernel in radians

    Outputs:
      categories: probabilities of belonging to each bin (?, M)
      prototypes: M quaternion prototypes (M, 4)
      residuals: residual quaternion w.r.t. each prototypes (?, M, 4)
    """
    prototypes = make_quaternion_prototypes(device=quaternions.device)
    num_bins = len(prototypes)
    prototypes_conj = prototypes * torch.tensor(
        [[1., -1., -1., -1.]], device=quaternions.device)
    expanded_dims = [1]*len(quaternions.shape[:-1]) + [num_bins, 4]
    prototypes_rep = torch.reshape(prototypes, expanded_dims).expand(
        list(quaternions.shape[:-1]) + [-1, -1])
    prototypes_conj_rep = torch.reshape(prototypes_conj, expanded_dims)

    quat_rep = torch.repeat_interleave(
        quaternions.unsqueeze(-2), num_bins, dim=-2)
    trans_rep = torch.zeros(*quat_rep.shape[:-1], 3).to(quat_rep)
    v_corr_rep = torch.repeat_interleave(
        v_corr.unsqueeze(-3), num_bins, dim=-3)
    q_sym_rep = torch.repeat_interleave(
        q_sym.unsqueeze(-3), num_bins, dim=-3)
    t_sym_rep = torch.repeat_interleave(
        t_sym.unsqueeze(-3), num_bins, dim=-3)
    mask_sym_rep = torch.repeat_interleave(
        mask_sym.unsqueeze(-2), num_bins, dim=-2)
    num_sym = mask_sym.shape[-1]
    quat_matched, trans_matched = top_matched_pose(
        quat_rep.view(-1, 4), trans_rep.view(-1, 3),
        prototypes_rep.view(-1, 4), trans_rep.view(-1, 3),
        v_corr_rep.view(-1, 3, 3), None, q_sym_rep.view(-1, num_sym, 4),
        t_sym_rep.view(-1, num_sym, 3), mask_sym_rep.view(-1, num_sym))
    quat_matched = quat_matched.reshape(-1, num_bins, 4)
    trans_matched = trans_matched.reshape(-1, num_bins, 3)
    diff_vertex = vertex_loss_quaternion(
        quat_matched, trans_matched, prototypes_rep, trans_rep, vertices,
        vertices_mask, None, loss_type='l1', reduction='none')
    residuals = multiply_quaternion(quat_matched, prototypes_conj_rep)
    categories = torch.exp(-diff_vertex / sigma / diameter.unsqueeze(-1))
    return categories, prototypes, residuals


def quaternion_from_bin_logits(logits):
    """
    Reconstruct coarse quaternions from bin logits
    Inputs:
      logits: logits of quaternion belonging to each bin (N, M)

    Outputs:
      quaternions: reconstructed quaternions (N, 4)
    """
    batch_size, num_bins = logits.shape
    prototypes = make_quaternion_prototypes(device=logits.device)
    prototypes_rep = torch.repeat_interleave(
        prototypes.unsqueeze(0), batch_size, dim=0)
    indices = torch.repeat_interleave(torch.argmax(
        logits, dim=-1, keepdim=True), repeats=4, dim=1)
    quaternions = torch.gather(
        prototypes_rep, dim=1, index=indices.unsqueeze(1))
    return torch.squeeze(quaternions, dim=1)


def quaternion_from_prototypes(logits, residual):
    """
    Reconstruct quaternions from bin logits and residuals per bin
    Inputs:
      logits: logits of quaternion belonging to each bin (N, M)
      residual: (flattened) residual quaternions for each bin (N, 4)

    Outputs:
      quaternions: reconstructed quaternions (N, 4)
    """
    batch_size, num_bins = logits.shape
    residual = torch.reshape(residual, [batch_size, 1, -1])
    prototypes = make_quaternion_prototypes(device=logits.device)
    prototypes_rep = torch.repeat_interleave(
        prototypes.unsqueeze(0), len(residual), dim=0)
    candidates = multiply_quaternion(residual, prototypes_rep)
    indices = torch.repeat_interleave(torch.argmax(
        logits, dim=-1, keepdim=True), repeats=residual.shape[-1], dim=1)
    quaternions = torch.gather(candidates, dim=1, index=indices.unsqueeze(1))
    return torch.squeeze(quaternions, dim=1)


def rotation_from_prototypes(logits, residual_6d):
    """
    Reconstruct rotations from bin logits and residuals per bin
    Inputs:
      logits: logits of rotation belonging to each bin (N, M)
      residual_6d: (flattened) residual rotations (N, 6)

    Outputs:
      rotations: reconstructed rotations (N, 3, 3)
    """
    batch_size = len(logits)
    rot_res = compute_rotation_matrix_from_ortho6d(
        residual_6d.reshape(batch_size, -1, 6))
    prototypes = make_quaternion_prototypes(device=logits.device)
    prototypes_rep = torch.repeat_interleave(
        prototypes.unsqueeze(0), batch_size, dim=0)
    rot_protos = quaternion_to_rotation(prototypes_rep)
    rot_candidates = rot_res @ rot_protos
    indices = torch.repeat_interleave(torch.argmax(
        logits, dim=-1, keepdim=True), repeats=3, dim=-1)
    indices = torch.repeat_interleave(
        indices.unsqueeze(-1), repeats=3, dim=-1).unsqueeze(1)
    rotations = torch.gather(rot_candidates, dim=1, index=indices)
    return torch.squeeze(rotations, dim=1)


def make_depth_bins(depth_min, depth_max, n_depth_bin, device='cpu'):
    """
    Generate bin centers on uniform depth grid
    Inputs:
      depth_min, depth_max: specifies perspective depth range
      n_depth_bin: number of bins to quantize depth into

    Outputs:
      depth_bins: depth bin centers (1, C)
    """
    spacing = (depth_max - depth_min) / n_depth_bin
    index = torch.arange(n_depth_bin, device=device).unsqueeze(0)
    depth_bins = depth_min + (index + 0.5) * spacing
    return depth_bins


def quantize_depth(depth, depth_min, depth_max, n_depth_bin,
                   sigma=cfg.DEPTH_SIGMA):
    """
    Quantize depth uniformly into each bin
    Inputs:
      depth: perspective depth (N,)
      (depth_min, depth_max): range of perspective depth
      n_depth_bin: number of bins to quantize depth into

    Outputs:
      depth_bin: depth probabilities in each bin (N, C)
      depth_res: residual depth per bin (N, C)
    """
    spacing = (depth_max - depth_min) / n_depth_bin
    depth_bins = make_depth_bins(
        depth_min, depth_max, n_depth_bin, device=depth.device)
    depth_res = depth.unsqueeze(-1) - depth_bins
    depth_prob = torch.softmax(-depth_res**2 / sigma**2 / spacing**2, dim=-1)
    return depth_prob, depth_res


def depth_from_bin_logits(depth_scores, depth_res,
                          depth_min, depth_max, n_depth_bin):
    """
    Reconstruct depth from predicted logits
    Inputs:
      depth_scores: logits of depth belonging to each bin (N, M)
      depth_res: residual of depth per bin (N, 1)
      (depth_min, depth_max): range of perspective depth
      n_depth_bin: number of bins to quantize depth into

    Outputs:
      depth: reconstructed depth (N, 1)
    """
    # index = torch.count_nonzero(depth_scores >= 0.0, dim=-1).unsqueeze(-1)
    index = torch.argmax(depth_scores, dim=-1).unsqueeze(-1)
    depth_bins = make_depth_bins(
        depth_min, depth_max, n_depth_bin, device=depth_scores.device)
    batch_size = len(depth_scores)
    depth_bins = torch.repeat_interleave(
        depth_bins, batch_size, dim=0)
    depth_per_bin = depth_bins + depth_res
    depth = torch.gather(depth_per_bin, dim=-1, index=index)
    return depth


def make_grid_2d(batch_dims, device='cpu'):
    """
    Generate 2D grid for heatmap regression reshaped with batch dimensions

    Inputs:
      batch_dims: int number of batch dimensions
      device: torch.device where to place the grid tensors

    Outputs:
      batch_grid_x, batch_grid_y: x, y grid (?, C, C)
    """
    spacing = 1.0 / cfg.CELL_SIZE
    xx = torch.arange(0, cfg.CELL_SIZE) + 0.5
    yy = torch.arange(0, cfg.CELL_SIZE) + 0.5
    grid_x, grid_y = torch.meshgrid(xx, yy, indexing='xy')
    grid_x = grid_x.to(device)
    grid_y = grid_y.to(device)
    batch_grid_x = spacing * torch.reshape(
        grid_x, [1]*batch_dims + [cfg.CELL_SIZE, cfg.CELL_SIZE])
    batch_grid_y = spacing * torch.reshape(
        grid_y, [1]*batch_dims + [cfg.CELL_SIZE, cfg.CELL_SIZE])
    return batch_grid_x, batch_grid_y


def trans_from_bins(trans_logits, trans_res):
    """
    Reconstruct 2D translation in normalized pixel coordinates

    Inputs:
      trans_logits: logits for each translation bin (N, C * C)
      trans_res: residual perspective translation in 2D (N, 2)

    Outputs:
      trans_xy: 2D translation in normalized pixel coordinates (N, 2)
    """
    batch_grid_x, batch_grid_y = make_grid_2d(1, device=trans_logits.device)
    trans_pred = torch.stack([
        trans_res[:, :1] + batch_grid_x.view(1, -1) - 0.5,
        trans_res[:, 1:] + batch_grid_y.view(1, -1) - 0.5], dim=-1)
    return top_score_prediction(trans_pred, trans_logits)


def top_score_prediction(candidates, scores):
    """"
    Select the best prediction with the highest score
    Inputs:
      candidates: candidates per class (?, M, D)
      scores: score for each class (?, M)

    Outputs:
      pred: selected prediction (?, D)
    """
    dim = candidates.shape[-1]
    index = torch.repeat_interleave(torch.argmax(
        scores, dim=-1, keepdim=True), repeats=dim, dim=-1)  # (?, D)
    pred = torch.gather(
        candidates, dim=-2, index=index.unsqueeze(-2))
    return torch.squeeze(pred, dim=-2)


@torch.no_grad()
def top_matched_pose(q_src, t_src, q_tgt, t_tgt, v_corr, v_mean,
                     q_sym, t_sym, mask_sym):
    """
    Select the rotation from q_src that is closest to q_tgt
    among the whole symmetry group specified by q_sym
    Inputs:
      q_src: source rotation (N, 4)
      t_src: source translation (N, 3)
      q_tgt: target rotation (N, 4)
      t_tgt: target translation (N, 3)
      v_corr: correlation matrix of the object vertices (N, 3, 3)
      v_mean: mean of the object vertices (N, 3)
      q_sym: rotational symmetry group (N, S, 4)
      t_sym: translation symmetry group (N, S, 3)
      mask_sym: indicates whether each symmetric transformation is valid (N, S)

    Outputs:
     q_match: best matched quaternion (N, 4)
     t_match: best matched translation (N, 3)
    """

    # Determine target rotation accounting for symmetries
    q_src_rep = q_src.unsqueeze(1)
    R_src_rep = quaternion_to_rotation(q_src_rep)
    R_src_sym = R_src_rep @ quaternion_to_rotation(q_sym)
    R_tgt = quaternion_to_rotation(q_tgt)
    R_tgt_trans = torch.transpose(
        R_tgt.unsqueeze(1), dim0=-1, dim1=-2)
    t_src_sym = torch.squeeze(
        R_src_rep @ t_sym.unsqueeze(-1), dim=-1) + t_src.unsqueeze(1)

    v_corr = v_corr.unsqueeze(1)
    rots_corr = torch.einsum(
        '...ij,...ji->...', v_corr @ R_tgt_trans, R_src_sym)
    rots_corr[torch.logical_not(mask_sym)] = -INF

    max_index = torch.argmax(rots_corr, dim=1, keepdim=True)
    q_src_sym = multiply_quaternion(q_src_rep, q_sym)
    q_match = torch.squeeze(torch.gather(
        q_src_sym, dim=1, index=max_index.unsqueeze(-1).expand(
            [-1] * len(max_index.shape) + [4])), 1)

    # Determine target translation accounting for symmetries
    t_match = torch.squeeze(torch.gather(
        t_src_sym, dim=1, index=max_index.unsqueeze(-1).expand(
            [-1] * len(max_index.shape) + [3])), 1)
    return q_match, t_match


def cross_entropy(one_hot_labels, logits, reduction='mean'):
    """
    Cross entropy loss with soft labels for classification
    Inputs:
      one_hot_labels: soft class assignments per sample (?, M)
      logits: predicted logit scores in each class (?, M)

    Outputs:
      batch_loss: batch loss (?,)
    """
    batch_loss = torch.sum(one_hot_labels * -F.log_softmax(logits), dim=-1)
    if reduction == 'mean':
        loss = torch.mean(batch_loss)
    elif reduction == 'sum':
        loss = torch.sum(batch_loss)
    elif reduction == 'none':
        loss = batch_loss
    else:
        raise ValueError('Invalid reduction type specified')
    return loss


def softmax_2d(logits):
    """
    Apply softmax over the innermost two dimensions

    Inputs:
      logits: raw logistic scores (?, H, W)

    Outputs:
      weights: probabilities in (0, 1) after applying softmax (?, H, W)
    """
    logits_flat = torch.flatten(logits, start_dim=-2)  # (?, H*W)
    weights = torch.softmax(logits_flat, dim=-1)
    weights = torch.unflatten(weights, dim=-1, sizes=logits.shape[-2:])
    return weights


def focal_loss(one_hot_labels, logits, alpha=2., beta=1., weights=[1.0, 1.0],
               reduction='mean'):
    """
    Focal loss with soft labels for classification
    Inputs:
      one_hot_labels: soft class assignments per sample (?, M)
      logits: predicted logit scores in each class (?, M)

    Outputs:
      batch_loss: batch focal loss (?,)
    """
    prob = torch.sigmoid(logits)
    pos = weights[0] * one_hot_labels**beta * -F.logsigmoid(logits)
    neg = weights[1] * (1 - one_hot_labels**beta) * -F.logsigmoid(-logits)
    batch_loss = torch.sum(
        (1 - prob)**alpha * pos + prob**alpha * neg, dim=-1)
    if reduction == 'mean':
        loss = torch.mean(batch_loss)
    elif reduction == 'sum':
        loss = torch.sum(batch_loss)
    elif reduction == 'none':
        loss = batch_loss
    else:
        raise ValueError('Invalid reduction type specified')
    return loss


def multiclass_focal_loss(one_hot_labels, logits, alpha=2., weights=None,
                          reduction='mean'):
    """
    Focal loss on multiclass classification
    Inputs:
      one_hot_labels: soft class assignments per sample (?, C)
      logits: predicted logit scores in each class (?, C)

    Outputs:
      batch_loss: batch focal loss (?,)
    """
    if weights is None:
        weights = torch.ones_like(one_hot_labels)
    prob = torch.sum(one_hot_labels * torch.softmax(logits, dim=-1), dim=-1)
    cross_entropy = torch.sum(
        weights * one_hot_labels * F.log_softmax(logits, dim=-1), dim=-1)
    batch_loss = -(1 - prob)**alpha * cross_entropy
    if reduction == 'mean':
        loss = torch.mean(batch_loss)
    elif reduction == 'sum':
        loss = torch.sum(batch_loss)
    elif reduction == 'none':
        loss = batch_loss
    else:
        raise ValueError('Invalid reduction type specified')
    return loss


def trans_3d_to_perspective(trans_3d, image_size, intrinsics):
    """
    Convert 3D translations into perspective translations

    Inputs:
      trans_3d: ground truth translations in 3D (?, 3)
      image_size: image size in (height, width)
      intrinsics: camera intrinsic matrices (?, 3, 3)

    Outputs:
     trans_persp: perspective 3D translation (?, 3)
    """
    h, w = image_size
    fx, fy = intrinsics[..., 0, 0], intrinsics[..., 1, 1]
    cx, cy = intrinsics[..., 0, 2], intrinsics[..., 1, 2]
    px = fx * trans_3d[..., 0] / trans_3d[..., 2] + cx
    py = fy * trans_3d[..., 1] / trans_3d[..., 2] + cy
    dx = (px - w / 2) / w
    dy = (py - h / 2) / h
    dz = normalize_depth(trans_3d[..., -1], intrinsics)
    trans_persp = torch.stack([dx, dy, dz], dim=-1)
    return trans_persp


def perspective_to_trans_3d(trans_pred, image_size, intrinsics):
    """
    Convert perspective translations (model predictions) to 3D translations

    Inputs:
      trans_pred: raw (perspective) translation predictions (?, 3)

    Outputs:
      trans_3d: translations in 3D (?, 3)
    """
    h, w = image_size
    fx, cx = intrinsics[..., 0, 0], intrinsics[..., 0, 2]
    fy, cy = intrinsics[..., 1, 1], intrinsics[..., 1, 2]
    tz = unnormalize_depth(trans_pred[..., -1], intrinsics)
    tx = tz / fx * (w * trans_pred[..., 0] + w / 2 - cx)
    ty = tz / fy * (h * trans_pred[..., 1] + h / 2 - cy)
    trans_3d = torch.stack([tx, ty, tz], dim=-1)
    return trans_3d


def smooth_l1_loss(error, beta, reduction='none'):
    """A modified version of the function from fvcore:
    https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/smooth_l1_loss.py

    Smooth L1 loss defined in the Fast R-CNN paper as:
    ::
                      | 0.5 * x ** 2 / beta + 0.5 * beta   if abs(x) < beta
        smoothl1(x) = |
                      | abs(x)                             otherwise,
    where x = ||error||_2.

    Smooth L1 loss can be seen as exactly L1 loss, but with the abs(x) < beta
    portion replaced with a quadratic function such that at abs(x) = beta, its
    slope is 1. The quadratic segment smooths the L1 loss near x = 0.

    Args:
        input (Tensor): input tensor of shape (*, 3)
        target (Tensor): target value tensor with the same shape as input
        beta (float): L1 to L2 change point.
            For beta values < 1e-5, L1 loss is computed.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        The loss with the reduction option applied.
    """
    if beta < 1e-5:
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = torch.norm(error, dim=-1)
    else:
        n = torch.norm(error, dim=-1)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n**2 / beta + 0.5 * beta, n)

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def compute_rotation_matrix_from_ortho6d(poses):
    """
    Code from https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html
    """
    assert poses.shape[-1] == 6
    x_raw = poses[..., 0:3]
    y_raw = poses[..., 3:6]
    x = x_raw / torch.norm(x_raw, p=2, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, p=2, dim=-1, keepdim=True)
    y = torch.cross(z, x, dim=-1)
    matrix = torch.stack((x, y, z), -1)
    return matrix


def vertex_loss_rotation(R_pred, t_pred, q_gt, t_gt, vertices, vertices_mask,
                         intrinsics, reduction='mean', vertex_reduction='mean',
                         loss_type='smooth_l1', do_projection=False):
    """
    Loss function for measuring the difference of 6d rotation

    Inputs:
      R_pred: predicted rotation (N, ?, 3, 3)
      t_pred: predicted translation in 3D (N, ?, 3)
      q_gt: ground truth quaternions (N, ?, 4)
      t_gt: groundtruth 3D translation (N, ?, 3)
      vertices: vertices of object model (N, V, 3)
      vertices_mask: takes value 1 wherever the vertex is valid (N, V)

    Outputs:
      loss: scalar loss value
    """

    R_gt = quaternion_to_rotation(q_gt)
    pad_dims = [1] * (len(R_gt.shape) - len(vertices.shape))
    vertices = vertices.view([
        len(vertices)] + pad_dims + list(vertices.shape[1:]))
    vertices_mask = vertices_mask.view(vertices.shape[:-1])

    if do_projection:
        v_pred = vertices @ torch.transpose(R_pred, dim0=-1, dim1=-2)
        v_gt = vertices @ torch.transpose(R_gt, dim0=-1, dim1=-2)
        if t_pred is not None and t_gt is not None:
            v_pred += torch.unsqueeze(t_pred, dim=-2)
            v_gt += torch.unsqueeze(t_gt, dim=-2)
        v_pred = v_pred @ torch.transpose(intrinsics, dim0=-1, dim1=-2)
        v_gt = v_gt @ torch.transpose(intrinsics, dim0=-1, dim1=-2)
        v_diff = v_pred[..., :-1] / v_pred[..., -1:] \
            - v_gt[..., :-1] / v_gt[..., -1:]
    else:
        v_diff = vertices @ torch.transpose(
            R_pred - R_gt, dim0=-1, dim1=-2)
        if t_pred is not None and t_gt is not None:
            v_diff += torch.unsqueeze(t_pred - t_gt, dim=-2)

    if loss_type == 'smooth_l1':
        losses = smooth_l1_loss(v_diff, beta=1.0, reduction='none')
    else:
        losses = torch.linalg.norm(v_diff, dim=-1)

    if vertex_reduction == 'mean':
        all_loss = torch.sum(
            losses * vertices_mask, dim=-1) / (
                torch.sum(vertices_mask, dim=-1) + 1e-3)
    else:
        all_loss = torch.max(
            losses * vertices_mask - ~vertices_mask * INF, dim=-1).values

    if reduction == 'mean':
        loss = torch.mean(all_loss)
    elif reduction == 'sum':
        loss = torch.sum(all_loss)
    elif reduction == 'none':
        loss = all_loss
    else:
        raise ValueError('Invalid reduction type specified!')
    return loss


def vertex_loss_quaternion(q_pred, t_pred, q_gt, t_gt, vertices, vertices_mask,
                           intrinsics, reduction='mean',
                           vertex_reduction='mean', loss_type='smooth_l1',
                           do_projection=False):
    """
    Loss function for measuring the difference of quaternions

    Inputs:
      q_pred: predicted quaternion (N, ?, 4)
      t_pred: predicted translation in 3D (N, ?, 3)
      q_gt: ground truth quaternion (N, ?, 4)
      t_gt: groundtruth 3D translation (N, ?, 3)
      vertices: vertices of object model (N, V, 3)
      vertices_mask: takes value 1 wherever the vertex is valid (N, V)

    Outputs:
      loss: scalar loss value
    """

    R_pred = quaternion_to_rotation(q_pred)
    loss = vertex_loss_rotation(
        R_pred, t_pred, q_gt, t_gt, vertices, vertices_mask, intrinsics,
        reduction, vertex_reduction, loss_type, do_projection)
    return loss


def vertex_rotation_loss(q_pred_logits, q_pred_res, q_gt, q_gt_bin, q_init,
                         vertices, vertices_mask, image_size, intrinsics):
    """
    Loss function for mulitbin poses. Measure errors based on average l1
    distance between object vertices. Also account for object symmetries.

    Inputs:
      q_pred_logits: predicted per-bin logits (N, M)
      q_pred_res: predicted per-bin quaternion residuals (N, 4) or (N, 6)
      q_gt: ground truth quaternion (N, 4)
      q_gt_bin: ground truth quaternion class label (N, M)
      q_init: quaternion of 1st stage prediction (N, 4)
      vertices: vertices of object model (N, V, 3)
      vertices_mask: takes value 1 wherever the vertex is valid (N, V)
      image_size: tuple image size in (height, width) format
      intrinsics: camera intrinsics matrices (N, 3, 3)

    Outputs:
      loss: scalar multibin loss
    """

    if cfg.USE_6D:
        R_init = quaternion_to_rotation(q_init)
        R_pred_res = compute_rotation_matrix_from_ortho6d(q_pred_res)
        R_pred = R_pred_res @ R_init
        reg_loss = vertex_loss_rotation(
            R_pred, None, q_gt, None, vertices, vertices_mask, intrinsics)
    else:
        q_pred = multiply_quaternion(q_pred_res, q_init)
        reg_loss = vertex_loss_quaternion(
            q_pred, None, q_gt, None, vertices, vertices_mask, intrinsics)

    clf_loss = focal_loss(
        q_gt_bin, q_pred_logits, weights=[10.0, 0.1], reduction='none')
    loss_dict = {
        'quat_clf': torch.mean(clf_loss) * cfg.WEIGHT_QUAT_CLF,
        'quat_reg': torch.mean(reg_loss) * cfg.WEIGHT_QUAT_REG}
    return loss_dict


def quantize_trans_2d(trans_xy, get_res=False):
    """
    Inputs:
      trans_xy: perspective 2d translation in [-0.5, 0.5] (?, 2)

    Outputs:
      trans_prob: flattend translation heatmap in 2D (?, C*C)
    """
    batch_dims = len(trans_xy.shape) - 1
    batch_grid_x, batch_grid_y = make_grid_2d(
        batch_dims, device=trans_xy.device)
    res_x = trans_xy[..., 0, None, None] + 0.5 - batch_grid_x
    res_y = trans_xy[..., 1, None, None] + 0.5 - batch_grid_y
    dist = res_x**2 + res_y**2
    dist = torch.flatten(dist, start_dim=-2)
    trans_prob = torch.softmax(-dist * cfg.CELL_SIZE**2, dim=-1)
    if not get_res:
        return trans_prob
    else:
        return trans_prob, torch.stack([res_x, res_y], dim=-1)


def quantize_persp_trans(t_gt, intrinsics, image_size, beta=4.0):
    """
    Quantize object center into individual cells and associated residuals

    Inputs:
      t_gt: ground truth translation (N, 3)
      intrinsics: camera intrinsics matrices (N, 3, 3)
      image_size: tuple image size in (height, width) format

    Outputs:
      xy_weights: probability of object belonging to each cell (N, C*C)
    """
    t_gt_persp = trans_3d_to_perspective(t_gt, image_size, intrinsics)
    batch_grid_x, batch_grid_y = make_grid_2d(1, device=t_gt.device)
    dist = (t_gt_persp[..., 0, None, None] + 0.5 - batch_grid_x)**2 \
        + (t_gt_persp[..., 1, None, None] + 0.5 - batch_grid_y)**2
    dist = torch.flatten(dist, start_dim=-2)
    xy_weights = torch.softmax(-dist * cfg.CELL_SIZE**2 * beta, dim=-1)
    return xy_weights


def vertex_translation_loss(t_logits, t_res_pred, trans_2d, t_gt,
                            depth_bin_pred, depth_res_pred, depth_id_gt,
                            intrinsics, R_gt, image_size,
                            depth_min, depth_max, n_depth_bin):
    """
    Loss function for 3D translation using smooth l1 distance

    Inputs:
      t_logits: predicted xy translation logits (N, C * C)
      t_res_pred: predicted xy translation residual (N, 2)
      trans_2d: perspective 2D translation (N, 2)
      t_gt: ground truth translation (N, 3)
      depth_bin_pred: predicted probility of depth belonging to each bin (N, B)
      depth_res_pred: predicted residual depth (N, 1)
      depth_id_gt: depth index of rendered image (N,)
      intrinsics: camera intrinsics matrices (N, 3, 3)
      R_gt: ground truth rotation matrices (N, 3, 3)
      image_size: tuple image size in (height, width) format
      (depth_min, depth_max): range of perspective depth
      n_depth_bin: number of bins to quantize depth into

    Outputs:
      loss: scalar translation loss
    """

    xy_weights = quantize_persp_trans(t_gt, intrinsics, image_size)
    trans_clf_loss = multiclass_focal_loss(
        xy_weights, t_logits, reduction='none')

    t_gt_persp = trans_3d_to_perspective(t_gt, image_size, intrinsics)
    t_pred_xy = perspective_to_trans_3d(
        torch.concat([trans_2d + t_res_pred, t_gt_persp[:, -1:]], dim=-1),
        image_size, intrinsics)
    xy_loss = smooth_l1_loss(t_pred_xy - t_gt, beta=1.0)  # (N, C*C)

    depth_bin_gt = quantize_depth(
        t_gt_persp[..., -1], depth_min, depth_max, n_depth_bin)[0]
    depth_clf_loss = multiclass_focal_loss(
        depth_bin_gt, depth_bin_pred, reduction='none')
    depth_bins = make_depth_bins(
        depth_min, depth_max, n_depth_bin, device=depth_bin_gt.device)
    depth_pred = depth_res_pred + depth_bins  # (N, B)
    depth_pred = torch.gather(
        depth_pred, dim=-1, index=depth_id_gt.unsqueeze(-1))  # (N, 1)
    t_pred_z = torch.concat(
        [t_gt_persp[:, :-1], depth_pred], dim=-1)  # (N, 3)
    t_pred_z = perspective_to_trans_3d(t_pred_z, image_size, intrinsics)
    z_loss = smooth_l1_loss(t_pred_z - t_gt, beta=1.0, reduction='none')
    loss_dict = {
        'depth_clf': torch.mean(depth_clf_loss) * cfg.WEIGHT_DEPTH_CLF,
        'txty_clf': torch.mean(trans_clf_loss) * cfg.WEIGHT_TRANS_CLF,
        'txty_reg': torch.mean(xy_loss) * cfg.WEIGHT_TRANS_REG,
        'tz_reg': torch.mean(z_loss) * cfg.WEIGHT_DEPTH_REG}
    return loss_dict


def angle_between_quaternion(q1, q2, is_degree=True):
    """
    Angle (in degree) between batches of quaternions q1 and q2

    Inputs:
      q1, q2: two quaternions to measure relative angle against (?, 4)

    Outputs:
      theta: absolution angular difference in degree (?,)
    """
    q1 = unify_quaternion(q1 / (torch.norm(q1, dim=-1, keepdim=True) + 1e-10))
    q2 = unify_quaternion(q2 / (torch.norm(q2, dim=-1, keepdim=True) + 1e-10))
    phi = 2 * torch.acos(torch.clamp(
        torch.abs(torch.sum(q1 * q2, dim=-1)), max=0.99999))
    return phi * 180. / PI if is_degree else phi


def unify_quaternion(quaternions):
    """
    Make sure each quaternion has leading positive entries

    Inputs:
      quaternions: (?, 4)

    Outpus:
     unified_quaternions: (?, 4)
    """
    quat_signs = torch.sign(quaternions)
    sr, si, sj, sk = quat_signs[..., 0], quat_signs[..., 1], \
        quat_signs[..., 2], quat_signs[..., 3]
    qs = sr
    qs[sr == 0.] = si[sr == 0.]
    mi = torch.logical_and(sr == 0., si == 0.)
    qs[mi] = sj[mi]
    mj = torch.logical_and(mi, sj == 0.)
    qs[mj] = sk[mj]
    return qs.unsqueeze(-1) * quaternions


def quaternion_to_rotation(quaternion, normalize=True):
    """
    Convert unit quaternion to 3x3 rotation matrices

    Inputs:
      quaternion: quaternions (N, 4)
      normalize: whether to normalize quaternion to have unit norm bool

    Outputs:
      rotation: rotation matrix (N, 3, 3)
    """

    if normalize:
        quaternion = quaternion / torch.sqrt(torch.sum(
            quaternion**2, dim=-1, keepdim=True) + 1e-15)
    q0, q1, q2, q3 = quaternion[..., 0], quaternion[..., 1], \
        quaternion[..., 2], quaternion[..., 3]
    s = 0.5 * (q0**2 + q1**2 + q2**2 + q3**2)
    r1 = torch.stack(
        [q0**2 + q1**2 - s, q1*q2 - q0*q3, q0*q2 + q1*q3], dim=-1)
    r2 = torch.stack(
        [q0*q3 + q1*q2, q0**2 + q2**2 - s, q2*q3 - q0*q1], dim=-1)
    r3 = torch.stack(
        [q1*q3 - q0*q2, q0*q1 + q2*q3, q0**2 + q3**2 - s], dim=-1)
    rotation = 2 * torch.stack([r1, r2, r3], dim=-2)
    return rotation


def rotation_to_quaternion(rotation_matrix):
    """Convert 3x3 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 3)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 3)  # Nx3x3
        >>> output = rotation_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < 0

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 & mask_d0_d1
    mask_c1 = mask_d2 & ~mask_d0_d1
    mask_c2 = ~mask_d2 & mask_d0_nd1
    mask_c3 = ~mask_d2 & ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return unify_quaternion(q)


def cross(v1, v2):
    """
    Compute cross product between (batched) 3D vectors v1 and v2

    Inputs:
      v1, v2: two vectors to multiply (?, 3)

    Outputs:
      v: cross product of v1 and v2 (?, 3)
    """
    a1, a2, a3 = v1[..., 0], v1[..., 1], v1[..., 2]
    b1, b2, b3 = v2[..., 0], v2[..., 1], v2[..., 2]
    s1 = torch.unsqueeze(a2 * b3 - a3 * b2, dim=-1)
    s2 = torch.unsqueeze(a3 * b1 - a1 * b3, dim=-1)
    s3 = torch.unsqueeze(a1 * b2 - a2 * b1, dim=-1)
    return torch.cat([s1, s2, s3], dim=-1)


def multiply_quaternion(q1, q2):
    """
    Performs quaternion multiplication

    Inputs:
      q1, q2: two quaternions to multiply (?, 4)

    Outputs:
      r: multiplication result (?, 4)
    """
    q1r, q1i = q1[..., :1], q1[..., 1:]
    q2r, q2i = q2[..., :1], q2[..., 1:]
    real = q1r*q2r - torch.sum(q1i*q2i, dim=-1, keepdim=True)
    imag = q1r*q2i + q2r*q1i + cross(q1i, q2i)
    return torch.cat([real, imag], dim=-1)


def axis_angle_to_rotations(axis_angles):
    """
    Converts Rodrigues vectors to rotation matrices

    Inputs:
      axis_angles: (N, 3)

    Outputs:
      rotations: (N, 3, 3)
    """
    theta = np.linalg.norm(axis_angles, axis=-1, keepdims=True)
    u = axis_angles / (theta + 1e-20)
    u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
    o = np.zeros_like(u1)
    r1 = np.stack([o, -u3, u2], axis=-1)
    r2 = np.stack([u3, o, -u1], axis=-1)
    r3 = np.stack([-u2, u1, o], axis=-1)
    ux = np.stack([r1, r2, r3], axis=1)
    theta = np.expand_dims(theta, axis=-1)
    rotations = np.cos(theta) * np.expand_dims(np.eye(3), axis=0) \
        + (1 - np.cos(theta)) * np.expand_dims(
            u, axis=-1) @ np.expand_dims(u, axis=1) \
        + np.sin(theta) * ux
    return rotations


def get_symmetry_transformations(model_info, max_sym_disc_step):
    """
    Reimplementation of bop_toolkit_lib.misc.get_symmetry_transformations
    in vectorized format to acclelerate data preprocessing
    """
    # Discrete symmetries.
    rotations_disc = [np.eye(3)]
    translations_disc = [np.zeros((3, 1))]
    if 'symmetries_discrete' in model_info:
        for sym in model_info['symmetries_discrete']:
            sym_4x4 = np.reshape(sym, (4, 4))
            R = sym_4x4[:3, :3]
            t = sym_4x4[:3, 3].reshape((3, 1))
            rotations_disc.append(R)
            translations_disc.append(t)
    rotations_disc = np.stack(rotations_disc, axis=0)
    translations_disc = np.stack(translations_disc, axis=0)

    # Discretized continuous symmetries.
    rotations_cont, translations_cont = [], []
    if 'symmetries_continuous' in model_info:
        for sym in model_info['symmetries_continuous']:
            axis = np.array(sym['axis'])
            offset = np.array(sym['offset']).reshape((1, 3, 1))

            # (PI * diam.) / (max_sym_disc_step * diam.) = discrete_steps_count
            discrete_steps_count = int(np.ceil(np.pi / max_sym_disc_step))

            # Discrete step in radians.
            discrete_step = 2.0 * np.pi / discrete_steps_count

            angle = np.expand_dims(np.arange(
                1, discrete_steps_count) * discrete_step, axis=1)
            angle_axis = angle @ np.expand_dims(axis, axis=0)
            batch_rotations = axis_angle_to_rotations(angle_axis)
            rotations_cont.append(batch_rotations)
            translations_cont.append(-batch_rotations @ offset + offset)

    # Combine the discrete and the discretized continuous symmetries.
    if len(rotations_cont):
        rotations_cont = np.concatenate(rotations_cont, axis=0)
        translations_cont = np.concatenate(translations_cont, axis=0)
        rotations = np.expand_dims(
            rotations_cont, axis=0) @ np.expand_dims(rotations_disc, axis=1)
        rotations = np.reshape(rotations, (-1, 3, 3))
        translations = np.expand_dims(
            rotations_cont, axis=0) @ np.expand_dims(translations_disc, axis=1)
        translations = np.reshape(translations, (-1, 3))
    else:
        rotations = rotations_disc
        translations = np.reshape(translations_disc, (-1, 3))

    return rotations, translations


def normalize_depth(depth, intrinsics):
    depth_n = depth / intrinsics[..., 0, 0]
    return depth_n.view(depth.shape)


def unnormalize_depth(depth, intrinsics):
    depth_un = depth * intrinsics[..., 0, 0]
    return depth_un.view(depth.shape)


def image_shift(image, trans_2d, out_size=[cfg.INPUT_SIZE, cfg.INPUT_SIZE]):
    """
    Shift a batch of images `image` according to `trans_2d`

    Inputs:
      image: batch image (N, C, H, W)
      trans_2d: translation in normalized pixel coordinates (N, 2)
      out_size: spatial output image size [Hout, Wout]

    Outputs:
      image_shifted: shifted versions of `image` (N, C, Hout, Wout)
    """
    batch_size = len(image)
    height, width = image.shape[-2], image.shape[-1]
    tx = (out_size[1] - width) / 2 + width * trans_2d[..., 0]
    ty = (out_size[0] - height) / 2 + height * trans_2d[..., 1]
    xx, yy = torch.meshgrid(
        torch.arange(out_size[1], device=image.device),
        torch.arange(out_size[0], device=image.device), indexing='xy')
    xx = torch.repeat_interleave(xx[None], batch_size, dim=0)
    yy = torch.repeat_interleave(yy[None], batch_size, dim=0)
    gx = 2 * (xx - tx.view(-1, 1, 1)) / (width - 1) - 1
    gy = 2 * (yy - ty.view(-1, 1, 1)) / (height - 1) - 1
    grid = torch.stack([gx, gy], dim=-1)
    image_shifted = F.grid_sample(image, grid, align_corners=False)
    return image_shifted


def make_roi(bbox_loc, image_size):
    """
    Create a binary image with dilated bounding box region set to 1
    """
    roi = torch.zeros((image_size, image_size)).to(bbox_loc)
    if torch.all(bbox_loc < 0.):
        return roi
    xmin = bbox_loc[0].item()
    ymin = bbox_loc[1].item()
    xmax = bbox_loc[2].item()
    ymax = bbox_loc[3].item()
    delta = (cfg.ZOOM_SCALE_RATIO + cfg.ZOOM_SHIFT_RATIO) * 0.25
    dilate_width = (xmax - xmin) * delta
    dilate_height = (ymax - ymin) * delta
    xmin = xmin - dilate_width
    xmax = xmax + dilate_width
    ymin = ymin - dilate_height
    ymax = ymax + dilate_height
    xmin = max(round(xmin*image_size), 0)
    ymin = max(round(ymin*image_size), 0)
    xmax = min(round(xmax*image_size), image_size)
    ymax = min(round(ymax*image_size), image_size)
    roi[ymin:ymax, xmin:xmax] = 1.0
    return roi


def image_to_bbox_map(image_tensor):
    """
    Create bounding box maps based on visible image regions
    """
    image_mask = (image_tensor.sum(dim=1) > 0).float()
    batch_size = len(image_mask)
    batch_size = len(image_mask)
    batch_bbox = torch.zeros((batch_size, 4), device=image_mask.device,
                             dtype=torch.float)
    for index, mask in enumerate(image_mask):
        if torch.any(mask):
            y, x = torch.where(mask != 0)
            batch_bbox[index, 0] = torch.min(x) / image_mask.shape[-1]
            batch_bbox[index, 1] = torch.min(y) / image_mask.shape[-2]
            batch_bbox[index, 2] = torch.max(x) / image_mask.shape[-1]
            batch_bbox[index, 3] = torch.max(y) / image_mask.shape[-2]
        else:
            batch_bbox[index] = -torch.ones(4)
    bbox_map = torch.stack([
        make_roi(bbox, image_mask.shape[-1]) for bbox in batch_bbox], dim=0)
    return bbox_map


class BatchModelRenderer:
    def __init__(self, image_width, image_height, dataset='tless',
                 surf_color=[0.6, 0.6, 0.6]):
        # Parameters
        self.width = image_width
        self.height = image_height
        self.objects = defaultdict(list)
        self.dataset = dataset
        dataset_cfg = bop_cfg.DATASET_CONFIG[dataset]
        model_folder = dataset_cfg['model_folders']['train_pbr']

        # Load every objects and cache them into memory
        for obj_id in dataset_cfg['id2cls']:
            model_path = '{}/{}/{}/obj_{:06d}.ply'.format(
                cfg.DATASET_ROOT, dataset, model_folder, obj_id)
            verts, faces = pytorch3d.io.load_ply(model_path)
            self.objects['verts'].append(verts)
            self.objects['faces'].append(faces)
        self.surf_color = surf_color

    def render_object(self, obj_cls, rotation, translation, intrinsics,
                      height=None, width=None):
        image_size = (self.height, self.width)
        if height is not None:
            image_size[0] = height
        if width is not None:
            image_size[1] = width

        verts_list = [self.objects['verts'][i] for i in obj_cls]
        faces_list = [self.objects['faces'][i] for i in obj_cls]
        colors_list = [torch.tensor(
            [self.surf_color], device=rotation.device).expand(
                len(self.objects['verts'][i]), 3) for i in obj_cls]
        textures = pytorch3d.renderer.TexturesVertex(
            verts_features=colors_list)
        meshes = pytorch3d.structures.Meshes(
            verts=verts_list, faces=faces_list,
            textures=textures).to(rotation.device)
        batch_size = len(obj_cls)
        materials = pytorch3d.renderer.Materials(
            device=rotation.device,
            diffuse_color=[self.surf_color for _ in range(batch_size)],
            specular_color=[self.surf_color for _ in range(batch_size)],
            shininess=10.0)

        fx, fy = intrinsics[..., 0, 0], intrinsics[..., 1, 1]
        cx, cy = intrinsics[..., 0, 2], intrinsics[..., 1, 2]
        focal_length = torch.stack([-fx, -fy], dim=-1).to(torch.float32)
        principal_point = torch.stack([cx, cy], dim=-1).to(torch.float32)
        rotation = rotation.to(torch.float32)
        translation = translation.to(torch.float32)
        cam_pose = torch.transpose(rotation, 1, 2)
        lights_direction = torch.squeeze(
            -translation.unsqueeze(-2) @ rotation, dim=-2)
        translation = translation.to(torch.float32)
        cameras = pytorch3d.renderer.cameras.PerspectiveCameras(
            focal_length, principal_point, cam_pose, translation,
            device=rotation.device, in_ndc=False, image_size=(image_size,))
        lights = pytorch3d.renderer.DirectionalLights(
            ambient_color=[[0.8, 0.8, 0.8]], diffuse_color=[[0.5, 0.5, 0.5]],
            specular_color=[[0.1, 0.1, 0.1]],
            direction=lights_direction, device=rotation.device)

        raster_settings = pytorch3d.renderer.RasterizationSettings(
            image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
        rasterizer = pytorch3d.renderer.MeshRasterizer(
            cameras=cameras, raster_settings=raster_settings)
        shader = pytorch3d.renderer.HardPhongShader(
            device=rotation.device, cameras=cameras,
            lights=lights, materials=materials,
            blend_params=pytorch3d.renderer.BlendParams(
                background_color=[0., 0., 0.]))
        renderer = pytorch3d.renderer.MeshRendererWithFragments(
            rasterizer, shader)

        images, fragments = renderer(meshes)
        rgb = images[..., :3]
        depth = fragments.zbuf
        return {'rgb': rgb, 'depth': depth}


class SyntheticRenderer(BatchModelRenderer):
    def __init__(self, *args, **kwargs):
        super(SyntheticRenderer, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def __call__(self, obj_cls, rotation, translation, intrinsics):
        rend_dict = self.render_object(
            obj_cls, rotation, translation, intrinsics)
        rgb = rend_dict['rgb'].permute(0, 3, 1, 2)
        mask = (rend_dict['depth'].permute(0, 3, 1, 2) > 0).float()
        bbox_map = image_to_bbox_map(
            rend_dict['depth'].permute(0, 3, 1, 2)).unsqueeze(1)
        return rgb, mask, bbox_map
