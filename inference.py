from lib import data_utils as misc
import config as bop_cfg
from bop_toolkit_lib import dataset_params
import models
import utils
import os
import sys
import time
import json
import torch
import argparse
import numpy as np
from bop_toolkit_lib import inout
import mmcv
import pycocotools.mask as cocomask
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)


def rle2mask(rle, height, width):
    if "counts" in rle and isinstance(rle["counts"], list):
        # if compact RLE, ignore this conversion
        # Magic RLE format handling painfully discovered by looking at the
        # COCO API showAnns function.
        rle = cocomask.frPyObjects(rle, height, width)
    mask = cocomask.decode(rle)
    return mask


def inference_func(net, device, obj_cls, roi_rgb, bbox_loc, roi_camK, fov, Rz):
    batch_image = torch.from_numpy(roi_rgb).to(device)
    batch_obj_cls = torch.from_numpy(obj_cls).to(device)
    im_height, im_width = batch_image.shape[2:]
    batch_bbox_map = torch.stack([utils.make_roi(
        torch.as_tensor(x, dtype=torch.float32), im_height).to(device)
        for x in bbox_loc], dim=0).unsqueeze(1)
    batch_fov = torch.as_tensor(
        fov, dtype=torch.float32).to(device)
    batch_input = torch.cat([batch_image, batch_bbox_map], dim=1)
    intrinsics = torch.from_numpy(roi_camK).to(device)

    with torch.no_grad():
        predictions = net(batch_input,
                          {'obj_cls': batch_obj_cls,
                           'fov': batch_fov,
                           'intrinsics': intrinsics})
        R_conf = predictions['quat_bin']
        R_index = torch.argmax(torch.amax(R_conf, dim=1))
        t_conf = predictions['depth_bin']
        t_index = torch.argmax(torch.amax(t_conf, dim=1))

        Rz_inv = np.transpose(Rz, [0, 2, 1])
        R_pred = Rz_inv @ predictions['roi_obj_R'].cpu().numpy()
        R_pred = R_pred[R_index]
        t_pred = utils.perspective_to_trans_3d(
            predictions['translation'],
            (bop_cfg.INPUT_SIZE, bop_cfg.INPUT_SIZE), intrinsics)
        t_pred = Rz_inv @ np.expand_dims(t_pred.cpu().numpy(), axis=-1)
        t_pred = t_pred[t_index, :, 0]

    return R_pred, t_pred


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--checkpoint_name', type=str, required=True)
    parser.add_argument('--output_suffix', type=str, default='')
    parser.add_argument('--is_real', action='store_true')
    parser.add_argument('--model_name', type=str, default='')
    args = parser.parse_args()

    p = {
        'dataset': args.dataset,
        'bop_root': bop_cfg.DATASET_ROOT,
        'eval_root': bop_cfg.EVAL_ROOT,
        'output_suffix_name': '{}_{}'.format(
            args.checkpoint_name, args.output_suffix),
        'checkpoint': './{}/{}.pth'.format(args.checkpoint_name,
                                           args.model_name)
    }
    dataset_id2cls = bop_cfg.DATASET_CONFIG[p['dataset']]['id2cls']

    model_type = 'eval'
    dp_model = dataset_params.get_model_params(
        p['bop_root'], p['dataset'], model_type)
    dp_data = dataset_params.get_split_params(
        p['bop_root'], p['dataset'], 'test')
    with open(dp_model['models_info_path'], 'r') as fp:
        model_info = json.load(fp)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    depth_min = bop_cfg.DATASET_CONFIG[p['dataset']]['Tz_near']
    depth_max = bop_cfg.DATASET_CONFIG[p['dataset']]['Tz_far']
    n_decoders = len(dataset_id2cls)
    net = models.MRCNet(
        p['dataset'], depth_min=depth_min,
        depth_max=depth_max, n_decoders=n_decoders,
        n_depth_bin=bop_cfg.Tz_BINS_NUM).to(device)
    print('building model for {}'.format(p['dataset']))
    checkpoint = torch.load(p['checkpoint'], map_location=device)
    print('loading pre-trained model from {}'.format(p['checkpoint']))
    net.load_state_dict(checkpoint['network'])
    net.eval()

    est_pose_file = '{}/mrcnet_{}-test_{}.csv'.format(
        p['eval_root'], p['dataset'], p['output_suffix_name'])
    if args.is_real:
        det_file = os.path.join(
            root_dir, 'bop22_default_detections_and_segmentations',
            'cosypose_maskrcnn_synt+real',
            'challenge2022-642947_{}-test.json'.format(p['dataset']))
    else:
        det_file = os.path.join(
            root_dir, 'bop22_default_detections_and_segmentations',
            'cosypose_maskrcnn_pbr',
            'challenge2022-524061_{}-test.json'.format(p['dataset']))

    print('Loading cosypose detection: ', det_file)
    with open(det_file, 'r') as f:
        object_pred_dets = json.load(f)

    image_detect_dict = dict()
    for det_entry in object_pred_dets:
        view_id = det_entry['image_id']
        scene_id = det_entry['scene_id']
        obj_id = det_entry['category_id']
        obj_conf = det_entry['score']
        x1, y1, w, h = det_entry['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x1 + w), int(y1 + h)
        mask = rle2mask(det_entry['segmentation'], det_entry['segmentation']
                        ['size'][0], det_entry['segmentation']['size'][1])
        mask = mask*255  # convert to 0-255 range
        scene_view_str = '{:06d}/{:06d}'.format(scene_id, view_id)
        if scene_view_str not in image_detect_dict:
            image_detect_dict[scene_view_str] = {
                'ids': list(),
                'bboxes': list(),
                'scores': list(),
                'labels': list(),
                'masks': list()
            }
        image_detect_dict[scene_view_str]['ids'].append(obj_id)
        image_detect_dict[scene_view_str]['scores'].append(obj_conf)
        image_detect_dict[scene_view_str]['bboxes'].append(
            np.array([x1, y1, x2, y2]))
        image_detect_dict[scene_view_str]['labels'].append(obj_id)
        image_detect_dict[scene_view_str]['masks'].append(mask)

    num_test_images = len(image_detect_dict)
    num_test_instances = len(object_pred_dets)

    print('Evaluation on BOP19 challenge: {} instances in {} images'.format(
        num_test_images, num_test_instances))  # 10079
    print(est_pose_file)

    eval_steps = 0
    obj_runtime = list()
    view_runtime = list()
    bop19_pose_est_results = list()
    for ii, (scene_view_str, det_data) in enumerate(
            sorted(image_detect_dict.items())):
        eval_steps += 1
        scene_id_str, view_id_str = scene_view_str.split('/')
        view_id = int(view_id_str)
        scene_id = int(scene_id_str)

        scene_dir = '{}/{:06d}'.format(dp_data['split_path'], scene_id)
        scene_camK = mmcv.load(os.path.join(scene_dir, 'scene_camera.json'))
        view_rgb_file = os.path.join(
            scene_dir, 'rgb', '{:06d}.png'.format(view_id))
        if not os.path.exists(view_rgb_file):
            view_rgb_file = os.path.join(
                # gray images in ITODD
                scene_dir, 'gray', '{:06d}.tif'.format(view_id))

        view_cam_K = np.asarray(
            scene_camK[str(view_id)]['cam_K'],
            dtype=np.float32).reshape((3, 3))
        view_image = torch.as_tensor(mmcv.imread(
            view_rgb_file, 'color', 'BGR'), dtype=torch.float32)
        img_H, img_W = view_image.shape[:2]

        det_objIDs = det_data['ids']        # list: N
        det_bboxes = det_data['bboxes']     # array: Nx4
        det_scores = det_data['scores']     # array: N
        det_labels = det_data['labels']     # list: N
        det_masks = det_data['masks']

        inst_time = list()
        view_objs_ts = list()
        view_objs_Rs = list()
        view_objs_IDs = list()
        view_objs_scores = list()

        for inst_ix, inst_id in enumerate(det_objIDs):
            if inst_id not in dataset_id2cls:
                continue
            inst_timer = time.time()
            inst_score = det_scores[inst_ix]
            mask_visib = None
            inst_cls = inst_id
            diameter = model_info[str(inst_id)]['diameter']

            x1, y1, x2, y2 = det_bboxes[inst_ix]
            cx = min((x1 + x2) / 2.0, img_W)
            cy = min((y1 + y2) / 2.0, img_H)
            bw = int(max(0, min(x2 - x1, img_W)))
            bh = int(max(0, min(y2 - y1, img_H)))
            bx = int(max(0, cx - bw // 2))
            by = int(max(0, cy - bh // 2))

            # box square size max(w, h) * pad
            box_scale = max(bw, bh) * bop_cfg.ZOOM_PAD_SCALE
            hr = (y2 - y1) / box_scale
            wr = (x2 - x1) / box_scale
            zooming_factor = bop_cfg.INPUT_IMG_SIZE / box_scale
            fx, fy, ox, oy = view_cam_K[0, 0], view_cam_K[1, 1], \
                view_cam_K[0, 2], view_cam_K[1, 2]

            b_Rz, b_obj_cls, b_roi_rgb, b_bbox_loc, b_roi_camK, b_fov \
                = [], [], [], [], [], []
            for rot_index in range(4):
                rot_rad = rot_index * np.pi / 2
                bbox_center = torch.as_tensor([cx, cy], dtype=torch.float32)
                bbox_loc = np.array(
                    [0.5 - wr/2, 0.5 - hr/2, 0.5 + wr/2, 0.5 + hr/2])

                Rz = np.array([[np.cos(rot_rad), -np.sin(rot_rad), 0.0],
                               [np.sin(rot_rad), np.cos(rot_rad), 0.0],
                               [0.0, 0.0, 1.0]], dtype=np.float32)
                T_rot = view_cam_K @ Rz @ np.linalg.inv(view_cam_K)
                center_hom = T_rot @ np.array(
                    [*bbox_center, 1.0], dtype=np.float32)
                bbox_center = center_hom[:2]
                fov = np.array([(bbox_center[0] - cx) / fx,
                               (bbox_center[1] - cy) / fy, box_scale / fx])
                b_Rz.append(Rz)
                b_fov.append(fov)

                Tz = np.array([[1.0, 0.0, -0.5],
                               [0.0, 1.0, -0.5],
                               [0.0, 0.0, 1.0]], dtype=np.float32)
                T_loc = np.linalg.inv(Tz) @ Rz @ Tz
                bbox_loc = utils.transform_bounding_box(bbox_loc, T_loc)
                b_bbox_loc.append(bbox_loc)

                # transformation from RGB image X to object-centric crop B
                T_img2roi = misc.transform_to_local_ROIcrop(
                    bbox_center=bbox_center, bbox_scale=box_scale,
                    zoom_scale=bop_cfg.INPUT_IMG_SIZE)
                roi_camK = T_img2roi @ view_cam_K
                b_roi_camK.append(roi_camK)

                roi_rgb = misc.crop_resize_by_warp_affine(
                    view_image.numpy(), np.array([cx, cy]), box_scale,
                    bop_cfg.INPUT_IMG_SIZE, view_cam_K, rot_rad,
                    interpolation='bilinear')
                roi_rgb = roi_rgb / 255.0  # 1x3xHxW
                b_roi_rgb.append(roi_rgb)
                b_obj_cls.append(dataset_id2cls[inst_id])

            b_obj_cls = np.stack(b_obj_cls, axis=0)
            b_roi_rgb = np.stack(b_roi_rgb, axis=0)
            b_bbox_loc = np.stack(b_bbox_loc, axis=0)
            b_roi_camK = np.stack(b_roi_camK, axis=0)
            b_fov = np.stack(b_fov, axis=0)
            b_Rz = np.stack(b_Rz, axis=0)

            est_R, est_t = inference_func(
                net, device, b_obj_cls, b_roi_rgb, b_bbox_loc,
                b_roi_camK, b_fov, b_Rz)

            view_objs_ts.append(est_t)
            view_objs_Rs.append(est_R)
            view_objs_IDs.append(inst_id)
            view_objs_scores.append(inst_score)
            inst_time.append(time.time() - inst_timer)

        view_cost = np.sum(inst_time)
        inst_cost = np.mean(inst_time)
        view_runtime.append(view_cost)
        obj_runtime.append(inst_cost)

        for eix, obj_id in enumerate(view_objs_IDs):
            est_t = view_objs_ts[eix]
            est_R = view_objs_Rs[eix]
            det_conf = view_objs_scores[eix]
            bop19_pose_est_results.append({'time': view_cost,
                                           'scene_id': int(scene_id),
                                           'im_id': int(view_id),
                                           'obj_id': int(obj_id),
                                           'score': det_conf,
                                           'R': est_R,
                                           't': est_t})

        if eval_steps % 10 == 0:
            time_stamp = time.strftime('%m-%d_%H:%M:%S', time.localtime())
            print('[{}/{}], img: {:.1f} ms, inst:{:.1f} ms, {}'.format(
                eval_steps, num_test_images,
                np.mean(view_runtime) * 1000,
                np.mean(obj_runtime) * 1000,
                time_stamp))

    inout.save_bop_results(est_pose_file, bop19_pose_est_results)
    print('Results saved to {}.'.format(est_pose_file))
