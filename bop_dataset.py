import os
import sys
import cv2
import copy
import json
import mmcv
import torch
import random
import hashlib
import numpy as np
import pytorch3d.structures
from tqdm import tqdm
import os.path as osp
import torch.nn.functional as F
import trimesh
import utils
import config as cfg

import logging
logger = logging.getLogger(__name__)

from pytorch3d.transforms import euler_angles_to_matrix
from typing import Dict, List

CUR_FILE_DIR = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(CUR_FILE_DIR, '..'))
sys.path.append(PROJ_ROOT)
cv2.setNumThreads(0)

from lib import data_utils as misc

from imgaug.augmenters.arithmetic import multiply_elementwise
from imgaug.augmenters import (Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, Noop,
                               Lambda, AssertLambda, AssertShape, Scale, CropAndPad, Pad, Crop, Fliplr,
                               Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, Grayscale,
                               GaussianBlur, AverageBlur, MedianBlur, Convolve, Sharpen, Emboss, EdgeDetect,
                               DirectedEdgeDetect, Add, AddElementwise, AdditiveGaussianNoise, Multiply,
                               MultiplyElementwise, Dropout, CoarseDropout, Invert, ContrastNormalization,
                               Affine, PiecewiseAffine, ElasticTransformation, pillike, LinearContrast)


class DropoutWithMask(CoarseDropout):
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images
        segmentation_maps = batch.segmentation_maps
        nb_images = len(images)
        rss = random_state.duplicate(1+nb_images)
        per_channel_samples = self.per_channel.draw_samples(
            (nb_images,), random_state=rss[0])

        gen = enumerate(zip(images, per_channel_samples, rss[1:], segmentation_maps))
        for i, (image, per_channel_samples_i, rs, seg_map) in gen:
            height, width, nb_channels = image.shape
            sample_shape = (height,
                            width,
                            nb_channels if per_channel_samples_i > 0.5 else 1)
            mul = self.mul.draw_samples(sample_shape, random_state=rs)
            mul = mul.astype(bool, copy=False)

            batch.images[i] = multiply_elementwise(image, mul)
            batch.segmentation_maps[i].arr = multiply_elementwise(
                seg_map.arr.astype(np.uint8), mul)

        return batch


def collate_fn(batch: List[Dict]):  # pragma: no cover
    """
    Take a list of objects in the form of dictionaries and merge them
    into a single dictionary. This function can be used with a Dataset
    object to create a torch.utils.data.Dataloader which directly
    returns Meshes objects.

    Modified from pytorch3d.datasets.collate_batched_meshes

    Args:
        batch: List of dictionaries containing information about objects
            in the dataset.

    Returns:
        collated_dict: Dictionary of collated lists. If batch contains
        vertices, a collated vertices batch (padded with nan) is returned.
    """
    if batch is None or len(batch) == 0:
        return None
    collated_dict = {}
    for k in batch[0].keys():
        collated_dict[k] = [d[k] for d in batch]

    for k in ["vertices", "normals", "quaternion_symmetries",
              "translation_symmetries"]:
        if k not in collated_dict:
            continue

        elem_padded = pytorch3d.structures.list_to_padded(
            collated_dict[k], pad_value=np.nan)
        mask = torch.logical_not(torch.isnan(
            elem_padded[..., 0]))
        elem_padded[torch.isnan(elem_padded)] = 0.0
        collated_dict[k] = elem_padded

        if k == "vertices":
            collated_dict["vertices_mask"] = mask
        elif k == "quaternion_symmetries":
            collated_dict["symmetries_mask"] = mask

    for k, v in collated_dict.items():
        if k not in ["vertices", "normals", "vertices_mask",
                     "quaternion_symmetries", "translation_symmetries",
                     "symmetries_mask"]:
            collated_dict[k] = torch.utils.data.default_collate(v)

    return collated_dict


class BOP_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, split, rank=0):
        self.dataset_name = dataset_name
        self.rgb_size = cfg.INPUT_IMG_SIZE
        self.mask_size = cfg.OUTPUT_MASK_SIZE
        self.data_dir = os.path.join(cfg.DATASET_ROOT, dataset_name)

        self.width = cfg.DATASET_CONFIG[dataset_name]['width']
        self.height = cfg.DATASET_CONFIG[dataset_name]['height']
        self.split = split

        self.depth_min = cfg.DATASET_CONFIG[dataset_name]['Tz_near']
        self.depth_max = cfg.DATASET_CONFIG[dataset_name]['Tz_far']
        self.num_objects = cfg.DATASET_CONFIG[dataset_name]['num_class']

        if split == 'train':
            self.name_set = cfg.DATASET_CONFIG[dataset_name]['train_set']
        elif split == 'finetune':
            self.name_set = cfg.DATASET_CONFIG[dataset_name]['finetune_set']
        else:
            self.name_set = cfg.DATASET_CONFIG[dataset_name]['test_set']

        assert(isinstance(self.name_set, list)), 'train_set(s) must be a list' # ['train_pbr', 'train_real', ...]
        self.dataset_id2cls = cfg.DATASET_CONFIG[dataset_name]['id2cls']
        self.num_classes = len(self.dataset_id2cls)

        self.img_format = 'BGR'
        self.mask_morph = True
        self.filter_invalid = True
        self.mask_morph_kernel_size = 3
        self.color_augmentor = Sequential([
            Sometimes(0.5, AdditiveGaussianNoise(scale=(0, 0.01*255), per_channel=0.5)),
            Sometimes(0.5, DropoutWithMask(p=0.2, size_percent=0.05)),
            Sometimes(0.5, GaussianBlur(1.2*np.random.rand())),
            Sometimes(0.5, Add((-25, 25), per_channel=0.3)),
            Sometimes(0.3, Invert(0.2, min_value=0, max_value=255, per_channel=True)),
            Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),
            Sometimes(0.5, Multiply((0.6, 1.4))),
            Sometimes(0.5, LinearContrast((0.5, 2.2), per_channel=0.3))
        ], random_order=False)  # aae

        self.DZI_PAD_SCALE = cfg.ZOOM_PAD_SCALE
        self.DZI_SCALE_RATIO = cfg.ZOOM_SCALE_RATIO  # wh scale
        self.DZI_SHIFT_RATIO = cfg.ZOOM_SHIFT_RATIO  # center shift
        self.Rz_rotation_aug = cfg.RZ_ROTATION_AUG
        self.CHANGE_BG_PROB = cfg.CHANGE_BG_PROB
        self.COLOR_AUG_PROB = cfg.COLOR_AUG_PROB

        self.TRUNCATE_FG = False
        self.BG_KEEP_ASPECT_RATIO = True
        self.NUM_BG_IMGS = 10000
        self.BG_TYPE = "VOC_table"      # VOC_table | coco | VOC | SUN2012
        self.BG_ROOT = cfg.VOC_BG_ROOT  # "datasets/coco/train2017/"

        self.use_cache = cfg.USE_CACHE
        self.cache_dir = os.path.join(CUR_FILE_DIR, ".cache")  # .cache

        hashed_file_name = hashlib.md5(("_".join(self.name_set)
            + "dataset_dicts_{}_{}_{}".format(self.dataset_name, self.data_dir, __name__)
        ).encode("utf-8")).hexdigest()
        cache_path = os.path.join(self.cache_dir,
            "dataset_dicts_{}_{}_{}.pkl".format(self.dataset_name, "_".join(self.name_set), hashed_file_name))
        symmetries_cache_path = os.path.join(self.cache_dir,
            "symm_dataset_dicts_{}_{}_{}.pkl".format(self.dataset_name, "_".join(self.name_set), hashed_file_name))
        cad_cache_path = os.path.join(self.cache_dir,
            "cad_dataset_dicts_{}_{}_{}.pkl".format(self.dataset_name, "_".join(self.name_set), hashed_file_name))

        self.model_folders = cfg.DATASET_CONFIG[dataset_name]['model_folders']

        self.dataset_dicts = list()
        if self.use_cache and os.path.exists(cache_path) and os.path.exists(symmetries_cache_path) and os.path.exists(cad_cache_path):
            # print("load cached dataset dicts from {}".format(cache_path))
            self.dataset_dicts = mmcv.load(cache_path)
            # print('done')
            self.symmetries_dict = mmcv.load(symmetries_cache_path)
            self.cad_model_dict = mmcv.load(cad_cache_path)
        else:
            self.symmetries_dict = dict()
            self.cad_model_dict = dict()
            for img_type in self.name_set:
                image_counter = 0
                instance_counter = 0
                train_dir = os.path.join(self.data_dir, img_type)
                logger.info("preparing data from {}".format(img_type))
                ## load CAD model related information ##
                model_folder = os.path.join(self.data_dir, self.model_folders[img_type])
                self.load_model_data(model_folder, img_type)

                ## process scene and images ############
                for scene in sorted(os.listdir(train_dir)):  # scene
                    if not scene.startswith('00'):  # BOP images start with '0000xx'
                        return
                    scene_id = int(scene)
                    scene_dir = os.path.join(train_dir, scene)
                    scene_cam_dict = mmcv.load(os.path.join(scene_dir, "scene_camera.json"))      # gt_intrinsic
                    scene_gt_pose_dict = mmcv.load(os.path.join(scene_dir, "scene_gt.json"))      # gt_poses
                    scene_gt_bbox_dict = mmcv.load(os.path.join(scene_dir, "scene_gt_info.json"))  # gt_bboxes
                    for img_id_str in tqdm(scene_gt_pose_dict, postfix=f"{scene_id}"):  # image
                        img_id_int = int(img_id_str)
                        color_type = "gray" if dataset_name == 'itodd' else "rgb"
                        rgb_path = os.path.join(scene_dir, "{}/{:06d}.jpg").format(color_type, img_id_int)
                        if not os.path.exists(rgb_path):
                            rgb_path = os.path.join(scene_dir, "{}/{:06d}.png").format(color_type, img_id_int)
                        if not os.path.exists(rgb_path):
                            rgb_path = os.path.join(scene_dir, "{}/{:06d}.tif").format(color_type, img_id_int)
                        assert os.path.exists(rgb_path), rgb_path
                        cam_K = np.array(scene_cam_dict[img_id_str]["cam_K"], dtype=np.float32).reshape(3, 3)

                        record = {
                            "dataset_name": self.dataset_name,
                            "scene_id": scene_id,
                            "image_id": img_id_int,
                            "img_type": img_type,
                            "height": self.height,
                            "width": self.width,
                        }
                        view_insts = []
                        view_inst_count = dict() # count the object number per instance in a single image
                        for anno_idx, anno_dict in enumerate(scene_gt_pose_dict[img_id_str]):
                            obj_id = anno_dict["obj_id"]
                            if obj_id not in self.dataset_id2cls: # ignore the non-target objects 
                                continue
                            R = np.array(anno_dict["cam_R_m2c"], dtype="float32").reshape(3, 3)
                            t = np.array(anno_dict["cam_t_m2c"], dtype="float32")

                            quat_path = os.path.join(scene_dir, "quat_label/{:06d}_{:06d}.npy").format(img_id_int, anno_idx)

                            bbox_visib = scene_gt_bbox_dict[img_id_str][anno_idx]["bbox_visib"]
                            x1, y1, w, h = bbox_visib
                            if self.filter_invalid:
                                if h <= 10 or w <= 10:
                                    continue
                            ### Load precompute quaternion bin id and residual #########
                            model_path = os.path.join(
                                model_folder, 'obj_{:06d}.ply'.format(int(obj_id)))

                            mask_visib_file = os.path.join(scene_dir, "mask_visib/{:06d}_{:06d}.png".format(img_id_int, anno_idx))
                            assert os.path.exists(mask_visib_file), mask_visib_file
                            visib_fract = scene_gt_bbox_dict[img_id_str][anno_idx]["visib_fract"]
                            if visib_fract < 0.10:  # filter out too small or nearly invisible instances
                                continue

                            if cfg.CACHE_MASK:
                                mask_single = mmcv.imread(mask_visib_file, "unchanged").astype(bool).astype(np.uint8)
                                if self.mask_morph:
                                    kernel = np.ones((self.mask_morph_kernel_size, self.mask_morph_kernel_size))
                                    mask_single = cv2.morphologyEx(mask_single.astype(np.uint8), cv2.MORPH_CLOSE, kernel)  # remove holes
                                    mask_single = cv2.morphologyEx(mask_single, cv2.MORPH_OPEN, kernel)  # remove outliers
                                mask_single = misc.binary_mask_to_rle(mask_single, compressed=True)

                            else:
                                mask_single = mask_visib_file

                            if obj_id not in view_inst_count:
                                view_inst_count[obj_id] = 0
                            view_inst_count[obj_id] += 1  # accumulate the object number per instance in a single image

                            # Object instance level information dict
                            inst = {
                                'sub_dataset_folder': img_type,
                                'image_file': rgb_path,
                                'mask_file': mask_single,
                                'model_file': model_path,
                                'bbox': bbox_visib,
                                'quat_file': quat_path,
                                'rotation': R,
                                'translation': t,
                                'intrinsics': cam_K,
                                'scene_id': scene_id,
                                'im_id': img_id_int,
                                'obj_id': int(obj_id),
                            }

                            view_insts.append(inst)
                        if len(view_insts) == 0:  # filter im without anno
                            continue
                        record["annotations"] = view_insts
                        record['obj_inst_count'] = view_inst_count
                        self.dataset_dicts.append(record)

                        image_counter += 1
                        instance_counter += len(view_insts)

                    print(img_type, ', images: ', image_counter, ', instances: ', instance_counter)

                mmcv.dump(self.dataset_dicts, cache_path, protocol=5)
                mmcv.dump(self.symmetries_dict, symmetries_cache_path, protocol=5)
                mmcv.dump(self.cad_model_dict, cad_cache_path, protocol=5)
                logger.info("Dumped dataset_dicts to {}".format(cache_path))
                logger.info("Dumped symm_dicts to {}".format(symmetries_cache_path))
                logger.info("Dumped cad_model_dicts to {}".format(cad_cache_path))

        self.dataset_dicts = misc.flat_dataset_dicts(self.dataset_dicts) # flatten the image-level dict to instance-level dict

    def load_model_data(self, model_folder, sub_dataset_folder):
        self.symmetries_dict[sub_dataset_folder] = dict()
        with open(os.path.join(model_folder, 'models_info.json'), 'r') as fp:
            model_info = json.load(fp)
        for obj_id, info in model_info.items():
            rotations_sym, translations_sym = utils.get_symmetry_transformations(info, 0.01)
            model_path = os.path.join(model_folder, 'obj_{:06d}.ply'.format(int(obj_id)))
            mesh = trimesh.load(model_path)
            vertices = torch.from_numpy(
                mesh.vertices.copy()).to(torch.float32)
            faces = torch.from_numpy(
                mesh.faces.copy()).to(torch.float32)
            normals = torch.from_numpy(
                mesh.vertex_normals.copy()).to(torch.float32)
            alpha = torch.mean(torch.sum(vertices**2, dim=1))
            mu = torch.mean(vertices, dim=0)
            sigma = vertices.T @ vertices / len(vertices)
            diameter = np.array(info['diameter'], dtype=np.float32)
            self.cad_model_dict[model_path] = {
                'vertices': vertices,
                'faces': faces,
                'normals': normals,
                'diameter': diameter,
                'alpha': alpha,
                'mean': mu,
                'correlation': sigma}
            self.symmetries_dict[sub_dataset_folder][obj_id] = {
                'rotation': rotations_sym,
                'translation': translations_sym}

    def get_info(self):
        return {
            'depth_min': self.depth_min,
            'depth_max': self.depth_max,
            'num_objects': self.num_objects}

    def __len__(self):
        return len(self.dataset_dicts)

    def _rand_another(self, idx):
        pool = [i for i in range(self.__len__()) if i != idx]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        data_dict = self.dataset_dicts[idx]
        batch = self.read_data(data_dict)
        return batch

    def read_data(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        inst_infos = dataset_dict.pop("inst_infos")
        obj_id = inst_infos['obj_id']
        scene_id = inst_infos['scene_id']
        image_id = inst_infos['im_id']

        image_file = inst_infos["image_file"]
        img_type = dataset_dict['img_type']
        model_folder = os.path.join(self.data_dir, self.model_folders[img_type])
        model = self.cad_model_dict[inst_infos['model_file']]

        image = mmcv.imread(image_file, 'color', self.img_format)
        image = image.astype(np.float32)
        im_H, im_W = image.shape[:2]
        if cfg.CACHE_MASK:
            mask = misc.cocosegm2mask(inst_infos["mask_file"], im_H, im_W)
        else:
            mask = mmcv.imread(inst_infos["mask_file"], "unchanged").astype(bool).astype(np.uint8)
        ### RGB augmentation ###
        if (self.split == 'train' or self.split == 'finetune') and np.random.rand() < self.COLOR_AUG_PROB:
            image, mask = self.color_augmentor.augment(
                image=image, segmentation_maps=mask[None, :, :, None])
            mask = mask[0, :, :, 0]

        obj_R = inst_infos['rotation'].astype("float32").reshape(3, 3)
        obj_t = inst_infos['translation'].astype("float32").reshape(3,)
        cam_K = inst_infos['intrinsics'].astype("float32")

        bx, by, bw, bh = inst_infos["bbox"]
        bbox_xyxy = np.array([bx, by, bx+bw, by+bh])

        if self.split == 'train' or self.split == 'finetune':
            bbox_center, bbox_scale, bbox_loc = misc.aug_bbox_DZI(
                bbox_xyxy, im_H, im_W,
                scale_ratio=self.DZI_SCALE_RATIO,
                shift_ratio=self.DZI_SHIFT_RATIO,
                pad_scale=self.DZI_PAD_SCALE,
            )  # Dynamic zoom-in see the paper GDR-Net
        else:
            x1, y1, x2, y2 = bbox_xyxy.copy()
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            bbox_center = np.array([cx, cy])
            bbox_scale = self.DZI_PAD_SCALE * max(y2 - y1, x2 - x1)
            hr = (y2 - y1) / bbox_scale
            wr = (x2 - x1) / bbox_scale
            bbox_loc = np.array([0.5 - wr/2, 0.5 - hr/2, 0.5 + wr/2, 0.5 + hr/2])

        obj_inst_count = dataset_dict.pop('obj_inst_count')
        rot_index = 0
        if self.split == 'train': #### randomly replace the background if an image contains multiple instances of the same object ####
            if obj_inst_count[obj_id] > 2 and np.random.rand() < self.CHANGE_BG_PROB:
                image = self.replace_bg(image.copy(), mask)  # multiple instances in a ROI
            rot_index = np.random.randint(4)
        elif self.split == 'finetune':
            if np.random.rand() < self.CHANGE_BG_PROB:
                image = self.replace_bg(image.copy(), mask)  # multiple instances in a ROI
            rot_index = np.random.randint(4)

        rot_rad = rot_index * np.pi / 2
        Rz = np.array([[np.cos(rot_rad), -np.sin(rot_rad), 0.0],
                       [np.sin(rot_rad), np.cos(rot_rad), 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float32)
        obj_R = Rz @ obj_R
        obj_t = Rz @ obj_t
        roi_mask = misc.crop_resize_by_warp_affine(
            mask, bbox_center, bbox_scale, self.mask_size, cam_K, rot_rad, interpolation='bilinear'
        ).squeeze(0)  # HxW
        roi_mask = (roi_mask > 0.5).float()

        roi_img = misc.crop_resize_by_warp_affine(
            image, bbox_center, bbox_scale, self.rgb_size, cam_K, rot_rad, interpolation='bilinear'
        ) / 255.0  # HxWx3 -> 3xHxW

        T_rot = cam_K @ Rz @ np.linalg.inv(cam_K)
        center_hom = T_rot @ np.array(
            [*bbox_center, 1.0], dtype=np.float32)
        bbox_center = center_hom[:2]

        fx, fy, cx, cy = cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]
        fov = np.array([(bbox_center[0] - cx) / fx, (bbox_center[1] - cy) / fy, bbox_scale / fx])
        dataset_dict["fov"] = torch.as_tensor(fov, dtype=torch.float32)

        T_img2roi = misc.transform_to_local_ROIcrop(bbox_center=bbox_center, bbox_scale=bbox_scale, zoom_scale=self.rgb_size)
        roi_camK = T_img2roi.numpy() @ cam_K
        roi_PEmap = misc.generate_PEmap(im_hei=self.rgb_size, im_wid=self.rgb_size, cam_K=roi_camK) # 2xHxW

        Tz = np.array([[1.0, 0.0, -0.5],
                       [0.0, 1.0, -0.5],
                       [0.0, 0.0, 1.0]], dtype=np.float32)
        T_loc = np.linalg.inv(Tz) @ Rz @ Tz
        bbox_loc = utils.transform_bounding_box(bbox_loc, T_loc)
        bbox_map = utils.make_roi(torch.as_tensor(bbox_loc, dtype=torch.float32), self.rgb_size).unsqueeze(0)

        dataset_dict["roi_camK"] = torch.as_tensor(roi_camK, dtype=torch.float32).squeeze()      # 3x3
        dataset_dict["T_img2roi"] = torch.as_tensor(T_img2roi, dtype=torch.float32).squeeze()    # 3x3
        dataset_dict["roi_image"] = torch.as_tensor(roi_img, dtype=torch.float32).contiguous()   # 3xHxW
        dataset_dict["roi_mask"] = torch.as_tensor(roi_mask, dtype=torch.float32).contiguous()   # H/4xW/4 
        dataset_dict["roi_PEmap"] = torch.as_tensor(roi_PEmap, dtype=torch.float32).contiguous() # 2xHxW
        dataset_dict["obj_cls"] = torch.as_tensor(self.dataset_id2cls[obj_id], dtype=torch.int64)

        dataset_dict["roi_obj_t"] = torch.as_tensor(obj_t, dtype=torch.float32)          # object GT 3D location
        dataset_dict["roi_obj_R"] = torch.as_tensor(obj_R, dtype=torch.float32)          # object GT egocentric 3D orientation
        dataset_dict["bbox_scale"] = torch.as_tensor(bbox_scale, dtype=torch.float32)    # object (padded) bbox scale
        dataset_dict["bbox_center"] = torch.as_tensor(bbox_center, dtype=torch.float32)  # object bbox center
        dataset_dict["bbox_loc"] = torch.as_tensor(bbox_loc, dtype=torch.float32)  # relative bbox coordinates [0, 1] inside roi
        dataset_dict["bbox_map"] = torch.as_tensor(bbox_map, dtype=torch.float32)  # binary mask indicating the bounding box areas

        ######### CAD model ############################
        obj_symmetries = self.symmetries_dict[inst_infos['sub_dataset_folder']][str(inst_infos['obj_id'])]
        quat_symmetries = utils.rotation_to_quaternion(torch.from_numpy(obj_symmetries['rotation']).to(torch.float32))
        trans_symmetries = torch.from_numpy(obj_symmetries['translation']).to(torch.float32)
        assert len(quat_symmetries) == len(trans_symmetries)
        dataset_dict['vertices'] = model['vertices']
        dataset_dict['normals'] = model['normals']
        dataset_dict['vertices_norm'] = model['alpha']
        dataset_dict['vertices_mean'] = model['mean']
        dataset_dict['vertices_correlation'] = model['correlation']
        dataset_dict['diameter'] = torch.as_tensor(model['diameter'], dtype=torch.float32)
        dataset_dict['quaternion_symmetries'] = quat_symmetries
        dataset_dict['translation_symmetries'] = trans_symmetries
        dataset_dict['obj_id'] = obj_id
        dataset_dict['scene_id'] = scene_id
        dataset_dict['image_id'] = image_id

        ######### Quaternion conversion  ###############
        rot_ego = torch.reshape(dataset_dict["roi_obj_R"], (1, 3, 3))
        quat_ego = utils.rotation_to_quaternion(rot_ego)[0]
        dataset_dict['quat_ego'] = quat_ego
        quat_bin = np.load(inst_infos["quat_file"])
        dataset_dict['quat_bin'] = torch.as_tensor(quat_bin[rot_index], dtype=torch.float32)

        roi_delta_pxpy, roi_delta_tz = misc.convert_TxTyTz_to_delta_PxPyTz(T3=obj_t, camK=cam_K, bbox_center=bbox_center,
                                                                           bbox_scale=bbox_scale, zoom_scale=self.rgb_size)

        dataset_dict["roi_delta_tz"] = roi_delta_tz  # scale-invariant z-axis translation
        dataset_dict["roi_delta_pxpy"] = torch.as_tensor(roi_delta_pxpy, dtype=torch.float32)    # object GT scale-invariant projection shift delta_pxpy

        if self.Rz_rotation_aug: # rotation augmentation
            Rz_index = torch.randperm(4)[0] # 0:0˚, 1:90˚, 2:180˚, 3:270˚
            Rz_rad = torch.tensor([0.0, 0.0, math.pi * Rz_index * 0.5]) # 0˚, 90˚, 180˚, 270˚
            Rz_mat = euler_angles_to_matrix(Rz_rad, 'XYZ').type(torch.float32)

            roi_img = dataset_dict["roi_image"].clone()
            roi_mask = dataset_dict["roi_mask"].clone()

            ##### rotate the corresponding RGB, Mask, rotation, object projection
            if Rz_index == 1:
                roi_img = torch.flip(roi_img, [-2]).transpose(-1, -2)   # 90 deg
                roi_mask = torch.flip(roi_mask, [-2]).transpose(-1, -2) # 90 deg
            elif Rz_index == 2:
                roi_img = torch.flip(roi_img, [-1, -2])                 # 180 deg
                roi_mask = torch.flip(roi_mask, [-1, -2])               # 180 deg
            elif Rz_index == 3:
                roi_img = torch.flip(roi_img, [-1]).transpose(-1, -2)   # 270 deg
                roi_mask = torch.flip(roi_mask, [-1]).transpose(-1, -2) # 270 deg

            dataset_dict["roi_image"] = roi_img
            dataset_dict["roi_mask"] = roi_mask

            # calculate the object pose after in-plane rotation
            dataset_dict["roi_obj_R"] = Rz_mat @ dataset_dict["roi_obj_R"]
            dataset_dict["roi_delta_pxpy"] = Rz_mat[:2, :2] @ dataset_dict["roi_delta_pxpy"]

            # calculate the object location after in-plane rotation
            roi_obj_camK = dataset_dict["roi_camK"]
            roi_homo_proj = F.pad(dataset_dict["roi_delta_pxpy"] * self.rgb_size, pad=[0, 1], value=1.0)  # [s_zoom * delta_x, s_zoom * delta_y, 1.0]
            dataset_dict["roi_obj_t"] = self.rgb_size / bbox_scale * roi_delta_tz * torch.inverse(roi_obj_camK) @ roi_homo_proj  # r * delta_z * inv(K_B) @ P_B

        return dataset_dict

    @misc.lazy_property
    def _bg_img_paths(self):
        bg_type = self.BG_TYPE
        bg_root = self.BG_ROOT
        bg_num = self.NUM_BG_IMGS

        logger.info("get bg image paths")
        hashed_file_name = hashlib.md5(
            ("{}_{}_{}_get_bg_imgs".format(bg_root, bg_num, bg_type)).encode("utf-8")
        ).hexdigest()
        cache_path = osp.join(".cache/bg_paths_{}_{}.pkl".format(bg_type, hashed_file_name))
        mmcv.mkdir_or_exist(osp.dirname(cache_path))
        if osp.exists(cache_path):
            logger.info("get bg_paths from cache file: {}".format(cache_path))
            bg_img_paths = mmcv.load(cache_path)
            logger.info("num bg imgs: {}".format(len(bg_img_paths)))
            assert len(bg_img_paths) > 0
            return bg_img_paths

        logger.info("building bg imgs cache {}...".format(bg_type))
        assert osp.exists(bg_root), f"BG ROOT: {bg_root} does not exist"
        if bg_type == "coco":
            img_paths = [
                osp.join(bg_root, fn.name) for fn in os.scandir(bg_root) if ".png" in fn.name or "jpg" in fn.name
            ]
        elif bg_type == "VOC_table":  # used in original deepim
            VOC_root = bg_root  # path to "VOCdevkit/VOC2012"
            VOC_image_set_dir = osp.join(VOC_root, "ImageSets/Main")
            VOC_bg_list_path = osp.join(VOC_image_set_dir, "diningtable_trainval.txt")
            with open(VOC_bg_list_path, "r") as f:
                VOC_bg_list = [
                    line.strip("\r\n").split()[0] for line in f.readlines() if line.strip("\r\n").split()[1] == "1"
                ]
            img_paths = [osp.join(VOC_root, "JPEGImages/{}.jpg".format(bg_idx)) for bg_idx in VOC_bg_list]
        elif bg_type == "VOC":
            VOC_root = bg_root  # path to "VOCdevkit/VOC2012"
            img_paths = [
                osp.join(VOC_root, "JPEGImages", fn.name)
                for fn in os.scandir(osp.join(bg_root, "JPEGImages"))
                if ".jpg" in fn.name
            ]
        elif bg_type == "SUN2012":
            img_paths = [
                osp.join(bg_root, "JPEGImages", fn.name)
                for fn in os.scandir(osp.join(bg_root, "JPEGImages"))
                if ".jpg" in fn.name
            ]
        else:
            raise ValueError(f"BG_TYPE: {bg_type} is not supported")
        assert len(img_paths) > 0, len(img_paths)

        num_bg_imgs = min(len(img_paths), bg_num)
        bg_img_paths = np.random.choice(img_paths, num_bg_imgs)

        mmcv.dump(bg_img_paths, cache_path)
        logger.info("num bg imgs: {}".format(len(bg_img_paths)))
        assert len(bg_img_paths) > 0
        return bg_img_paths

    def trunc_mask(self, mask):
        # return the bool truncated mask
        mask = mask.copy().astype(np.bool)
        nonzeros = np.nonzero(mask.astype(np.uint8))
        x1, y1 = np.min(nonzeros, axis=1)
        x2, y2 = np.max(nonzeros, axis=1)
        c_h = 0.5 * (x1 + x2)
        c_w = 0.5 * (y1 + y2)
        rnd = random.random()
        if rnd < 0.2:  # block upper
            c_h_ = int(random.uniform(x1, c_h))
            mask[:c_h_, :] = False
        elif rnd < 0.4:  # block bottom
            c_h_ = int(random.uniform(c_h, x2))
            mask[c_h_:, :] = False
        elif rnd < 0.6:  # block left
            c_w_ = int(random.uniform(y1, c_w))
            mask[:, :c_w_] = False
        elif rnd < 0.8:  # block right
            c_w_ = int(random.uniform(c_w, y2))
            mask[:, c_w_:] = False
        else:
            pass
        return mask

    def replace_bg(self, im, im_mask, return_mask=False, truncate_fg=False,
                   synthesize_blending_artifacts=True):
        # add background to the image
        H, W = im.shape[:2]
        ind = random.randint(0, len(self._bg_img_paths) - 1)
        filename = self._bg_img_paths[ind]

        if self.BG_KEEP_ASPECT_RATIO:
            bg_img = self.get_bg_image(filename, H, W)
        else:
            bg_img = self.get_bg_image_v2(filename, H, W)

        if synthesize_blending_artifacts:
            idx = random.randint(0, len(self.dataset_dicts) - 1)
            sample_dict = self.dataset_dicts[idx]
            sample_info = sample_dict['inst_infos']
            sample_mask = misc.cocosegm2mask(
                sample_info['mask_file'], *im_mask.shape[:2])
            bx, by, bw, bh = sample_info['bbox']
            bbox_center = np.array([bx + bw / 2, by + bh / 2])
            y, x = np.where(im_mask)
            if y.size > 0:
                xmin, xmax = np.min(x), np.max(x)
                ymin, ymax = np.min(y), np.max(y)
                mask_center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
                shift = mask_center - bbox_center
                ys, xs = np.where(sample_mask)
                yt = np.minimum(np.maximum(ys + round(shift[1]), 0), H-1)
                xt = np.minimum(np.maximum(xs + round(shift[0]), 0), W-1)
                bg_img[yt, xt] = bg_img[ys, xs]

        if len(bg_img.shape) != 3:
            bg_img = np.zeros((H, W, 3), dtype=np.uint8)
            logger.warning("bad background image: {}".format(filename))

        mask = im_mask.copy().astype(bool)
        if truncate_fg:
            mask = self.trunc_mask(im_mask)
        mask_bg = ~mask
        bg_img = bg_img.astype(np.float32)
        im[mask_bg] = bg_img[mask_bg]
        if return_mask:
            return im, mask  # bool fg mask
        else:
            return im

    def get_bg_image(self, filename, imH, imW, channel=3):
        """keep aspect ratio of bg during resize target image size:

        imHximWxchannel.
        """
        target_size = min(imH, imW)
        max_size = max(imH, imW)
        real_hw_ratio = float(imH) / float(imW)
        bg_image = mmcv.imread(filename, 'color', self.img_format)
        bg_h, bg_w, bg_c = bg_image.shape
        bg_image_resize = np.zeros((imH, imW, channel), dtype="uint8")
        if (float(imH) / float(imW) < 1 and float(bg_h) / float(bg_w) < 1) or (
            float(imH) / float(imW) >= 1 and float(bg_h) / float(bg_w) >= 1
        ):
            if bg_h >= bg_w:
                bg_h_new = int(np.ceil(bg_w * real_hw_ratio))
                if bg_h_new < bg_h:
                    bg_image_crop = bg_image[0:bg_h_new, 0:bg_w, :]
                else:
                    bg_image_crop = bg_image
            else:
                bg_w_new = int(np.ceil(bg_h / real_hw_ratio))
                if bg_w_new < bg_w:
                    bg_image_crop = bg_image[0:bg_h, 0:bg_w_new, :]
                else:
                    bg_image_crop = bg_image
        else:
            if bg_h >= bg_w:
                bg_h_new = int(np.ceil(bg_w * real_hw_ratio))
                bg_image_crop = bg_image[0:bg_h_new, 0:bg_w, :]
            else:  # bg_h < bg_w
                bg_w_new = int(np.ceil(bg_h / real_hw_ratio))
                bg_image_crop = bg_image[0:bg_h, 0:bg_w_new, :]
        bg_image_resize_0 = misc.resize_short_edge(bg_image_crop, target_size, max_size)
        h, w, c = bg_image_resize_0.shape
        bg_image_resize[0:h, 0:w, :] = bg_image_resize_0
        return bg_image_resize

    def get_bg_image_v2(self, filename, imH, imW, channel=3):
        _bg_img = mmcv.imread(filename, 'color', self.img_format)
        try:
            # randomly crop a region as background
            bw = _bg_img.shape[1]
            bh = _bg_img.shape[0]
            x1 = np.random.randint(0, int(bw / 3))
            y1 = np.random.randint(0, int(bh / 3))
            x2 = np.random.randint(int(2 * bw / 3), bw)
            y2 = np.random.randint(int(2 * bh / 3), bh)
            bg_img = cv2.resize(_bg_img[y1:y2, x1:x2], (imW, imH),
                                interpolation=cv2.INTER_LINEAR)
        except:
            bg_img = np.zeros((imH, imW, 3), dtype=np.uint8)
            logger.warning("bad background image: {}".format(filename))
        return bg_img
