import json
import torch
import utils
import trimesh
import argparse
import config as cfg
import multiprocessing
import numpy as np
from tqdm import tqdm
from pathlib import Path


def process_folder(rank, obj_folder, model_folder, n_objs=30):
    image_folder = obj_folder / 'rgb'
    image_files = list(image_folder.glob('*.png')) \
        + list(image_folder.glob('*.jpg'))
    model_dict = dict()
    symmetries = dict()
    with open(model_folder / 'models_info.json', 'r') as fp:
        model_info = json.load(fp)
    for obj_id in range(1, n_objs + 1):
        info = model_info[str(obj_id)]
        model_path = model_folder / 'obj_{:06d}.ply'.format(obj_id)
        mesh = trimesh.load(model_path)
        vertices = mesh.vertices.copy()
        sigma = vertices.T @ vertices / len(vertices)
        norm = np.mean(np.sum(vertices**2, axis=-1), keepdims=True)
        diameter = np.array(info['diameter'])
        model_dict[obj_id] = {
            'vertices': torch.from_numpy(
                vertices).unsqueeze(0).to(torch.float32).to(rank),
            'diameter': torch.from_numpy(diameter).unsqueeze(0).to(rank),
            'norm': torch.from_numpy(norm).unsqueeze(0).to(rank),
            'correlation': torch.from_numpy(sigma).unsqueeze(0).to(rank)}
        rotations_sym, translations_sym = \
            utils.get_symmetry_transformations(info, 0.01)
        symmetries[obj_id] = {
            'rotation': torch.from_numpy(rotations_sym).to(rank),
            'translation': torch.from_numpy(translations_sym).to(rank)}

    with open(obj_folder / 'scene_gt.json', 'r') as fp:
        annotations_3d = json.load(fp)
    label_path = obj_folder / 'quat_label'
    label_path.mkdir(exist_ok=True)
    for obj_image_file in tqdm(image_files):
        image_id = int(obj_image_file.stem)

        for gt_id, anno_3d in enumerate(annotations_3d[str(image_id)]):
            obj_id = anno_3d['obj_id']

            rotation = np.array(anno_3d['cam_R_m2c']).astype(
                "float32").reshape(3, 3)
            diameter = model_dict[obj_id]['diameter']

            categories = []
            for theta in [0.0, 0.5*np.pi, np.pi, 1.5*np.pi]:
                Rz = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                               [np.sin(theta), np.cos(theta), 0.0],
                               [0.0, 0.0, 1.0]], dtype=np.float32)
                quaternion = utils.rotation_to_quaternion(
                    torch.from_numpy(Rz @ rotation).unsqueeze(0))
                vertices = model_dict[obj_id]['vertices']
                vertices_mask = torch.ones_like(vertices[..., -1])
                v_corr = model_dict[obj_id]['correlation'].to(torch.float32)
                R_sym = symmetries[obj_id]['rotation'].to(torch.float32)
                q_sym = utils.rotation_to_quaternion(
                    R_sym).unsqueeze(0).to(torch.float32)
                t_sym = symmetries[obj_id][
                    'translation'].unsqueeze(0).to(torch.float32)
                mask_sym = torch.ones_like(q_sym[..., -1]).to(torch.bool)

                categories.append(utils.quantize_quaternion_vertex(
                    quaternion.to(rank), vertices, vertices_mask, diameter,
                    v_corr, q_sym, t_sym, mask_sym)[0])

            categories = torch.concat(categories, dim=0)
            np.save(label_path / '{:06d}_{:06d}.npy'.format(
                image_id, gt_id), categories.cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tless',
                        help='name of the dataset for training and validation')
    args = parser.parse_args()

    root_dir = Path(cfg.DATASET_ROOT)
    obj_dataset = args.dataset
    n_objs = cfg.DATASET_CONFIG[obj_dataset]['num_class']
    max_procs = 8
    gpu_count = torch.cuda.device_count()
    for data_folder, model_path in cfg.DATASET_CONFIG[
            obj_dataset]['model_folders'].items():
        obj_dataset_dir = root_dir / obj_dataset / data_folder
        model_folder = root_dir / obj_dataset / model_path
        print('Processing {}'.format(data_folder))
        procs = []
        for folder_id, obj_folder in enumerate(obj_dataset_dir.iterdir()):
            p = multiprocessing.Process(
                target=process_folder,
                args=(folder_id % gpu_count, obj_folder, model_folder, n_objs))
            p.start()
            procs.append(p)
            if len(procs) > max_procs:
                for p in procs:
                    p.join()
                procs = []
