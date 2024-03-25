INPUT_SIZE = 256
CELL_SIZE = 64
MASK_SIZE = 64
USE_6D = True
N_POSE_BIN = 4608  # number of bins for 3D orientation
N_BIN_TO_KEEP = 9

WEIGHT_TRANS_CLF = 2.0
WEIGHT_DEPTH_CLF = 2.0  # 1.0 for hard labels
WEIGHT_QUAT_CLF = 0.5
WEIGHT_MASK = 10.0
WEIGHT_QUAT_REG = 1.0
WEIGHT_TRANS_REG = 1.0
WEIGHT_DEPTH_REG = 0.2

RANDOM_SEED = 2022      # random seed
INPUT_IMG_SIZE = 256    # the input image size of network
OUTPUT_MASK_SIZE = 64   # the output mask size of network
Tz_BINS_NUM = 1000      # the number of discretized depth bins
POSE_SIGMA = 0.03  # standard deviation of quaternion bin distribution
DEPTH_SIGMA = 0.5     # standard deviation of depth Gaussian distribution
TRANS_SIGMA = 10 / INPUT_SIZE  # standard deviation of noise in 2D center

DATASET_ROOT = "./data/bop_datasets"
EVAL_ROOT = "./data/bop_datasets_eval"
VOC_BG_ROOT = "./data/VOCdevkit/VOC2012"

END_LR = 5e-6
START_LR = 2e-4
END_LR_FT = 5e-7
START_LR_FT = 5e-6
DECAY_WEIGHT = 1e-3

USE_CACHE = True
CACHE_MASK = True
ZOOM_PAD_SCALE = 1.5
ZOOM_SCALE_RATIO = 0.25
ZOOM_SHIFT_RATIO = 0.25   # center shift
COLOR_AUG_PROB = 0.8
CHANGE_BG_PROB = 0.8      # the prob for changing the background

RZ_ROTATION_AUG = False

DATASET_CONFIG = {
    'tless': {
        'width': 720,
        'height': 540,
        'Tz_near': 0.050,
        'Tz_far': 1.700,
        'num_class': 30,
        'id2mod': {v+1: "obj_{:02d}".format(v+1) for v in range(30)},
        'id2cls': {v+1: v for v in range(30)},  # from object_id to class_index
        'model_folders': {
            'train_pbr': 'models_cad',
            'train_primesense': 'models_cad',
            'test_primesense': 'models_cad'},
        'train_set': ['train_pbr'],
        'finetune_set': ['train_primesense'],
        'test_set': ['test_primesense'],
    },  # √
    'itodd': {
        'width': 1280,
        'height': 960,
        'Tz_near': 0.01,
        'Tz_far': 1.85,
        'num_class': 28,
        'id2mod': {v+1: "obj_{:02d}".format(v+1) for v in range(28)},
        'id2cls': {v+1: v for v in range(28)},
        'model_folders': {
            'train_pbr': 'models',
            'val': 'models',
            'test': 'models'},
        'train_set': ['train_pbr'],
        'test_set': ['val'],
    },
    'ycbv': {
        'width': 640,
        'height': 480,
        'Tz_near': 0.030,
        'Tz_far': 2.000,
        'num_class': 21,
        'id2mod': {v+1: "obj_{:02d}".format(v+1) for v in range(21)},
        'id2cls': {v+1: v for v in range(21)},  # from object_id to class_index
        'model_folders': {
            'train_pbr': 'models',
            'train_real': 'models',
            'train_synt': 'models',
            'test': 'models'},
        'train_set': ['train_pbr'],
        'finetune_set': ['train_real', 'train_synt'],
        'test_set': ['test'],
    },  # √
    'lmo': {
        'width': 640,
        'height': 480,
        'Tz_near': 0.010,
        'Tz_far': 2.150,
        'num_class': 15,
        'id2mod': {v: "obj_{:02d}".format(v) for v in [
            1, 5, 6, 8, 9, 10, 11, 12]},
        'id2cls': {v: i for i, v in enumerate([
            1, 5, 6, 8, 9, 10, 11, 12])},  # from object_id to class_index
        'model_folders': {
            'train_pbr': 'models',
            'test': 'models'},
        'train_set': ['train_pbr'],
        'finetune_set': ['train'],
        'test_set': ['test'],
    },  # √
}
