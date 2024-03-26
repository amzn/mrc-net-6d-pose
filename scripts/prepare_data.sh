#!/bin/bash
# CosyPose detections
if ! [ -d bop22_default_detections_and_segmentations ]; then
  wget https://bop.felk.cvut.cz/media/data/bop_datasets_extra/bop22_default_detections_and_segmentations.zip
  unzip bop22_default_detections_and_segmentations.zip
fi

# VOC
if ! [ -d ./data/VOCdevkit/VOC2012 ]; then
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -P ./data
  tar xf ./data/VOCtrainval_11-May-2012.tar -C ./data
fi

if ! [ -d ./data/bop_datasets ]; then
  mkdir -p ./data/bop_datasets
fi

if ! [ -d ./data/bop_datasets_eval ]; then
  mkdir -p ./data/bop_datasets_eval
fi

# TLESS
if ! [ -d ./data/bop_datasets/tless ]; then
  mkdir -p ./data/bop_datasets/tless
  wget https://bop.felk.cvut.cz/media/data/bop_datasets/tless_train_pbr.zip -P ./data/bop_datasets
  unzip ./data/bop_datasets/tless_train_pbr.zip -d ./data/bop_datasets/tless
  wget https://bop.felk.cvut.cz/media/data/bop_datasets/tless_train_primesense.zip -P ./data/bop_datasets
  unzip ./data/bop_datasets/tless_train_primesense.zip -d ./data/bop_datasets/tless
  wget https://bop.felk.cvut.cz/media/data/bop_datasets/tless_test_primesense_all.zip -P ./data/bop_datasets
  unzip ./data/bop_datasets/tless_test_primesense_all.zip -d ./data/bop_datasets/tless
  wget https://bop.felk.cvut.cz/media/data/bop_datasets/tless_models.zip -P ./data/bop_datasets
  unzip ./data/bop_datasets/tless_models.zip -d ./data/bop_datasets/tless
  python precompute_quaternion_labels.py --dataset tless
  python -c 'import bop_dataset; bop_dataset.BOP_Dataset("tless", split="train")'
fi

# LMO
if ! [ -d ./data/bop_datasets/lmo ]; then
  mkdir -p ./data/bop_datasets/lmo
  wget https://bop.felk.cvut.cz/media/data/bop_datasets/lm_train_pbr.zip -P ./data/bop_datasets
  unzip ./data/bop_datasets/lm_train_pbr.zip -d ./data/bop_datasets/lmo
  wget https://bop.felk.cvut.cz/media/data/bop_datasets/lmo_test_all.zip -P ./data/bop_datasets
  unzip ./data/bop_datasets/lmo_test_all.zip -d ./data/bop_datasets/lmo
  wget https://bop.felk.cvut.cz/media/data/bop_datasets/lm_models.zip -P ./data/bop_datasets
  unzip ./data/bop_datasets/lm_models.zip -d ./data/bop_datasets/lmo
  python precompute_quaternion_labels.py --dataset lmo
  python -c 'import bop_dataset; bop_dataset.BOP_Dataset("lmo", split="train")'
fi

# YCBV
if ! [ -d ./data/bop_datasets/ycbv ]; then
  mkdir -p ./data/bop_datasets/ycbv
  wget https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_train_pbr.zip -P ./data/bop_datasets
  unzip ./data/bop_datasets/ycbv_train_pbr.zip -d ./data/bop_datasets/ycbv
  wget https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_train_real.zip -P ./data/bop_datasets
  unzip ./data/bop_datasets/ycbv_train_real.zip -d ./data/bop_datasets/ycbv
  wget https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_test_all.zip -P ./data/bop_datasets
  unzip ./data/bop_datasets/ycbv_test_all.zip -d ./data/bop_datasets/ycbv
  wget https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_models.zip -P ./data/bop_datasets
  unzip ./data/bop_datasets/ycbv_models.zip -d ./data/bop_datasets/ycbv
  python precompute_quaternion_labels.py --dataset ycbv
  python -c 'import bop_dataset; bop_dataset.BOP_Dataset("ycbv", split="train")'
fi

# ITODD
if ! [ -d ./data/bop_datasets/itodd ]; then
  mkdir -p ./data/bop_datasets/itodd
  wget https://bop.felk.cvut.cz/media/data/bop_datasets/itodd_train_pbr.zip -P ./data/bop_datasets
  unzip ./data/bop_datasets/itodd_train_pbr.zip -d ./data/bop_datasets/itodd
  wget https://bop.felk.cvut.cz/media/data/bop_datasets/itodd_test_all.zip -P ./data/bop_datasets
  unzip ./data/bop_datasets/itodd_test_all.zip -d ./data/bop_datasets/itodd
  wget https://bop.felk.cvut.cz/media/data/bop_datasets/itodd_models.zip -P ./data/bop_datasets
  unzip ./data/bop_datasets/itodd_models.zip -d ./data/bop_datasets/itodd
  python precompute_quaternion_labels.py --dataset itodd
  python -c 'import bop_dataset; bop_dataset.BOP_Dataset("itodd", split="train")'
fi
