#!/bin/bash

conda create -n mrcnet -y python=3.10.13
conda activate mrcnet
conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install -r requirements.txt
pip install "git+https://github.com/thodan/bop_toolkit"
pip install spatial-correlation-sampler==0.4.0
