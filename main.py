import os
import cv2
import torch
import random
import numpy as np
import bop_dataset as data
import models
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from lib import warmup_lr
import config as bop_cfg


def worker_init_fn(*_):
    # each worker should only use one os thread
    # numpy/cv2 takes advantage of multithreading by default
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    cv2.setNumThreads(0)

    # random seed
    np.random.seed(bop_cfg.RANDOM_SEED)


def main_worker(rank, world_size, args):
    # Fix random seed
    random.seed(bop_cfg.RANDOM_SEED)
    np.random.seed(bop_cfg.RANDOM_SEED)
    torch.manual_seed(bop_cfg.RANDOM_SEED)
    torch.backends.cudnn.benchmark = True

    if args.is_parallel:
        world_rank = int(os.environ['RANK']) if 'RANK' in os.environ else rank
        print('Running DDP on rank {:d}.'.format(world_rank))
        dist.init_process_group(
            'nccl', rank=world_rank, world_size=world_size)
        torch.cuda.set_device(rank)
    else:
        world_rank = rank

    train_dataset = data.BOP_Dataset(
        args.dataset, split='train')
    data_info = train_dataset.get_info()

    if args.is_parallel:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=train_sampler is None,
        pin_memory=True, num_workers=args.num_workers,
        drop_last=True, sampler=train_sampler,
        collate_fn=data.collate_fn)

    model = models.MRCNet(
        dataset=args.dataset,
        n_decoders=data_info['num_objects'],
        depth_min=data_info['depth_min'],
        depth_max=data_info['depth_max'],
        n_depth_bin=bop_cfg.Tz_BINS_NUM).to(rank)

    if args.is_parallel:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], static_graph=True,
            find_unused_parameters=False, gradient_as_bucket_view=True)

    def group_parameters(model):
        decay = set()
        no_decay = set()
        for mn, md in model.named_modules():
            for pn, p in md.named_parameters():
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(p)
                elif isinstance(md, (torch.nn.BatchNorm2d,
                                     torch.nn.GroupNorm)):
                    no_decay.add(p)
                elif isinstance(md, (torch.nn.Conv2d, torch.nn.Linear,
                                     models.CondConv)):
                    decay.add(p)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, \
            "parameter %s made it into both decay/no_decay sets!" % (
                str(inter_params), )
        assert len(param_dict) - len(union_params) == 0, \
            "parameter %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": list(decay), "weight_decay": bop_cfg.DECAY_WEIGHT},
            {"params": list(no_decay), "weight_decay": 0.0}]
        return optim_groups

    total_batches = len(train_loader) * args.n_epochs
    init_lr = bop_cfg.START_LR
    end_lr = bop_cfg.END_LR
    grouped_parameters = group_parameters(model)
    optimizer = torch.optim.AdamW(grouped_parameters, lr=init_lr)
    scheduler = warmup_lr.CosineAnnealingWarmupRestarts(
        optimizer, total_batches, warmup_steps=args.warmup_step,
        max_lr=init_lr, min_lr=end_lr)
    start_epoch = 0

    if world_rank == 0:
        train_writer = SummaryWriter(log_dir=os.path.join(
            args.log_dir, 'train'))
        if not os.path.exists(args.chkpt_dir):
            os.makedirs(args.chkpt_dir)

    train_step = 0
    target_keys = ['quat_ego', 'roi_obj_R', 'fov',
                   'roi_obj_t', 'vertices', 'vertices_mean', 'bbox_map',
                   'obj_id', 'obj_cls', 'roi_mask', 'quat_bin', 'diameter',
                   'vertices_mask', 'quaternion_symmetries',
                   'translation_symmetries', 'symmetries_mask',
                   'vertices_correlation', 'roi_camK']
    for epoch in range(start_epoch, args.n_epochs):
        if args.is_parallel:
            train_sampler.set_epoch(epoch)
        # Start main training loop
        with tqdm(desc='Epoch {:03d} - '.format(epoch), unit='iter',
                  total=len(train_loader), disable=rank != 0) as pbar:
            for it, batch in enumerate(train_loader):
                model.train()
                batch_image = batch['roi_image']
                batch_image = batch_image.to(rank, non_blocking=True)
                batch_bbox_map = batch['bbox_map'].to(rank, non_blocking=True)
                batch_image_roi = torch.concat([
                    batch_image, batch_bbox_map], dim=1)

                targets = dict()
                for key in target_keys:
                    batch_value = batch[key].to(rank, non_blocking=True)
                    targets[key] = batch_value

                predictions = model(
                    batch_image_roi,
                    {'obj_cls': targets['obj_cls'],
                     'fov': targets['fov'],
                     'intrinsics': targets['roi_camK']}, targets)
                loss_dict = predictions['losses']
                loss = sum(loss_dict.values())
                loss.backward()

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr = optimizer.param_groups[0]['lr']

                # Log training progress
                if world_rank == 0:
                    loss_val = sum([v for k, v in loss_dict.items()])
                    train_writer.add_scalar('Loss', loss_val, train_step)
                    train_writer.add_scalar('Learning_rate', lr, train_step)
                    train_writer.add_scalar(
                        'Weight_decay', bop_cfg.DECAY_WEIGHT, train_step)
                    for name, val in loss_dict.items():
                        train_writer.add_scalar(
                            'Loss/' + name, val, train_step)

                pbar.update()
                pbar.set_postfix(loss=loss.item())
                train_step += 1
                scheduler.step()

            # Save models; this only needs to be done in one replica
            if world_rank == 0:
                if args.is_parallel:
                    state_dict = {
                        'network': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'last_epoch': epoch}
                else:
                    state_dict = {
                        'network': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'last_epoch': epoch}
                model_path = os.path.join(
                    args.chkpt_dir, 'epoch_{:03d}.pth'.format(epoch))
                torch.save(state_dict, model_path)
                print('Saved model to {}'.format(model_path))

    if args.is_parallel:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
                        help='minibatch size for data')
    parser.add_argument('--is_parallel', action='store_true',
                        help='whether to enable multi-gpu training')
    parser.add_argument('--n_epochs', type=int, default=120,
                        help='total number of epochs')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers for parallel data loading')
    parser.add_argument('--dataset', type=str, default='lm',
                        help='name of the dataset for training and validation')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='directory holding tensorboard logs')
    parser.add_argument('--chkpt_dir', type=str, default='./checkpoints',
                        help='directory holding all model checkpoints')
    parser.add_argument('--warmup_step', type=int, default=1000,
                        help='number of steps to warmup learning rate')
    args = parser.parse_args()

    if args.is_parallel:
        # Training on single node
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            world_size = torch.cuda.device_count()
            ngpus = torch.cuda.device_count()
            args.batch_size = int(args.batch_size / world_size)
            args.num_workers = int(args.num_workers / world_size)
            mp.spawn(
                main_worker, args=(world_size, args), nprocs=ngpus, join=True)

        # Training on multiple nodes
        else:
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])
            args.batch_size = int(args.batch_size / world_size)
            args.num_workers = int(args.num_workers / world_size)
            main_worker(local_rank, world_size, args)
    else:
        main_worker(0, 1, args)
