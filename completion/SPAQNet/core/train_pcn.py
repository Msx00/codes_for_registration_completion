import logging
import os
from collections import defaultdict, OrderedDict
import torch
import utils.data_loaders
import utils.helpers
import argparse
from datetime import datetime
from tqdm import tqdm
from time import time
import time as _time
from tensorboardX import SummaryWriter
from core.test_pcn import test_net
from utils.average_meter import AverageMeter
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.schedular import GradualWarmupScheduler
from utils.loss_utils import get_loss_HyperCD
from models.model_utils import SoftSplatCCM
from models.SplAttN import Model
from utils import misc, dist_utils
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'  # Adjust according to your GPU (RTX 4090 is 8.9)

def build_cosine_scheduler(optimizer, cfg, warmup_epochs, last_epoch=-1):
    scheduler_cfg = getattr(cfg.TRAIN, 'LR_SCHEDULER', None)
    lowest_decay = getattr(scheduler_cfg, 'LOWEST_DECAY', 0.02)
    # Ensure cosine phase spans remaining epochs after warmup
    t_max = max(1, cfg.TRAIN.N_EPOCHS - warmup_epochs)
    eta_min = cfg.TRAIN.LEARNING_RATE * lowest_decay
    return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min, last_epoch=last_epoch)

def train_net(args, cfg):
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TRAIN), shuffle=True)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TRAIN),
                                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                                    num_workers=cfg.CONST.NUM_WORKERS,
                                                    collate_fn=utils.data_loaders.collate_fn,
                                                    pin_memory=True,
                                                    drop_last=False,
                                                    prefetch_factor=16,
                                                    persistent_workers=True,
                                                    sampler=train_sampler)
    val_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST), shuffle=False)
    val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST),
                                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                                  num_workers=cfg.CONST.NUM_WORKERS//2,
                                                  collate_fn=utils.data_loaders.collate_fn,
                                                  pin_memory=True,
                                                  drop_last=False,
                                                  prefetch_factor=4,
                                                  persistent_workers=True,
                                                  sampler=val_sampler)

    # Set up folders for logs and checkpoints
    # Create tensorboard writers
    if args.local_rank == 0:
        output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', datetime.now().isoformat())
        cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
        cfg.DIR.LOGS = output_dir % 'logs'
        train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
        val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))
        if not os.path.exists(cfg.DIR.CHECKPOINTS):
            os.makedirs(cfg.DIR.CHECKPOINTS)
    else:
        train_writer = None
        val_writer = None

    model = Model(cfg)
    if torch.cuda.is_available():
        model.to(args.local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
    
    # Create the optimizers
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=cfg.TRAIN.LEARNING_RATE,
                                 weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                 betas=cfg.TRAIN.BETAS)

    # lr scheduler
    warmup_epochs = min(cfg.TRAIN.WARMUP_STEPS, max(1, cfg.TRAIN.N_EPOCHS - 1))
    cosine_scheduler = build_cosine_scheduler(optimizer, cfg, warmup_epochs)
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                          after_scheduler=cosine_scheduler)

    init_epoch = 0
    best_metrics = float('inf')
    BestEpoch = 0
    splat_kernel = getattr(cfg.NETWORK, 'splat_kernel', 3)
    splat_sigma = getattr(cfg.NETWORK, 'splat_sigma', 1.5)
    render = SoftSplatCCM(TRANS=-cfg.NETWORK.view_distance,
                          RESOLUTION=224,
                          kernel_size=splat_kernel,
                          sigma=splat_sigma)

    grad_clip = getattr(cfg.TRAIN, 'GRAD_CLIP_NORM', 10.0)

    if 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        # map_location='cpu' is safer for loading
        checkpoint = torch.load(cfg.CONST.WEIGHTS, map_location='cpu') 
        # Allow loading when new modules (e.g., img_pos_embed) are absent from older checkpoints.
        model.load_state_dict(checkpoint['model'], strict=False)
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError:
            logging.warning('Skip optimizer state: param groups mismatch with current model.')
        
        warmup_epochs = min(cfg.TRAIN.WARMUP_STEPS, max(1, cfg.TRAIN.N_EPOCHS - 1))
        lr_scheduler = build_cosine_scheduler(optimizer, cfg, warmup_epochs, last_epoch=warmup_epochs)
        optimizer.param_groups[0]['lr']= cfg.TRAIN.LEARNING_RATE

        logging.info('Recover complete.')

    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
        _time.sleep(1)
        train_sampler.set_epoch(epoch_idx)
        epoch_start_time = time()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        model.train()

        epoch_loss_totals = defaultdict(float)

        batch_end_time = time()
        n_batches = len(train_data_loader)
        if args.local_rank == 0:
            print('epoch: ', epoch_idx, 'optimizer: ', optimizer.param_groups[0]['lr'])
        with tqdm(train_data_loader, disable=args.local_rank != 0) as t:
            for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(t):
                data_time.update(time() - batch_end_time)
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)
                partial = data['partial_cloud']
                gt = data['gtcloud']

                partial_depth = render.get_CCM(partial)
                
                optimizer.zero_grad()

                pcds_pred = model(partial, partial_depth)

                pcds_pred = [p.float() for p in pcds_pred]
                partial = partial.float()
                gt = gt.float()

                loss_total, losses = get_loss_HyperCD(
                    pcds_pred,
                    partial,
                    gt,
                    sqrt=True
                )

                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, norm_type=2)
                optimizer.step()

                reduced_losses = [dist_utils.reduce_tensor(loss, args) for loss in losses]
                torch.cuda.synchronize()

                scaled_losses = [loss.item() * 1e3 for loss in reduced_losses]
                extra_count = max(0, len(scaled_losses) - 4)
                loss_names = ['cd_pc', 'cd_p1', 'cd_p2'] + [f'cd_extra_{i+1}' for i in range(extra_count)] + ['partial']
                loss_log = OrderedDict((name, val) for name, val in zip(loss_names, scaled_losses))
                for name, val in loss_log.items():
                    epoch_loss_totals[name] += val
                n_itr = (epoch_idx - 1) * n_batches + batch_idx

                if train_writer is not None:
                    for name, val in loss_log.items():
                        train_writer.add_scalar(f'Loss/Batch/{name}', val, n_itr)
                    batch_time.update(time() - batch_end_time)
                    batch_end_time = time()
                    t.set_description('[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches))
                    t.set_postfix({k: f'{v:.4f}' for k, v in list(loss_log.items())})

        epoch_loss_avgs = OrderedDict((name, total / n_batches) for name, total in epoch_loss_totals.items())

        lr_scheduler.step()
        epoch_end_time = time()
        if train_writer is not None:
            for name, avg in epoch_loss_avgs.items():
                train_writer.add_scalar(f'Loss/Epoch/{name}', avg, epoch_idx)
            logging.info('[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
                         (epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time,
                          ['%s: %.4f' % (name, avg) for name, avg in epoch_loss_avgs.items()]))

        # ================= [Fix memory overflow] =================
        # 1. Pre-validation cleanup: release training graph and intermediate variables
        import gc
        del pcds_pred, partial, gt, partial_depth, loss_total, losses
        gc.collect()
        # ==================================================

        # Validate the current model
        # Note: Ensure test_net uses with torch.no_grad(): internally
        cd_eval = test_net(args, cfg, epoch_idx, val_data_loader, val_writer, model)
        
        # Save checkpoints
        if args.local_rank == 0:
            if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or cd_eval < best_metrics:
                if cd_eval < best_metrics:
                    best_metrics = cd_eval
                    BestEpoch = epoch_idx
                    file_name = 'ckpt-best.pth'

                else:
                    file_name = 'ckpt-epoch-%03d.pth' % epoch_idx
                output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, output_path)

                logging.info('Saved checkpoint to %s ...' % output_path)
            logging.info('Best Performance: Epoch %d -- CD %.4f' % (BestEpoch,best_metrics))

    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()