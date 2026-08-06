from easydict import EasyDict as edict

__C                                              = edict()
cfg                                              = __C

#
# Dataset Config
#
__C.DATASETS                                     = edict()
__C.DATASETS.SHAPENET                            = edict()
__C.DATASETS.SHAPENET.CATEGORY_FILE_PATH         = 'datasets/ShapeNet.json'
__C.DATASETS.SHAPENET.N_RENDERINGS               = 8
__C.DATASETS.SHAPENET.N_POINTS                   = 2048
__C.DATASETS.SHAPENET.PARTIAL_POINTS_PATH        = 'data/PCN/%s/partial/%s/%s/%02d.pcd'
__C.DATASETS.SHAPENET.COMPLETE_POINTS_PATH       = 'data/PCN/%s/complete/%s/%s.pcd'

# Used to test KITTI dataset
__C.DATASETS.SHAPENET.ONLY_CAR         = False
# __C.DATASETS.SHAPENET.ONLY_CAR         = True

#
# Dataset
#
__C.DATASET                                      = edict()
# Dataset Options: Completion3D, ShapeNet, ShapeNetCars, Completion3DPCCT
__C.DATASET.TRAIN_DATASET                        = 'ShapeNet'
__C.DATASET.TEST_DATASET                         = 'ShapeNet'
__C.DATASET.VAL_DATASET                         = 'ShapeNet'

#
# Constants
#
__C.CONST                                        = edict()

__C.CONST.NUM_WORKERS                            = 12
__C.CONST.N_INPUT_POINTS                         = 2048

#
# Directories
#

__C.DIR                                          = edict()
__C.DIR.OUT_PATH                                 = 'Ours_PCN'
__C.CONST.DEVICE                                 = '0, 1, 2, 3'
#__C.CONST.WEIGHTS                                = 'Ours_PCN/checkpoints/2025-12-11T10:21:33.414037/ckpt-best.pth'

# Memcached
#
__C.MEMCACHED                                    = edict()
__C.MEMCACHED.ENABLED                            = False
__C.MEMCACHED.LIBRARY_PATH                       = '/mnt/lustre/share/pymc/py3'
__C.MEMCACHED.SERVER_CONFIG                      = '/mnt/lustre/share/memcached_client/server_list.conf'
__C.MEMCACHED.CLIENT_CONFIG                      = '/mnt/lustre/share/memcached_client/client.conf'

#
# Network
#
__C.NETWORK                                      = edict()
__C.NETWORK.N_SAMPLING_POINTS                    = 2048
__C.NETWORK.step1                    = 4
__C.NETWORK.step2                    = 8
__C.NETWORK.merge_points = 512
__C.NETWORK.local_points = 512
__C.NETWORK.view_distance = 0.7
__C.NETWORK.splat_kernel = 4
__C.NETWORK.splat_sigma = 1.5
__C.NETWORK.tinyvit_variant                      = '5m'
__C.NETWORK.tinyvit_pretrained_path              = 'models/tinyvit/tiny_vit_5m_22kto1k_distill.pth'
#
# Train
#
__C.TRAIN                                        = edict()
__C.TRAIN.BATCH_SIZE                             = 27
__C.TRAIN.N_EPOCHS                               = 420
__C.TRAIN.SAVE_FREQ                              = 50
__C.TRAIN.LEARNING_RATE                          = 0.0002
#__C.TRAIN.LR_MILESTONES                          = [50, 100, 150, 200, 250]
__C.TRAIN.WARMUP_STEPS                           = 20
__C.TRAIN.BETAS                                  = (.9, .999)
__C.TRAIN.WEIGHT_DECAY                           = 0
__C.TRAIN.LR_SCHEDULER                           = edict()
__C.TRAIN.LR_SCHEDULER.TYPE                      = 'CosineAnnealingLR'
__C.TRAIN.LR_SCHEDULER.LOWEST_DECAY              = 0.005  # eta_min = LR * LOWEST_DECAY

#
# Test
#
__C.TEST                                         = edict()
__C.TEST.METRIC_NAME                             = 'ChamferDistance'