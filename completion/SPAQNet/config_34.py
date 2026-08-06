from easydict import EasyDict as edict

__C = edict()
cfg = __C

#
# Dataset Config
#
__C.DATASETS = edict()

# ShapeNet34 (uses ShapeNet55-style indexing: datasets/ShapeNet34/{train,test}.txt)
__C.DATASETS.SHAPENET34 = edict()
__C.DATASETS.SHAPENET34.CATEGORY_FILE_PATH = 'datasets/ShapeNet-Unseen21'
__C.DATASETS.SHAPENET34.N_POINTS = 2048
# `line` format in txt is like: <taxonomy_id>-<model_id>.ply
__C.DATASETS.SHAPENET34.COMPLETE_POINTS_PATH = 'data/ShapeNet55-34/shapenet_pc/%s'

#
# Dataset
#
__C.DATASET = edict()
__C.DATASET.TRAIN_DATASET = 'ShapeNet34'
__C.DATASET.TEST_DATASET = 'ShapeNet-Unseen21'

#
# Constants
#
__C.CONST = edict()
__C.CONST.NUM_WORKERS = 12
__C.CONST.N_INPUT_POINTS = 2048
__C.CONST.WEIGHTS = 'Ours_34/checkpoints/2025-12-23T14:35:58.613921/ckpt-best.pth'
# eval crop mode: easy|median|hard
__C.CONST.mode = 'hard'

#
# Directories
#
__C.DIR = edict()
__C.DIR.OUT_PATH = 'Ours_34'
__C.CONST.DEVICE = '0, 1, 2, 3'

#
# Memcached
#
__C.MEMCACHED = edict()
__C.MEMCACHED.ENABLED = False
__C.MEMCACHED.LIBRARY_PATH = '/mnt/lustre/share/pymc/py3'
__C.MEMCACHED.SERVER_CONFIG = '/mnt/lustre/share/memcached_client/server_list.conf'
__C.MEMCACHED.CLIENT_CONFIG = '/mnt/lustre/share/memcached_client/client.conf'

#
# Network
#
__C.NETWORK = edict()
__C.NETWORK.step1 = 2
__C.NETWORK.step2 = 4
__C.NETWORK.merge_points = 1024
__C.NETWORK.local_points = 1024
__C.NETWORK.splat_kernel = 4
__C.NETWORK.splat_sigma = 1.5
__C.NETWORK.view_distance = 1.5
__C.NETWORK.tinyvit_variant = '5m'
#__C.NETWORK.tinyvit_pretrained_path = 'models/tinyvit/tiny_vit_5m_22kto1k_distill.pth'

#
# Train
#
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 40
__C.TRAIN.N_EPOCHS = 420
__C.TRAIN.SAVE_FREQ = 5
__C.TRAIN.LEARNING_RATE = 0.0002
__C.TRAIN.WARMUP_STEPS = 20
__C.TRAIN.BETAS = (.9, .999)
__C.TRAIN.WEIGHT_DECAY = 0
__C.TRAIN.LR_SCHEDULER = edict()
__C.TRAIN.LR_SCHEDULER.TYPE = 'CosineAnnealingLR'
__C.TRAIN.LR_SCHEDULER.LOWEST_DECAY = 0.01  # eta_min = LR * LOWEST_DECAY

#
# Test
#
__C.TEST = edict()
__C.TEST.METRIC_NAME = 'ChamferDistance'