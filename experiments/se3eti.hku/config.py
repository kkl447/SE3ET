import argparse
import os
import os.path as osp
from typing import Any

from easydict import EasyDict as edict

from geotransformer.utils.common import ensure_dir


_C = edict()

# common
_C.seed = 7351

# dirs
_C.working_dir = osp.dirname(osp.realpath(__file__))
_C.root_dir = osp.dirname(osp.dirname(_C.working_dir))
_C.exp_name = osp.basename(_C.working_dir)
_C.output_dir = osp.join(_C.root_dir, 'output', _C.exp_name)
_C.snapshot_dir = osp.join(_C.output_dir, 'snapshots')
_C.log_dir = osp.join(_C.output_dir, 'logs')
_C.event_dir = osp.join(_C.output_dir, 'events')
_C.feature_dir = osp.join(_C.output_dir, 'features')
_C.registration_dir = osp.join(_C.output_dir, 'registration')

ensure_dir(_C.output_dir)
ensure_dir(_C.snapshot_dir)
ensure_dir(_C.log_dir)
ensure_dir(_C.event_dir)
ensure_dir(_C.feature_dir)
ensure_dir(_C.registration_dir)

# data
_C.data = edict()
_C.data.dataset_type = os.environ.get('SE3ET_DATASET_TYPE', 'hku_mars')
_C.data.dataset_root = os.environ.get(
    'HKU_RGB_DATASET_ROOT',
    osp.abspath(
        osp.join(
            _C.root_dir,
            '..',
            'GeoTransformer',
            'data',
            'HKU_MARS',
            'MARS_Dataset_v015_dynamic_s030_rgb',
        )
    ),
)
_C.data.metadata_dir = os.environ.get('HKU_METADATA_DIR', 'metadata_amtown_valtest')

# train data
_C.train = edict()
_C.train.batch_size = 1
_C.train.num_workers = 8
_C.train.point_limit = 30000
_C.train.use_augmentation = True
_C.train.augmentation_noise = 0.005
_C.train.augmentation_rotation = 1.0

# test data
_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 8
_C.test.point_limit = None

# evaluation
_C.eval = edict()
_C.eval.acceptance_overlap = 0.0
_C.eval.acceptance_radius = 1.0
_C.eval.inlier_ratio_threshold = 0.05
_C.eval.rmse_threshold = 2.0
_C.eval.rre_threshold = 5.0
_C.eval.rte_threshold = 2.0

# ransac
_C.ransac = edict()
_C.ransac.distance_threshold = 0.05
_C.ransac.num_points = 3
_C.ransac.num_iterations = 1000

# optim
_C.optim = edict()
_C.optim.lr = 1e-4
_C.optim.lr_decay = 0.95
_C.optim.lr_decay_steps = 1
_C.optim.weight_decay = 1e-6
_C.optim.max_epoch = 30
_C.optim.grad_acc_steps = 1

# model - backbone
_C.backbone = edict()
_C.backbone.num_stages = 4
_C.backbone.init_voxel_size = 0.30
_C.backbone.kernel_size = 15
_C.backbone.base_radius = 4.25
_C.backbone.base_sigma = 2.0
_C.backbone.init_radius = _C.backbone.base_radius * _C.backbone.init_voxel_size
_C.backbone.init_sigma = _C.backbone.base_sigma * _C.backbone.init_voxel_size
_C.backbone.group_norm = 32
_C.backbone.input_dim = 1
_C.backbone.init_dim = 64
_C.backbone.output_dim = 256

# epn
_C.epn = edict()
_C.epn.kanchor = 6
_C.epn.quotient_factor = 4
_C.epn.num_kernel_points = 15
_C.epn.non_sep_conv = True
_C.epn.equiv_mode_kp = True
_C.epn.fixed_kernel_points = 'center'
_C.epn.rot_by_permute = True
_C.epn.ignore_steer_constraint = False
_C.epn.num_points = _C.train.point_limit
_C.epn.epn_kernel = False
_C.epn.att_pooling = False
_C.epn.att_permute = False
_C.epn.dual_feature = False
_C.epn.ctrness_w_track = False
_C.epn.rot_head_attn = False
_C.epn.gather_by_idxing = False
_C.epn.in_points_dim = 3
_C.epn.use_batch_norm = True
_C.epn.batch_norm_momentum = 0.99
_C.epn.first_subsampling_dl = 0.02
_C.epn.conv_radius = 2.5
_C.epn.deform_radius = 5.0
_C.epn.KP_extent = 1.0
_C.epn.KP_influence = 'linear'
_C.epn.aggregation_mode = 'sum'
_C.epn.modulated = False

# model - Global
_C.model = edict()
_C.model.ground_truth_matching_radius = 0.6
_C.model.num_points_in_patch = 64
_C.model.num_sinkhorn_iterations = 100

# model - Coarse Matching
_C.coarse_matching = edict()
_C.coarse_matching.num_targets = 128
_C.coarse_matching.overlap_threshold = 0.1
_C.coarse_matching.num_correspondences = 256
_C.coarse_matching.dual_normalization = True

# model - GeoTransformer
_C.geotransformer = edict()
_C.geotransformer.input_dim = 1024
_C.geotransformer.hidden_dim = 256
_C.geotransformer.output_dim = 256
_C.geotransformer.num_heads = 4
_C.geotransformer.blocks = ['self_eq', 'cross', 'self_eq', 'cross', 'self_eq', 'cross']
_C.geotransformer.sigma_d = 4.8
_C.geotransformer.sigma_a = 15
_C.geotransformer.angle_k = 3
_C.geotransformer.supervise_rotation = False
_C.geotransformer.reduction_a = 'max'
_C.geotransformer.align_mode = '0'
_C.geotransformer.alternative_impl = False
_C.geotransformer.n_level_equiv = 0
_C.geotransformer.attn_r_positive = 'softplus'
_C.geotransformer.attn_r_positive_rot_supervise = 'minus'

# model - Fine Matching
_C.fine_matching = edict()
_C.fine_matching.topk = 3
_C.fine_matching.acceptance_radius = 1.0
_C.fine_matching.mutual = True
_C.fine_matching.confidence_threshold = 0.05
_C.fine_matching.use_dustbin = False
_C.fine_matching.use_global_score = False
_C.fine_matching.correspondence_threshold = 3
_C.fine_matching.correspondence_limit = None
_C.fine_matching.num_refinement_steps = 5

# loss - Coarse level
_C.coarse_loss = edict()
_C.coarse_loss.positive_margin = 0.1
_C.coarse_loss.negative_margin = 1.4
_C.coarse_loss.positive_optimal = 0.1
_C.coarse_loss.negative_optimal = 1.4
_C.coarse_loss.log_scale = 24
_C.coarse_loss.positive_overlap = 0.1

# loss - Fine level
_C.fine_loss = edict()
_C.fine_loss.positive_radius = 0.6

# loss - Overall
_C.loss = edict()
_C.loss.weight_coarse_loss = 1.0
_C.loss.weight_fine_loss = 1.0
_C.loss.weight_rotation_loss = 1.0


def make_cfg() -> Any:
    return _C


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--link_output',
        dest='link_output',
        action='store_true',
        help='link output dir',
    )
    return parser.parse_args()


def main() -> None:
    cfg = make_cfg()
    args = parse_args()
    if args.link_output:
        os.symlink(cfg.output_dir, 'output')


if __name__ == '__main__':
    main()
