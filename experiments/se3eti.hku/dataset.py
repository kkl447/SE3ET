import os.path as osp
import pickle
from typing import Any, Tuple

import numpy as np
import torch.utils.data

from geotransformer.datasets.registration.threedmatch.dataset import ThreeDMatchPairDataset
from geotransformer.utils.data import (
    build_dataloader_stack_mode,
    calibrate_neighbors_stack_mode,
    registration_collate_fn_stack_mode,
)
from geotransformer.utils.pointcloud import (
    get_rotation_translation_from_transform,
    get_transform_from_rotation_translation,
    random_sample_rotation,
)


class HKUMarsPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root: str,
        subset: str,
        metadata_dir: str = 'metadata_amtown_valtest',
        point_limit: int = None,
        use_augmentation: bool = False,
        augmentation_noise: float = 0.005,
        augmentation_rotation: float = 1.0,
    ) -> None:
        super().__init__()
        self.dataset_root = dataset_root
        self.subset = subset
        self.metadata_dir = metadata_dir
        self.point_limit = point_limit
        self.use_augmentation = use_augmentation
        self.augmentation_noise = augmentation_noise
        self.augmentation_rotation = augmentation_rotation

        metadata_path = osp.join(self.dataset_root, self.metadata_dir, f'{subset}.pkl')
        with open(metadata_path, 'rb') as f:
            self.metadata_list = pickle.load(f)

    def __len__(self) -> int:
        return len(self.metadata_list)

    def _load_point_cloud(self, file_name: str) -> np.ndarray:
        points = np.load(osp.join(self.dataset_root, file_name))
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[: self.point_limit]
            points = points[indices]
        return points

    def _augment_point_cloud(
        self,
        ref_points: np.ndarray,
        src_points: np.ndarray,
        transform: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rotation, translation = get_rotation_translation_from_transform(transform)
        aug_rotation = random_sample_rotation(self.augmentation_rotation)
        if np.random.rand() > 0.5:
            ref_points = np.matmul(ref_points, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
        else:
            src_points = np.matmul(src_points, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)

        ref_points = ref_points + (
            np.random.rand(ref_points.shape[0], 3) - 0.5
        ) * self.augmentation_noise
        src_points = src_points + (
            np.random.rand(src_points.shape[0], 3) - 0.5
        ) * self.augmentation_noise
        transform = get_transform_from_rotation_translation(rotation, translation)
        return ref_points, src_points, transform

    def __getitem__(self, index: int) -> Any:
        metadata = self.metadata_list[index]
        ref_raw_points = self._load_point_cloud(metadata['pcd0'])
        src_raw_points = self._load_point_cloud(metadata['pcd1'])

        ref_points = ref_raw_points[:, :3].astype(np.float32)
        src_points = src_raw_points[:, :3].astype(np.float32)
        transform = metadata['transform'].astype(np.float32)

        if self.use_augmentation:
            ref_points, src_points, transform = self._augment_point_cloud(
                ref_points,
                src_points,
                transform,
            )

        data_dict = {
            'index': index,
            'scene_name': metadata.get('seq', str(metadata.get('seq_id', 0))),
            'ref_frame': metadata['frame0'],
            'src_frame': metadata['frame1'],
            'overlap': metadata['overlap'],
            'ref_points': ref_points.astype(np.float32),
            'src_points': src_points.astype(np.float32),
            'ref_feats': np.ones((ref_points.shape[0], 1), dtype=np.float32),
            'src_feats': np.ones((src_points.shape[0], 1), dtype=np.float32),
            'transform': transform.astype(np.float32),
        }
        return data_dict


def _make_dataset(
    cfg: Any,
    subset: str,
    point_limit: int,
    use_augmentation: bool = False,
) -> torch.utils.data.Dataset:
    if cfg.data.dataset_type == 'hku_mars':
        return HKUMarsPairDataset(
            cfg.data.dataset_root,
            subset,
            metadata_dir=cfg.data.metadata_dir,
            point_limit=point_limit,
            use_augmentation=use_augmentation,
            augmentation_noise=cfg.train.augmentation_noise,
            augmentation_rotation=cfg.train.augmentation_rotation,
        )
    return ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        subset,
        point_limit=point_limit,
        use_augmentation=use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
    )


def train_valid_data_loader(cfg: Any, distributed: bool) -> Tuple[Any, Any, Any]:
    train_dataset = _make_dataset(
        cfg,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
    train_loader = build_dataloader_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
    )

    valid_dataset = _make_dataset(
        cfg,
        'val',
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
    )
    valid_loader = build_dataloader_stack_mode(
        valid_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
    )

    return train_loader, valid_loader, neighbor_limits


def test_data_loader(cfg: Any, benchmark: str) -> Tuple[Any, Any]:
    train_dataset = _make_dataset(
        cfg,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )

    test_dataset = _make_dataset(
        cfg,
        benchmark,
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
    )
    test_loader = build_dataloader_stack_mode(
        test_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
    )

    return test_loader, neighbor_limits
