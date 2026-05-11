import argparse
import os.path as osp
import time
from typing import Any

import numpy as np

from geotransformer.engine import SingleTester
from geotransformer.utils.common import ensure_dir, get_log_string
from geotransformer.utils.torch import release_cuda

from config import make_cfg
from dataset import test_data_loader
from loss import Evaluator
from model import create_model


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--benchmark',
        choices=['3DMatch', '3DLoMatch', 'train', 'val', 'test'],
        help='test benchmark or HKU split',
    )
    return parser


class Tester(SingleTester):
    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg, parser=make_parser())

        start_time = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg, self.args.benchmark)
        loading_time = time.time() - start_time
        self.logger.info(f'Data loader created: {loading_time:.3f}s collapsed.')
        self.logger.info(f'Calibrate neighbors: {neighbor_limits}.')
        self.register_loader(data_loader)

        model = create_model(cfg).cuda()
        self.register_model(model)

        self.evaluator = Evaluator(cfg).cuda()
        self.output_dir = osp.join(cfg.feature_dir, self.args.benchmark)
        ensure_dir(self.output_dir)

    def test_step(self, iteration: int, data_dict: Any) -> Any:
        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration: int, data_dict: Any, output_dict: Any) -> Any:
        result_dict = self.evaluator(output_dict, data_dict)
        return result_dict

    def summary_string(
        self,
        iteration: int,
        data_dict: Any,
        output_dict: Any,
        result_dict: Any,
    ) -> str:
        scene_name = data_dict['scene_name']
        ref_frame = data_dict['ref_frame']
        src_frame = data_dict['src_frame']
        message = f'{scene_name}, id0: {ref_frame}, id1: {src_frame}'
        message += ', ' + get_log_string(result_dict=result_dict)
        message += ', nCorr: {}'.format(output_dict['corr_scores'].shape[0])
        return message

    def after_test_step(
        self,
        iteration: int,
        data_dict: Any,
        output_dict: Any,
        result_dict: Any,
    ) -> None:
        scene_name = data_dict['scene_name']
        ref_id = data_dict['ref_frame']
        src_id = data_dict['src_frame']

        ensure_dir(osp.join(self.output_dir, scene_name))
        file_name = osp.join(self.output_dir, scene_name, f'{ref_id}_{src_id}.npz')
        np.savez_compressed(
            file_name,
            ref_points=release_cuda(output_dict['ref_points']),
            src_points=release_cuda(output_dict['src_points']),
            ref_points_f=release_cuda(output_dict['ref_points_f']),
            src_points_f=release_cuda(output_dict['src_points_f']),
            ref_points_c=release_cuda(output_dict['ref_points_c']),
            src_points_c=release_cuda(output_dict['src_points_c']),
            ref_feats_c=release_cuda(output_dict['ref_feats_c']),
            src_feats_c=release_cuda(output_dict['src_feats_c']),
            ref_node_corr_indices=release_cuda(output_dict['ref_node_corr_indices']),
            src_node_corr_indices=release_cuda(output_dict['src_node_corr_indices']),
            ref_corr_points=release_cuda(output_dict['ref_corr_points']),
            src_corr_points=release_cuda(output_dict['src_corr_points']),
            corr_scores=release_cuda(output_dict['corr_scores']),
            gt_node_corr_indices=release_cuda(output_dict['gt_node_corr_indices']),
            gt_node_corr_overlaps=release_cuda(output_dict['gt_node_corr_overlaps']),
            estimated_transform=release_cuda(output_dict['estimated_transform']),
            transform=release_cuda(data_dict['transform']),
            overlap=data_dict['overlap'],
        )


def main() -> None:
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()


if __name__ == '__main__':
    main()
