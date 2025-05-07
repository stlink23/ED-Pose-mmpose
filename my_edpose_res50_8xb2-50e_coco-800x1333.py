
# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base
with read_base():
    from mmpose.configs._base_.default_runtime import *  # noqa

from mmcv.transforms import RandomChoice, RandomChoiceResize
from mmengine.dataset import DefaultSampler
from mmengine.model import PretrainedInit
from mmengine.optim import LinearLR, MultiStepLR
from torch.nn import GroupNorm
from torch.optim import Adam

from mmpose.codecs import EDPoseLabel
from mmpose.datasets import (BottomupRandomChoiceResize, BottomupRandomCrop,
                             CocoDataset, LoadImage, PackPoseInputs,
                             RandomFlip)
from mmpose.evaluation import CocoMetric
from mmpose.models import (BottomupPoseEstimator, ChannelMapper, EDPoseHead,
                           PoseDataPreprocessor, ResNet)
from mmpose.models.utils import FrozenBatchNorm2d
from mmpose.engine.hooks import PoseVisualizationHook
from mmpose.datasets.transforms import FilterAnnotations 
# runtime
train_cfg.update(max_epochs=50, val_interval=10)  # noqa

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',#使用混合精度训练
    # dtype='bfloat16',  # 可用值： ('float16', 'bfloat16', None)
    optimizer=dict(
    type=Adam,
    lr=0.0005,
))

# learning policy
param_scheduler = [
    dict(type=LinearLR, begin=0, end=500, start_factor=0.01,
         by_epoch=False),  # warm-up
    dict(
        type=MultiStepLR,
        begin=0,
        end=140,
        milestones=[33, 45],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=80)

# hooks
default_hooks.update(  # noqa
    checkpoint=dict(
        save_best='coco/AP', rule='greater',interval=1,max_keep_ckpts=3,
        out_dir=""),
    visualization=dict(
        type='mmpose.engine.hooks.visualization_hook.PoseVisualizationHook',
        enable=True,
        interval=1,
        out_dir='',
        show=False))

# codec settings
codec = dict(type=EDPoseLabel, num_select=50, num_keypoints=17)

# model settings
model = dict(
    type=BottomupPoseEstimator,
    data_preprocessor=dict(
        type=PoseDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type=ResNet,
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type=FrozenBatchNorm2d, requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            #type=PretrainedInit, checkpoint='torchvision://resnet50')),
            type=PretrainedInit, checkpoint='edpose_res50_coco_3rdparty.pth')),
    neck=dict(
        type=ChannelMapper,
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type=GroupNorm, num_groups=32),
        num_outs=4),
    head=dict(
        type=EDPoseHead,
        num_queries=900,
        num_feature_levels=4,
        num_keypoints=17,
        num_classes=2,
        dn_labelbook_size = 100,#标签字典大小
        as_two_stage=True,
        encoder=dict(
            num_layers=6,
            layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                self_attn_cfg=dict(  # MultiScaleDeformableAttention
                    embed_dims=256,
                    num_heads=8,
                    num_levels=4,
                    num_points=4,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.0))),
        decoder=dict(
            num_layers=6,
            embed_dims=256,
            layer_cfg=dict(  # DeformableDetrTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiScaleDeformableAttention
                    embed_dims=256,
                    batch_first=True),
                ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.1)
            ),
            query_dim=4,
            num_feature_levels=4,
            num_group=100,
            num_dn=100,
            num_box_decoder_layers=2,
            return_intermediate=True),
        out_head=dict(num_classes=2),
        positional_encoding=dict(
            num_pos_feats=128,
            temperatureH=20,
            temperatureW=20,
            normalize=True),
        denosing_cfg=dict(
            dn_box_noise_scale=0.4,
            dn_label_noise_ratio=0.5,
            dn_labelbook_size=100,
            dn_attn_mask_type_list=['match2dn', 'dn2dn', 'group2group']),
        loss_cfg=dict(
            matcher=dict(
                set_cost_class=2.0,
                cost_bbox = 5.0,
                cost_giou=2.0,
                focal_alpha = 0.25,
                cost_keypoints=10.0,
                cost_kpvis=0.0,
                cost_oks=4.0,
                num_body_points=17),
            weight_dict = dict(
                loss_ce = 2.0,
                loss_bbox= 5.0,
                loss_keypoints=10.0,
                loss_oks= 4.0,
                loss_giou=2.0)),       
        data_decoder=codec),
        
    test_cfg=dict(Pmultiscale_test=False, flip_test=False, num_select=50),
    train_cfg=dict())

# enable DDP training when rescore net is used
find_unused_parameters = True

# base dataset settings
dataset_type = CocoDataset
data_mode = 'bottomup'
data_root = ''

# pipelines
train_pipeline = [
    dict(type=LoadImage),
    #dict(type='mmpose.datasets.transforms.common_transforms.FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False, min_kpt_vis=3),
    #dict(type=RandomFlip, direction='horizontal',prob=0.5),
    dict(
        type=RandomChoice,
        transforms=[
            [
                dict(
                    type=BottomupRandomChoiceResize,
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            # [
            #     dict(
            #         type=BottomupRandomChoiceResize,
            #         # The radio of all image in train dataset < 7
            #         # follow the original implement
            #         scales=[(400, 4200), (500, 4200), (600, 4200)],
            #         keep_ratio=True),
            #     dict(
            #         type=BottomupRandomCrop,
            #         crop_type='absolute_range',
            #         crop_size=(384, 600),
            #         allow_negative_crop=True),
            #     dict(
            #         type=BottomupRandomChoiceResize,
            #         scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
            #                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
            #                 (736, 1333), (768, 1333), (800, 1333)],
            #         keep_ratio=True)
            # ]
        ]),
    dict(type=PackPoseInputs),
    #mmpose.datasets.transforms.common_transforms.GetBBoxCenterScale(padding: float = 1.25)
]

val_pipeline = [
    dict(type=LoadImage),
    dict(
        type=BottomupRandomChoiceResize,
        scales=[(800, 1333)],
        keep_ratio=True,
        backend='pillow'),
    dict(
        type=PackPoseInputs,
        meta_keys=('id', 'img_id', 'img_path', 'crowd_index', 'ori_shape',
                   'img_shape', 'input_size', 'input_center', 'input_scale',
                   'flip', 'flip_direction', 'flip_indices', 'raw_ann_info',
                   'skeleton_links'))
]

# data loaders
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
        # filter_cfg=dict(filter_empty_gt=True, min_keypoints=3),
    ))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type=CocoMetric,
    nms_mode='none',
    score_mode='keypoint',
)
test_evaluator = val_evaluator
