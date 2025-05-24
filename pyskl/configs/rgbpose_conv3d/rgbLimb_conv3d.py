# model_cfg
backbone_cfg = dict(
    type='RGBPoseConv3D',
    speed_ratio=4,
    channel_ratio=4,
    rgb_pathway=dict(
        num_stages=4,
        lateral=True,
        lateral_infl=1,
        lateral_activate=[0, 0, 1, 1],
        base_channels=64,
        conv1_kernel=(1, 7, 7),
        inflate=(0, 0, 1, 1)),
    pose_pathway=dict(
        num_stages=3,
        stage_blocks=(4, 6, 3),
        lateral=True,
        lateral_inv=True,
        lateral_infl=16,
        lateral_activate=(0, 1, 1),
        in_channels=36,
        base_channels=32,
        out_indices=(2, ),
        conv1_kernel=(1, 7, 7),
        conv1_stride=(1, 1),
        pool1_stride=(1, 1),
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 1)))
head_cfg = dict(
    type='RGBPoseHead',
    num_classes=32,
    in_channels=[2048, 512],
    loss_components=['rgb', 'pose'],
    loss_weights=[1., 1.])
test_cfg = dict(average_clips='prob')
model = dict(
    type='MMRecognizer3D',
    backbone=backbone_cfg,
    cls_head=head_cfg,
    test_cfg=test_cfg)

dataset_type = 'PoseDataset'
data_root = '../dataset/RGB/clips'
ann_file = '../dataset/Skeleton/iMiGUE_36.pkl'  # 标注文件路径
left_kp = [17, 8, 10, 12, 13, 20, 2, 3, 4, 30, 31, 32, 33, 34, 35]
right_kp = [18, 9, 11, 16, 15, 22, 5, 6, 7, 24, 25, 26, 27, 28, 29]

img_norm_cfg = dict(
    mean=[66.15693065312055, 59.39387486980162, 67.08884259986877], 
    std=[62.46999995594521, 57.96686051956861, 58.18312924970111], 
    to_bgr=False
)

train_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8, Pose=32), num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', sigma=0.7, use_score=True, with_kp=True, with_limb=False, scaling=0.25),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'heatmap_imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'heatmap_imgs', 'label'])
]
val_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8, Pose=32), num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='GeneratePoseTarget', sigma=0.7, use_score=True, with_kp=True, with_limb=False, scaling=0.25),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'heatmap_imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'heatmap_imgs', 'label'])
]
test_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8, Pose=32), num_clips=10),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='GeneratePoseTarget', sigma=0.7, use_score=True, with_kp=True, with_limb=False, scaling=0.25),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'heatmap_imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'heatmap_imgs', 'label'])
]

data = dict(
    videos_per_gpu=6,
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(type=dataset_type, ann_file=ann_file, split='train', data_prefix=data_root, pipeline=train_pipeline),
    val=dict(type=dataset_type, ann_file=ann_file, split='val', data_prefix=data_root, pipeline=val_pipeline),
    test=dict(type=dataset_type, ann_file=ann_file, split='test', data_prefix=data_root, pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.0075, momentum=0.9, weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[12, 16])
total_epochs = 20
checkpoint_config = dict(interval=1)
workflow = [('train', 1)]
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5), key_indicator='RGBPose_1:1_top1_acc')
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
work_dir = './work_dirs/rgbpose_conv3d/rgbLimb_conv3d'
load_from = '/root/autodl-tmp/pyskl/configs/rgbpose_conv3d/rgbLimb_conv3d_init.pth'
