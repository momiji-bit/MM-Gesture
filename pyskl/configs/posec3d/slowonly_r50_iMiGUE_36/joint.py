model = dict(
    type='Recognizer3D',  # 模型类型：使用 3D 动作识别器
    backbone=dict(  # 主干网络配置
        type='ResNet3dSlowOnly',  # 使用 SlowOnly 版本的 3D ResNet
        in_channels=36,  # 输入通道数，对应关键点的热图通道数（36 关键点）
        base_channels=32,  # 网络基础通道数
        num_stages=3,  # 网络阶段数
        out_indices=(2, ),  # 输出第 3 阶段的特征
        stage_blocks=(4, 6, 3),  # 每个阶段的残差块数
        conv1_stride=(1, 1),  # 第一层卷积（空间、时间）步长
        pool1_stride=(1, 1),  # 第一层池化（空间、时间）步长
        inflate=(0, 1, 1),  # 是否在各阶段进行时间维度扩张（inflate）
        spatial_strides=(2, 2, 2),  # 每个阶段空间步长
        temporal_strides=(1, 1, 2)  # 每个阶段时间步长
    ),
    cls_head=dict(  # 分类头配置
        type='I3DHead',  # 使用 I3DHead 进行分类
        in_channels=512,  # 输入通道数，与 backbone 输出通道一致
        num_classes=32,  # 分类类别数
        dropout=0.5  # Dropout 比例
    ),
    test_cfg=dict(average_clips='prob')  # 测试时对多个 clip 的分类概率取平均
)

dataset_type = 'PoseDataset'  # 数据集类型：姿态数据集
ann_file = '../dataset/Skeleton/iMiGUE_36.pkl'  # 标注文件路径

# 左右关键点索引，用于翻转时对调
left_kp = [17, 8, 10, 12, 13, 20, 2, 3, 4, 30, 31, 32, 33, 34, 35]
right_kp = [18, 9, 11, 16, 15, 22, 5, 6, 7, 24, 25, 26, 27, 28, 29]

# 训练数据预处理流水线
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),  # 均匀采样 48 帧
    dict(type='PoseDecode'),  # 解码关键点为热图或骨架形式
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),  # 紧凑裁剪包围盒，保证长宽比
    dict(type='Resize', scale=(-1, 64)),  # 按高度 64 等比例缩放宽度
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),  # 随机裁剪一定面积后再缩放
    dict(type='Resize', scale=(56, 56), keep_ratio=False),  # 强制缩放为 56x56
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),  # 随机水平翻转，并交换左右关键点
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),  # 生成关键点热图，不生成骨骼连接
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),  # 格式化为 NCTHW 热图格式
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),  # 收集输入和标签
    dict(type='ToTensor', keys=['imgs', 'label'])  # 转为张量
]

# 验证数据预处理流水线
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1),  # 采样 1 个 clip
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),  # 直接缩放为 64x64
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

# 测试数据预处理流水线
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=10),  # 采样 10 个 clip
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False, double=True,
         left_kp=left_kp, right_kp=right_kp),  # 双流测试时左右关键点都考虑
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=32,  # 每块 GPU 上的 batch size
    workers_per_gpu=4,  # 每块 GPU 上的数据加载线程数
    test_dataloader=dict(videos_per_gpu=1),  # 测试时设置 batch size=1
    train=dict(
        type='RepeatDataset',  # 重复数据集以增强样本
        times=10,  # 重复 10 次
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            split='train',  # 训练集划分
            pipeline=train_pipeline
        )
    ),
    val=dict(type=dataset_type, ann_file=ann_file, split='val', pipeline=val_pipeline),  # 验证集
    test=dict(type=dataset_type, ann_file=ann_file, split='test', pipeline=test_pipeline)  # 测试集（使用验证集数据）
)

# 优化器配置
optimizer = dict(
    type='SGD',  # 随机梯度下降
    lr=0.05,  # 学习率（针对 8 块 GPU 的总 lr）
    momentum=0.9,  # 动量
    weight_decay=0.0003  # 权重衰减
)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))  # 梯度裁剪，防止梯度爆炸

# 学习率调度
lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)  # 余弦退火策略

# 训练总轮数
total_epochs = 24

# 检查点保存配置
checkpoint_config = dict(interval=1)  # 每训练 1 个 epoch 保存一次

# 评估指标
evaluation = dict(
    interval=1,  # 每个 epoch 评估一次
    metrics=['top_k_accuracy', 'mean_class_accuracy'],  # 计算 Top-k 准确率 & 平均分类准确率
    topk=(1, 5)  # Top-1 和 Top-5
)

# 日志配置
log_config = dict(
    interval=20,  # 每 20 个迭代打印一次日志
    hooks=[dict(type='TextLoggerHook')]  # 文本日志钩子
)
log_level = 'INFO'  # 日志等级

# 工作目录，用于保存模型权重和日志
work_dir = './work_dirs/posec3d/slowonly_r50_iMiGUE_36/joint'
