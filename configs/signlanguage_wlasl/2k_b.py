import datetime
modality = 'b'
graph = 'sign_language'
work_dir = f'./work_dirs/signlanguage_wlasl/2k_bone_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
load_from = ""

drop_kp = [i for i in range(77)] 
num_drop_node = len(drop_kp)

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='SignReGraph',
        in_channels=2,
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
        graph_cfg=dict(layout=graph, mode='spatial', num_filter=8, init_off=.04, init_std=.02),
        num_drop_node = num_drop_node),
    cls_head=dict(type='SimpleHead', joint_cfg='sign_language', num_classes=2000, in_channels=384, weight=1.0))

dataset_type = 'PoseDataset'
ann_file = "./data/signlanguage_wlasl/2000wlasl_focus_hand_w_face.pkl"

left_hand_kp = [0, 2, 4] + [i for i in range(91 - 85, 112 - 85 )]
right_hand_kp = [1, 3, 5] + [i for i in range(112 - 85, 133 - 85)]

train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='Horizontal_Flip_Keypoint', flip_ratio=0.2),
    dict(type='Drop_Keypoint_Sign_Language', drop_area=drop_kp),
    dict(type='Part_Drop_Sign_Language',p=0.1, left_hand=left_hand_kp, right_hand=right_hand_kp),
    dict(type='Frames_Drop', drop_ratio=0.3, max_frames_drop=10),
    dict(type='GenSkeFeat', dataset='sign_language', feats=[modality]),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label','drop_area'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='GenSkeFeat',dataset = 'sign_language', feats=[modality]),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=100, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='GenSkeFeat',dataset = 'sign_language', feats=[modality]),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])    
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='train'),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='test'))

optimizer = dict(type='SGD', lr=5e-4, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 30
checkpoint_config = dict(interval=10)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook'),dict(type='TensorboardLoggerHook')])
train_dataloader = dict(
    worker_init_fn='mmengine.dataset.utils.worker_init_fn'
)
val_dataloader = dict(
    worker_init_fn='mmengine.dataset.utils.worker_init_fn'
)
test_dataloader = dict(
    worker_init_fn='mmengine.dataset.utils.worker_init_fn'
)