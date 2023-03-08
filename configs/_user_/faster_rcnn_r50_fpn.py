# The new config inherits a base config to highlight the necessary modification
_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

# Change the number of classes
model = dict(
    roi_head=dict(
        #mask_head=dict(type='FCNMaskHead', num_classes=1),
        bbox_head=dict(num_classes=1)
    )
)

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('object',)

metainfo = {
    'CLASSES': ('object', ),
    'PALETTE': [
        (220, 20, 60),
    ]
}

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 360), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 360),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    workers_per_gpu=0,
    samples_per_gpu=4,
    train=dict(
        img_prefix='/data/input/train/',
        classes=classes,
        ann_file='/data/input/train/annotation_coco.json',
        pipeline=train_pipeline),
    val=dict(
        img_prefix='/data/input/val/',
        classes=classes,
        ann_file='/data/input/val/annotation_coco.json',
        pipeline=test_pipeline),
    test=dict(
        img_prefix='/data/input/val/',
        classes=classes,
        ann_file='/data/input/val/annotation_coco.json',
        pipeline=test_pipeline)
)

work_dir = '/data/output'
runner = dict(type='EpochBasedRunner', max_epochs=5)


load_from = '/checkpoints/faster_rcnn_r50_fpn.pth'

# log_config = dict(
#     hooks = [
#     dict(type='MMDetWandbHook',
#          init_kwargs={'project': 'mmdetection'},
#          interval=1,
#          log_checkpoint=False,
#          log_checkpoint_metadata=True,
#          num_eval_images=100,
#          bbox_score_thr=0.3)
#         ]
# )
