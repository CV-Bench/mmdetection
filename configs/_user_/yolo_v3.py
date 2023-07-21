import json

_base_ = '../yolo/yolov3_d53_mstrain-608_273e_coco.py'

val_coco = '/mmdetection/data/input/val/annotation_coco.json'
train_coco = '/mmdetection/data/input/train/annotation_coco.json'

def preprocess(val):
    return tuple("TEST")
    with open(val) as f:
        val = json.load(f)

    classes = tuple((cat["name"] for cat in val["categories"]))
    return classes

# dataset settings
dataset_type = 'COCODataset'
classes = preprocess(val=val_coco)

# model settings
model = dict(
    bbox_head=dict(
        num_classes=len(classes),
    )
)



metainfo = {
    'classes': classes,
    'palette': [
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
    samples_per_gpu=8,
    train=dict(
        img_prefix='/mmdetection/data/input/train/',
        classes=classes,
        ann_file=train_coco,
        pipeline=train_pipeline),
    val=dict(
        img_prefix='/mmdetection/data/input/val/',
        classes=classes,
        ann_file=val_coco,
        pipeline=test_pipeline),
    test=dict(
        img_prefix='/mmdetection/data/input/val/',
        classes=classes,
        ann_file=val_coco,
        pipeline=test_pipeline)
)

work_dir = '/mmdetection/data/output'

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
grad_clip=dict(max_norm=35, norm_type=2)

runner = dict(type='EpochBasedRunner', max_epochs=5)

load_from = '../yolo/yolov3_d53_mstrain-608_273e_coco.pth'