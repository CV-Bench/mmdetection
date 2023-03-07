_base_ = '../yolo/yolov3_d53_mstrain-608_273e_coco.py'

# model settings
model = dict(
    bbox_head=dict(
        num_classes=1,
    )
)

# dataset settings
dataset_type = 'COCODataset'
classes = ('object',)

metainfo = {
    'CLASSES': ('object', ),
    'PALETTE': [
        (220, 20, 60),
    ]
}

data = dict(
    workers_per_gpu=0,
    samples_per_gpu=8,
    train=dict(
        img_prefix='/data/input/train/',
        classes=classes,
        ann_file='/data/input/train/annotation_coco.json'),
    val=dict(
        img_prefix='/data/input/val/',
        classes=classes,
        ann_file='/data/input/val/annotation_coco.json'),
    test=dict(
        img_prefix='/data/input/val/',
        classes=classes,
        ann_file='/data/input/val/annotation_coco.json'),
)


work_dir = '/data/output'


optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0005)