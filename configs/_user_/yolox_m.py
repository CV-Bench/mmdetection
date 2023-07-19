_base_ = '../yolox/yolox_s_8x8_300e_coco.py'

# model settings
model = dict(
    backbone=dict(deepen_factor=0.67, widen_factor=0.75),
    neck=dict(in_channels=[192, 384, 768], out_channels=192, num_csp_blocks=2),
    bbox_head=dict(num_classes=1, in_channels=192, feat_channels=192),
)

dataset_type = 'COCODataset'
classes = ('object',)

data = dict(
    workers_per_gpu=0,
    samples_per_gpu=4,
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
        ann_file='/data/input/val/annotation_coco.json')
)

work_dir = '/data/output'
runner = dict(type='EpochBasedRunner', max_epochs=5)


load_from = '/checkpoints/yolox_m.pth'