_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'


# Change the number of classes
model = dict(
    bbox_head=dict(num_classes=1)
)
# model = dict(
#     roi_head=dict(
#         #mask_head=dict(type='FCNMaskHead', num_classes=1),
#         bbox_head=dict(num_classes=1)
#     )
# )

# Modify dataset related settings
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
    samples_per_gpu=1,
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