# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='MobileNetV3Cifar', arch='small'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='StackedLinearClsHeadWithPred',
        num_classes=100,
        in_channels=576,
        mid_channels=[1280],
        act_cfg=dict(type='HSwish'),
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
