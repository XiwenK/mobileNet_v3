_base_ = [
    '../_base_/models/mobilenet_v3_large_cifar100.py',
    '../_base_/datasets/cifar100_bs128.py',
    '../_base_/schedules/cifar100_bs128_lr_poly_warmup5.py', '../_base_/default_runtime.py'
]

runner = dict(type='EpochBasedRunner', max_epochs=200)

model = dict(
    type='BSConvClassifier',
    backbone = dict(
        conv_cfg=dict(type='BSConvS')
    ),
    train_cfg=dict(augments=[
        dict(type='BatchCutMix', alpha=1.0, prob=0.5, num_classes=100),
]))