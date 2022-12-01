_base_ = [
    '../_base_/models/mobilenet_v3_large_cifar10.py',
    '../_base_/datasets/cifar10_bs128.py',
    '../_base_/schedules/cifar10_bs128.py', '../_base_/default_runtime.py'
]

lr_config = dict(policy='step', step=[100, 150, 180])
runner = dict(type='EpochBasedRunner', max_epochs=200)

model = dict(
    type='ImageClassifier',
    backbone=dict(type='MobileNetV3Cifar', arch='large', with_nam=True)
)