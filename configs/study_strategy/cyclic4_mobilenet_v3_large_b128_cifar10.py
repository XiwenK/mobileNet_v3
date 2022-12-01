_base_ = [
    '../_base_/models/mobilenet_v3_large_cifar10.py',
    '../_base_/datasets/cifar10_bs128.py',
    '../_base_/schedules/cifar10_bs128_lr_cyclic4.py', '../_base_/default_runtime.py'
]

runner = dict(type='EpochBasedRunner', max_epochs=200)
