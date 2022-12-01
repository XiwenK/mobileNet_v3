_base_ = [
    '../_base_/models/mobilenet_v2_cifar100.py',
    '../_base_/datasets/cifar100_bs128.py',
    '../_base_/schedules/cifar100_bs128.py',
    '../_base_/default_runtime.py'
]

lr_config = dict(policy='step', step=[100, 150, 180])