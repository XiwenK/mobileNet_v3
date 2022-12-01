# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineRestart', min_lr_ratio=1e-2, periods=[25, 50, 100, 200], restart_weights=[1, 1, 1, 1])
runner = dict(type='EpochBasedRunner', max_epochs=200)
