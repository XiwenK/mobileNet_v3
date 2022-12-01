# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(policy='cyclic', cyclic_times=4, target_ratio=(20, 1e-4))

runner = dict(type='EpochBasedRunner', max_epochs=200)
