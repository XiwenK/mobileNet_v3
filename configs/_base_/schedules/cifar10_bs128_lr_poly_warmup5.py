# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='poly', power=0.7, min_lr=1e-4, by_epoch=False, warmup='exp', warmup_ratio=0.1, warmup_iters=5, warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=200)
