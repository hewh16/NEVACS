model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SeqClif2Fc1CeItout',
        timestep=20,
        c0=1,
        h0=120,
        w0=120,
        nclass=2,
        cmode='analog',
        amode='mean',
        noise=0,
        soma_params='all_share',
        neuron='lif',
        neuron_config=None),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, loss_weight=1.0),
        topk=(1, 2),
        cal_acc=True))
dataset_type = 'CellIFC_cellball'
train_pipeline = [
    dict(type='ToTensorType', keys=['img'], dtype='float32'),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='ToTensorType', keys=['img'], dtype='float32'),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type='CellIFC_cellball',
        data_prefix='data/cell-dataset/train',
        pipeline=[
            dict(type='ToTensorType', keys=['img'], dtype='float32'),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='CellIFC_cellball',
        data_prefix='data/cell-dataset/test',
        pipeline=[
            dict(type='ToTensorType', keys=['img'], dtype='float32'),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True),
    test=dict(
        type='CellIFC_cellball',
        data_prefix='data/cell-dataset/test',
        pipeline=[
            dict(type='ToTensorType', keys=['img'], dtype='float32'),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True))
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    step=[10, 16],
    warmup='linear',
    warmup_ratio=0.01,
    warmup_iters=500)
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=1)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
fp16 = dict(loss_scale=512.0)
lynxi_devices = [[0], [1]]
work_dir = './work_dirs/clif2fc1ce_itout-b16x1-celldataset'
gpu_ids = range(0, 1)
