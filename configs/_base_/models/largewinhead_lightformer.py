# model settings
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='../pretrained/pvt_v2_b2.pth',
    backbone=dict(
        type='pvt_v2_b2',
        style='pytorch'),
    decode_head=dict(
        type='LargewindowHead',  # LawinHead_V9 ours
        # type='',   #lawinaspp
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        embed_dim=768, use_scale=True, reduction=2,
        channels=128,
        dropout_ratio=0.1,
        num_classes=6,  # 19
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
