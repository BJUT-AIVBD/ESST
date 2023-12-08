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
        type='MlPHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
