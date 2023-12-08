# model settings
norm_cfg = dict(type='BN', requires_grad=True)   #BN replace SyncBN
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        # type='SwinTransformer_Acmix_V5',
        # type='SwinTransformer_V2',
        # type='SwinTV2_Acmix',
        embed_dim=96,
        # depths=[2, 2, 6, 2],
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        #swinv2meiyou qk_scale
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False,
        # #swinv2 you pretrained_window_sizes
        # pretrained_window_sizes=[0, 0, 0, 0]
        ),
    # neck=dict(
    #     type='FPN',
    #     in_channels=[256, 512, 1024, 2048],
    #     out_channels=256,
    #     num_outs=4),
    decode_head=dict(
        type='FPNHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=512,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
