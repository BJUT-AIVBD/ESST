# model settings
norm_cfg = dict(type='BN', requires_grad=True)   #BN replace SyncBN
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        # type='Spatial_Specific_Involutionpath',
        # type='SwinTransformer_Acmix',
        # type='Spatial_Specific_Transformer', #SwinTransformer_Acmix_V5 ours
        # type='SwinTransformer_V2',
        # type='GGSwinTransformer',
        # type='LocalViT_Swin',
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
    decode_head=dict(
        type='UPerHead',
        # type='LargewindowHead', ##ours head
        # type='LawinHead',  ###lawin orignal_model
        # type='LawinHead_V8',
        # type='SegFormerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        ## LawinHead没有pool_scales
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=6,   #19
        norm_cfg=norm_cfg,
        align_corners=False,
        # loss_decode=[dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        #              dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.5)]),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        # loss_decode=[dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        #              dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.5)]),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
