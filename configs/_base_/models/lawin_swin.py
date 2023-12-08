# model settings
norm_cfg = dict(type='BN', requires_grad=True)   #BN replace SyncBN
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        # type='SwinTransformer',
        # type='SwinTransformer_Acmix',
        type='Spatial_Specific_Transformer', #SwinTransformer_Acmix_V5 ours
        # type='SwinTransformer_Acmix_V2',
        # type='SwinTransformer_V2',
        # type='SwinTV2_Acmix',
        # type='GGSwinT_Acmix',
        # type='GGSwinTransformer',
        # type='Split_Swin',
        # type='GGsplit_Swin',
        # type='LePE_Amix',
        # type='Swin_Amix',
        # type='SplitSwinT_Amix',
        # type='LePE_SwinT_Acmix',
        # type='NoPE_SwinT_Acmix',
        # type='LePE_SwinT',
        # type='LocalViT_Swin',
        # type='Local_SwinT_Acmix',
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
        # type='UPerHead',
        type='LargewindowHead',  #LawinHead_V9 ours
        # type='LawinHead_V10',
        # type='SegFormerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        embed_dim=768, use_scale=True, reduction=2,
        channels=512,
        dropout_ratio=0.1,
        num_classes=6,   #19
        norm_cfg=norm_cfg,
        align_corners=False,
        # loss_decode=[dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        #              dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.2)]),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # auxiliary_head=dict(
    #     type='FCNHead',
    #     in_channels=384,
    #     in_index=2,
    #     channels=256,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=6,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     # loss_decode=[dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
    #     #              dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.2)]),
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
