# model settings
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='IMTRv21_5',
        style='pytorch'),
decode_head=dict(
        type='UPerHead',
        # type='LargewindowHead', ##ours head
        # type='LawinHead',  ###lawin orignal_model
        # type='LawinHead_V8',
        # type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        ## LawinHead没有pool_scales
        # pool_scales=(1, 2, 3, 6),
        channels=128,
        dropout_ratio=0.1,
        num_classes=6,   #19
        norm_cfg=norm_cfg,
        align_corners=False,
        # loss_decode=[dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        #              dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.5)]),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),

# decode_head=dict(
#
#         type='LargewindowHead',  #LawinHead_V9 ours
#
#         in_channels=[64, 128, 320, 512],
#         in_index=[0, 1, 2, 3],
#         embed_dim=768, use_scale=True, reduction=2,
#         channels=128,
#         dropout_ratio=0.1,
#         num_classes=6,   #19
#         norm_cfg=norm_cfg,
#         align_corners=False,
#
#         loss_decode=dict(
#             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # decode_head=dict(
    #     type='SegFormerHeadN',
    #     in_channels=[64, 128, 320, 512],
    #     in_index=[0, 1, 2, 3],
    #     feature_strides=[4, 8, 16, 32],
    #     channels=128,
    #     dropout_ratio=0.1,
    #     num_classes=19,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     decoder_params=dict(),
    #     loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
