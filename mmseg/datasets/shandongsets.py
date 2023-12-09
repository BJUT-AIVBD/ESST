from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ShanDongDataset(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = (
        'surfaces', 'building', 'low vegetation', 'tree', 'car', 'cluster')

    PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
    ###### palette[0], palette[1], palette[2] = 255, 255, 255  # 不透水面
    # palette[3], palette[4], palette[5] = 0, 0, 255  # 建筑物
    # palette[6], palette[7], palette[8] = 0, 255, 255  # 低植被
    # palette[9], palette[10], palette[11] = 0, 255, 0  # 树木
    # palette[12], palette[13], palette[14] = 255, 255, 0  # 汽车
    # palette[15], palette[16], palette[17] = 255, 0, 0  # 背景


    def __init__(self, **kwargs):
        super(ShanDongDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)
