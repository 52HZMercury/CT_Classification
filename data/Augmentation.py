# import yaml
# from monai.transforms import (
#     AsDiscrete,
#     EnsureChannelFirstd,
#     Compose,
#     CropForegroundd,
#     LoadImaged,
#     Orientationd,
#     RandFlipd,
#     RandCropByPosNegLabeld,
#     RandShiftIntensityd,
#     ScaleIntensityRanged,
#     Spacingd,
#     RandRotate90d,
#     Rand3DElasticd,
#     ResizeWithPadOrCropd,
#     AdjustContrastd,
#     Lambdad
# )
#
# config_path = "config/config.yaml"
# with open(config_path, 'r', encoding='utf-8') as f:
#     config = yaml.safe_load(f)
#
#
# # 单标签的
# train_transforms = Compose(
#     [
#         LoadImaged(keys=["image", "label"], image_only=False),
#         EnsureChannelFirstd(keys=["image", "label"]),
#         Orientationd(keys=["image", "label"], axcodes="RAS"),
#         Spacingd(
#             keys=["image", "label"],
#             pixdim=config['transforms']['spacing']['pixdim'],
#             mode=(config['transforms']['spacing']['mode'], "nearest"),
#         ),
#         ScaleIntensityRanged(
#             keys=["image"],
#             a_min=config['transforms']['scale_intensity']['a_min'],
#             a_max=config['transforms']['scale_intensity']['a_max'],
#             b_min=config['transforms']['scale_intensity']['b_min'],
#             b_max=config['transforms']['scale_intensity']['b_max'],
#             clip=config['transforms']['scale_intensity']['clip'],
#         ),
#         CropForegroundd(keys=["image", "label"], source_key="label",
#                         margin=config['transforms']['crop_foreground']['margin']),
#         ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=config['transforms']['resize']['spatial_size'],
#                              mode=config['transforms']['resize']['mode']),
#         RandCropByPosNegLabeld(
#             keys=["image", "label"],
#             label_key="label",
#             spatial_size=config['transforms']['rand_crop']['spatial_size'],
#             pos=config['transforms']['rand_crop']['pos'],
#             neg=config['transforms']['rand_crop']['neg'],
#             num_samples=config['transforms']['rand_crop']['num_samples'],
#             image_key="image",
#             image_threshold=0,
#         ),
#         RandFlipd(
#             keys=["image", "label"],
#             spatial_axis=[0],
#             prob=0.5,
#         ),
#         RandFlipd(
#             keys=["image", "label"],
#             spatial_axis=[1],
#             prob=0.5,
#         ),
#         RandFlipd(
#             keys=["image", "label"],
#             spatial_axis=[2],
#             prob=0.5,
#         ),
#         RandRotate90d(
#             keys=["image", "label"],
#             prob=0.5,
#             max_k=4,
#         ),
#         RandShiftIntensityd(
#             keys=["image"],
#             offsets=config['transforms']['rand_shift_intensity']['offsets'],
#             prob=config['transforms']['rand_shift_intensity']['prob'],
#         ),
#     ]
# )
#
# val_transforms = Compose(
#     [
#         LoadImaged(keys=["image", "label"], image_only=False),
#         EnsureChannelFirstd(keys=["image", "label"]),
#         Orientationd(keys=["image", "label"], axcodes="RAS"),
#         Spacingd(
#             keys=["image", "label"],
#             pixdim=config['transforms']['spacing']['pixdim'],
#             mode=(config['transforms']['spacing']['mode'], "nearest"),
#         ),
#         ScaleIntensityRanged(
#             keys=["image"],
#             a_min=config['transforms']['scale_intensity']['a_min'],
#             a_max=config['transforms']['scale_intensity']['a_max'],
#             b_min=config['transforms']['scale_intensity']['b_min'],
#             b_max=config['transforms']['scale_intensity']['b_max'],
#             clip=config['transforms']['scale_intensity']['clip'],
#         ),
#         CropForegroundd(keys=["image", "label"], source_key="label",
#                         margin=config['transforms']['crop_foreground']['margin']),
#         ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=config['transforms']['resize']['spatial_size'],
#                              mode=config['transforms']['resize']['mode']),
#     ]
# )

import yaml
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandGaussianNoised,
    EnsureTyped
)

config_path = "config/config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 目标空间尺寸，例如 [128, 128, 128]
target_size = config['transforms']['resize']['spatial_size']

# --- 训练集数据增强 ---
train_transforms = Compose(
    [
        # 1. 只加载 image 路径，label 保持为整数
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),

        # 2. 空间归一化 (仅作用于 image)
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=config['transforms']['spacing']['pixdim'],
            mode=config['transforms']['spacing']['mode'],
        ),

        # 3. 数值归一化 (针对 CT 值范围)
        ScaleIntensityRanged(
            keys=["image"],
            a_min=config['transforms']['scale_intensity']['a_min'],
            a_max=config['transforms']['scale_intensity']['a_max'],
            b_min=config['transforms']['scale_intensity']['b_min'],
            b_max=config['transforms']['scale_intensity']['b_max'],
            clip=config['transforms']['scale_intensity']['clip'],
        ),

        # 4. 自动剪切背景（以图像本身非0区域为基准）
        CropForegroundd(keys=["image"], source_key="image"),

        # 6. 分类常用的数据增强 (仅 image)
        RandFlipd(keys=["image"], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=["image"], spatial_axis=[1], prob=0.5),
        RandFlipd(keys=["image"], spatial_axis=[2], prob=0.5),
        RandRotate90d(keys=["image"], prob=0.5, max_k=4),

        # 5. 统一缩放到 ResNet 输入尺寸
        Resized(keys=["image"], spatial_size=target_size),

        # 强度增强
        RandShiftIntensityd(
            keys=["image"],
            offsets=config['transforms']['rand_shift_intensity']['offsets'],
            prob=config['transforms']['rand_shift_intensity']['prob'],
        ),
        # 推荐增加：随机高斯噪声，提升分类鲁棒性
        RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1),

        # 7. 转换为 Tensor 格式供网络训练
        EnsureTyped(keys=["image", "label"]),
    ]
)

# --- 验证集数据预处理 ---
val_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=config['transforms']['spacing']['pixdim'],
            mode=config['transforms']['spacing']['mode'],
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=config['transforms']['scale_intensity']['a_min'],
            a_max=config['transforms']['scale_intensity']['a_max'],
            b_min=config['transforms']['scale_intensity']['b_min'],
            b_max=config['transforms']['scale_intensity']['b_max'],
            clip=config['transforms']['scale_intensity']['clip'],
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        Resized(keys=["image"], spatial_size=target_size),
        EnsureTyped(keys=["image", "label"]),
    ]
)

