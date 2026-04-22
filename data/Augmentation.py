# import yaml
# from monai.transforms import (
#     Compose,
#     LoadImaged,
#     EnsureChannelFirstd,
#     Orientationd,
#     Spacingd,
#     ScaleIntensityRanged,
#     CropForegroundd,
#     Resized,
#     RandFlipd,
#     RandRotate90d,
#     RandShiftIntensityd,
#     RandGaussianNoised,
#     EnsureTyped
# )

# config_path = "config/config.yaml"
# with open(config_path, 'r', encoding='utf-8') as f:
#     config = yaml.safe_load(f)

# # 目标空间尺寸，例如 [128, 128, 128]
# target_size = config['transforms']['resize']['spatial_size']

# # --- 训练集数据增强 ---
# train_transforms = Compose(
#     [
#         # 1. 只加载 image 路径，label 保持为整数
#         LoadImaged(keys=["image", "CA", "CAC"]),
#         EnsureChannelFirstd(keys=["image", "CA", "CAC"]),

#         # 2. 空间归一化 (仅作用于 image)
#         Orientationd(keys=["image", "CA", "CAC"], axcodes="RAS"),
#         Spacingd(
#             keys=["image", "CA", "CAC"],
#             pixdim=config['transforms']['spacing']['pixdim'],
#             mode=config['transforms']['spacing']['mode'],
#         ),

#         # 3. 数值归一化 (针对 CT 值范围)
#         ScaleIntensityRanged(
#             keys=["image", "CA", "CAC"],
#             a_min=config['transforms']['scale_intensity']['a_min'],
#             a_max=config['transforms']['scale_intensity']['a_max'],
#             b_min=config['transforms']['scale_intensity']['b_min'],
#             b_max=config['transforms']['scale_intensity']['b_max'],
#             clip=config['transforms']['scale_intensity']['clip'],
#         ),

#         # 4. 自动剪切背景（以图像本身非0区域为基准）
#         CropForegroundd(keys=["image", "CA", "CAC"], source_key="image"),

#         # 6. 分类常用的数据增强 (仅 image)
#         RandFlipd(keys=["image", "CA", "CAC"], spatial_axis=[0], prob=0.5),
#         RandFlipd(keys=["image", "CA", "CAC"], spatial_axis=[1], prob=0.5),
#         RandFlipd(keys=["image", "CA", "CAC"], spatial_axis=[2], prob=0.5),
#         RandRotate90d(keys=["image", "CA", "CAC"], prob=0.5, max_k=4),

#         # 5. 统一缩放到 ResNet 输入尺寸
#         Resized(keys=["image", "CA", "CAC"], spatial_size=target_size),

#         # 强度增强
#         RandShiftIntensityd(
#             keys=["image", "CA", "CAC"],
#             offsets=config['transforms']['rand_shift_intensity']['offsets'],
#             prob=config['transforms']['rand_shift_intensity']['prob'],
#         ),
#         # 推荐增加：随机高斯噪声，提升分类鲁棒性
#         RandGaussianNoised(keys=["image", "CA", "CAC"], prob=0.1, mean=0.0, std=0.1),

#         # 7. 转换为 Tensor 格式供网络训练
#         EnsureTyped(keys=["image", "CA", "CAC", "label"]),
#     ]
# )

# # --- 验证集数据预处理 ---
# val_transforms = Compose(
#     [
#         LoadImaged(keys=["image", "CA", "CAC"]),
#         EnsureChannelFirstd(keys=["image", "CA", "CAC"]),
#         Orientationd(keys=["image", "CA", "CAC"], axcodes="RAS"),
#         Spacingd(
#             keys=["image", "CA", "CAC"],
#             pixdim=config['transforms']['spacing']['pixdim'],
#             mode=config['transforms']['spacing']['mode'],
#         ),
#         ScaleIntensityRanged(
#             keys=["image", "CA", "CAC"],
#             a_min=config['transforms']['scale_intensity']['a_min'],
#             a_max=config['transforms']['scale_intensity']['a_max'],
#             b_min=config['transforms']['scale_intensity']['b_min'],
#             b_max=config['transforms']['scale_intensity']['b_max'],
#             clip=config['transforms']['scale_intensity']['clip'],
#         ),
#         CropForegroundd(keys=["image", "CA", "CAC"], source_key="image"),
#         Resized(keys=["image", "CA", "CAC"], spatial_size=target_size),
#         EnsureTyped(keys=["image", "CA", "CAC", "label"]),
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
        LoadImaged(keys=["image", "CAC"]),
        EnsureChannelFirstd(keys=["image", "CAC"]),

        # 2. 空间归一化 (仅作用于 image)
        Orientationd(keys=["image", "CAC"], axcodes="RAS"),
        Spacingd(
            keys=["image", "CAC"],
            pixdim=config['transforms']['spacing']['pixdim'],
            mode=config['transforms']['spacing']['mode'],
        ),

        # 3. 数值归一化 (针对 CT 值范围)
        ScaleIntensityRanged(
            keys=["image", "CAC"],
            a_min=config['transforms']['scale_intensity']['a_min'],
            a_max=config['transforms']['scale_intensity']['a_max'],
            b_min=config['transforms']['scale_intensity']['b_min'],
            b_max=config['transforms']['scale_intensity']['b_max'],
            clip=config['transforms']['scale_intensity']['clip'],
        ),

        # 4. 自动剪切背景（以图像本身非0区域为基准）
        CropForegroundd(keys=["image", "CAC"], source_key="image"),

        # 6. 分类常用的数据增强 (仅 image)
        RandFlipd(keys=["image", "CAC"], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=["image", "CAC"], spatial_axis=[1], prob=0.5),
        RandFlipd(keys=["image", "CAC"], spatial_axis=[2], prob=0.5),
        RandRotate90d(keys=["image", "CAC"], prob=0.5, max_k=4),

        # 5. 统一缩放到 ResNet 输入尺寸
        Resized(keys=["image", "CAC"], spatial_size=target_size),

        # 强度增强
        RandShiftIntensityd(
            keys=["image", "CAC"],
            offsets=config['transforms']['rand_shift_intensity']['offsets'],
            prob=config['transforms']['rand_shift_intensity']['prob'],
        ),
        # 推荐增加：随机高斯噪声，提升分类鲁棒性
        RandGaussianNoised(keys=["image", "CAC"], prob=0.1, mean=0.0, std=0.1),

        # 7. 转换为 Tensor 格式供网络训练
        EnsureTyped(keys=["image", "CAC", "label"]),
    ]
)

# --- 验证集数据预处理 ---
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "CAC"]),
        EnsureChannelFirstd(keys=["image", "CAC"]),
        Orientationd(keys=["image", "CAC"], axcodes="RAS"),
        Spacingd(
            keys=["image", "CAC"],
            pixdim=config['transforms']['spacing']['pixdim'],
            mode=config['transforms']['spacing']['mode'],
        ),
        ScaleIntensityRanged(
            keys=["image", "CAC"],
            a_min=config['transforms']['scale_intensity']['a_min'],
            a_max=config['transforms']['scale_intensity']['a_max'],
            b_min=config['transforms']['scale_intensity']['b_min'],
            b_max=config['transforms']['scale_intensity']['b_max'],
            clip=config['transforms']['scale_intensity']['clip'],
        ),
        CropForegroundd(keys=["image", "CAC"], source_key="image"),
        Resized(keys=["image", "CAC"], spatial_size=target_size),
        EnsureTyped(keys=["image", "CAC", "label"]),
    ]
)


# import yaml
# import os
# from monai.transforms import (
#     Compose,
#     LoadImaged,
#     EnsureChannelFirstd,
#     Orientationd,
#     Spacingd,
#     ScaleIntensityRanged,
#     CropForegroundd,
#     Resized,
#     RandFlipd,
#     RandRotate90d,
#     RandShiftIntensityd,
#     RandGaussianNoised,
#     EnsureTyped
# )
#
# config_path = "config/config.yaml"
# with open(config_path, 'r', encoding='utf-8') as f:
#     config = yaml.safe_load(f)
#
# # 目标空间尺寸，例如 [128, 128, 128]
# target_size = config['transforms']['resize']['spatial_size']
#
# # --- 训练集数据增强 ---
# train_transforms = Compose(
#     [
#         # 1. 只加载 image 路径，label 保持为整数
#         LoadImaged(keys=["image", "pred"]),
#         EnsureChannelFirstd(keys=["image", "pred"]),
#
#         # 2. 空间归一化 (仅作用于 image)
#         Orientationd(keys=["image", "pred"], axcodes="RAS"),
#         Spacingd(
#             keys=["image", "pred"],
#             pixdim=config['transforms']['spacing']['pixdim'],
#             mode=config['transforms']['spacing']['mode'],
#         ),
#
#         # 3. 数值归一化 (针对 CT 值范围)
#         ScaleIntensityRanged(
#             keys=["image", "pred"],
#             a_min=config['transforms']['scale_intensity']['a_min'],
#             a_max=config['transforms']['scale_intensity']['a_max'],
#             b_min=config['transforms']['scale_intensity']['b_min'],
#             b_max=config['transforms']['scale_intensity']['b_max'],
#             clip=config['transforms']['scale_intensity']['clip'],
#         ),
#
#         # 4. 自动剪切背景（以图像本身非0区域为基准）
#         CropForegroundd(keys=["image", "pred"], source_key="image"),
#
#         # 5. 分类常用的数据增强 (仅 image)
#         RandFlipd(keys=["image", "pred"], spatial_axis=[0], prob=0.5),
#         RandFlipd(keys=["image", "pred"], spatial_axis=[1], prob=0.5),
#         RandFlipd(keys=["image", "pred"], spatial_axis=[2], prob=0.5),
#         RandRotate90d(keys=["image", "pred"], prob=0.5, max_k=4),
#
#         # 6. 统一缩放到 ResNet 输入尺寸
#         Resized(keys=["image", "pred"], spatial_size=target_size),
#
#         # 强度增强
#         RandShiftIntensityd(
#             keys=["image", "pred"],
#             offsets=config['transforms']['rand_shift_intensity']['offsets'],
#             prob=config['transforms']['rand_shift_intensity']['prob'],
#         ),
#         # 推荐增加：随机高斯噪声，提升分类鲁棒性
#         RandGaussianNoised(keys=["image", "pred"], prob=0.1, mean=0.0, std=0.1),
#
#         # 7. 转换为 Tensor 格式供网络训练
#         EnsureTyped(keys=["image", "pred", "label"]),
#     ]
# )
#
# # --- 验证集数据预处理 ---
# val_transforms = Compose(
#     [
#         LoadImaged(keys=["image", "pred"]),
#         EnsureChannelFirstd(keys=["image", "pred"]),
#         Orientationd(keys=["image", "pred"], axcodes="RAS"),
#         Spacingd(
#             keys=["image", "pred"],
#             pixdim=config['transforms']['spacing']['pixdim'],
#             mode=config['transforms']['spacing']['mode'],
#         ),
#         ScaleIntensityRanged(
#             keys=["image", "pred"],
#             a_min=config['transforms']['scale_intensity']['a_min'],
#             a_max=config['transforms']['scale_intensity']['a_max'],
#             b_min=config['transforms']['scale_intensity']['b_min'],
#             b_max=config['transforms']['scale_intensity']['b_max'],
#             clip=config['transforms']['scale_intensity']['clip'],
#         ),
#         CropForegroundd(keys=["image", "pred"], source_key="image"),
#         Resized(keys=["image", "pred"], spatial_size=target_size),
#         EnsureTyped(keys=["image", "pred", "label"]),
#     ]
# )

