import torch
import torch.nn as nn
import torch.nn.functional as F

def find_centroid_torch(tensor: torch.Tensor):

    h, w = tensor.shape

    # 计算零阶矩 M00 = sum of all values
    M00 = tensor.sum()
    # 如果整张图都为 0，则无法定义质心
    if M00.item() == 0:
        return (None, None)

    # 生成行、列坐标网格
    # y_coords[i, j] = i,   x_coords[i, j] = j
    # 注意要保证和 tensor 在同一个 device 上，以免出现跨设备操作错误
    device = tensor.device
    y_coords = torch.arange(h, device=device).unsqueeze(1).expand(h, w)
    x_coords = torch.arange(w, device=device).unsqueeze(0).expand(h, w)

    # 计算第一阶矩:
    #   M10 = sum of y * I(y, x)
    #   M01 = sum of x * I(y, x)
    M10 = (y_coords * tensor).sum()
    M01 = (x_coords * tensor).sum()

    # 计算质心坐标
    cy = M10 / M00
    cx = M01 / M00

    # 返回 Python float
    return (cy.item(), cx.item())


def shift_image_to_center(tensor: torch.Tensor, mode='bilinear', padding_mode='zeros'):
    """
    1) 先计算二位张量 tensor 的质心
    2) 然后把它平移，使该质心落在图像中心位置

    参数:
    - tensor: torch.Tensor, 形状 (H, W)，可以是灰度或二值图等
    - mode: 插值模式，可选 'bilinear' 或 'nearest' 等
    - padding_mode: 超出边界区域的填充，可选 'zeros'/'border'/'reflection'

    返回:
    - shifted: 同尺寸的 2D Tensor，经过平移插值后的结果
    """

    # 计算质心
    cy, cx = find_centroid_torch(tensor)
    if cy is None:  # 全 0
        return tensor.clone()  # 直接返回原 tensor

    # 计算目标中心
    H, W = tensor.shape
    center_y = (H - 1) / 2.0
    center_x = (W - 1) / 2.0

    # 计算要平移多少像素
    shift_y = center_y - cy
    shift_x = center_x - cx

    # ---- 构造仿射变换参数 (2x3) ----
    # 归一化到 [-1, 1] 坐标系中
    trans_x = 2.0 * shift_x / (W - 1)  # x 方向的归一化平移
    trans_y = 2.0 * shift_y / (H - 1)  # y 方向的归一化平移

    # theta: (1, 2, 3), 其中:
    #   [ [1, 0, trans_x],
    #     [0, 1, trans_y] ]
    theta = torch.tensor([
        [1.0, 0.0, trans_x],
        [0.0, 1.0, trans_y]
    ], dtype=torch.float, device=tensor.device).unsqueeze(0)

    # grid_sample 需要 (N, C, H, W)，故先把 tensor reshape 成 (1,1,H,W)
    input_4d = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # 生成变换后的采样网格
    grid = F.affine_grid(
        theta, 
        size=input_4d.shape,  # (1, 1, H, W)
        align_corners=False
    )

    # 对原图做采样
    output_4d = F.grid_sample(
        input_4d, 
        grid, 
        mode=mode, 
        padding_mode=padding_mode, 
        align_corners=False
    )

    # 去掉批次和通道维度 => (H, W)
    shifted = output_4d.squeeze(0).squeeze(0)

    return shifted


class translation_layer(nn.Module):

    def __init__(self):

        super(translation_layer, self).__init__()

    def forward(self, x):

        outputs = []

        for i in range(4):
            rotated_weight = torch.rot90(self.weight, i, [2, 3])
            output = F.conv2d(x, rotated_weight, bias=None, padding='same')
            outputs.append(output)

        return torch.stack(outputs, dim=2)