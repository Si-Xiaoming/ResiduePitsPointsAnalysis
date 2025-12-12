
"""
点云着色工具
从GeoTIFF影像中提取RGB值并赋予LAS点云
"""

import argparse
import sys
from pathlib import Path
import laspy
import numpy as np
from osgeo import gdal


class PointCloudColorizer:
    """点云着色器类"""

    def __init__(self, las_path, tiff_path, output_path):
        """
        初始化着色器
        """
        self.las_path = Path(las_path)
        self.tiff_path = Path(tiff_path)
        self.output_path = Path(output_path)

        self.las = None
        self.points = None
        self.ds = None
        self.geotransform = None
        self.img_r = None
        self.img_g = None
        self.img_b = None

    def validate_inputs(self):
        """验证输入文件是否存在"""
        if not self.las_path.exists():
            raise FileNotFoundError(f"LAS文件不存在: {self.las_path}")
        if not self.tiff_path.exists():
            raise FileNotFoundError(f"GeoTIFF文件不存在: {self.tiff_path}")

        # 确保输出目录存在
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def load_las(self):
        """读取LAS文件"""
        print(f"正在读取LAS文件: {self.las_path}")
        self.las = laspy.read(str(self.las_path))
        self.points = np.vstack([self.las.x, self.las.y, self.las.z]).transpose()
        print(f"点云总数: {len(self.points)}")

    def load_geotiff(self):
        """读取GeoTIFF文件"""
        print(f"正在读取GeoTIFF: {self.tiff_path}")
        self.ds = gdal.Open(str(self.tiff_path))

        if self.ds is None:
            raise RuntimeError(f"无法打开GeoTIFF文件: {self.tiff_path}")

        # 获取地理变换参数
        self.geotransform = self.ds.GetGeoTransform()
        if not self.geotransform:
            raise RuntimeError("GeoTIFF缺少地理参考信息")

        print(f"影像尺寸: {self.ds.RasterXSize} x {self.ds.RasterYSize}")
        print(f"波段数: {self.ds.RasterCount}")

        # 读取RGB波段
        if self.ds.RasterCount < 3:
            raise RuntimeError(f"GeoTIFF波段数不足，需要至少3个波段，当前: {self.ds.RasterCount}")

        self.img_r = self.ds.GetRasterBand(1).ReadAsArray()
        self.img_g = self.ds.GetRasterBand(2).ReadAsArray()
        self.img_b = self.ds.GetRasterBand(3).ReadAsArray()

    def map_points_to_pixels(self):
        """
        将点云坐标映射到影像像素
        """
        print("正在映射点云坐标到影像像素...")
        cols = ((self.points[:, 0] - self.geotransform[0]) / self.geotransform[1]).astype(int)
        rows = ((self.points[:, 1] - self.geotransform[3]) / self.geotransform[5]).astype(int)
        return cols, rows

    def get_valid_mask(self, cols, rows):
        """
        获取有效点的掩码
        """
        width = self.ds.RasterXSize
        height = self.ds.RasterYSize

        valid_mask = (
                (cols >= 0) & (cols < width) &
                (rows >= 0) & (rows < height)
        )

        valid_count = np.sum(valid_mask)
        print(f"地理参考有效点: {valid_count}/{len(self.points)} ({valid_count / len(self.points) * 100:.2f}%)")

        if valid_count == 0:
            raise RuntimeError("没有有效的点可以赋色！请检查点云与影像的坐标系是否匹配")

        return valid_mask

    def extract_rgb_values(self, cols, rows, valid_mask):
        """
        提取有效点的RGB值
        """
        print("正在提取RGB值...")

        # 8位TIFF -> 16位LAS颜色 (255*257=65535)
        data_type = self.ds.GetRasterBand(1).DataType
        scale_factor = 257 if data_type == gdal.GDT_Byte else 1

        # 提取有效点的RGB值
        rgb_r = self.img_r[rows[valid_mask], cols[valid_mask]] * scale_factor
        rgb_g = self.img_g[rows[valid_mask], cols[valid_mask]] * scale_factor
        rgb_b = self.img_b[rows[valid_mask], cols[valid_mask]] * scale_factor

        return rgb_r, rgb_g, rgb_b

    def filter_zero_colors(self, rgb_r, rgb_g, rgb_b, valid_mask):
        """
        过滤RGB全为0的点
        """
        color_valid_mask = (rgb_r != 0) | (rgb_g != 0) | (rgb_b != 0)
        non_zero_count = np.sum(color_valid_mask)

        print(f"非零颜色点: {non_zero_count}/{len(rgb_r)} ({non_zero_count / len(rgb_r) * 100:.2f}%)")

        if non_zero_count == 0:
            raise RuntimeError("所有有效点的RGB均为0！请检查影像数据")

        # 创建最终有效点掩码
        final_mask = np.zeros(len(self.points), dtype=bool)
        final_mask[valid_mask] = color_valid_mask

        # 过滤RGB值
        filtered_rgb = (
            rgb_r[color_valid_mask].astype(np.uint16),
            rgb_g[color_valid_mask].astype(np.uint16),
            rgb_b[color_valid_mask].astype(np.uint16)
        )

        return final_mask, filtered_rgb

    def create_colored_las(self, final_mask, rgb_values):
        """
        创建着色后的LAS文件

        Args:
            final_mask: 最终有效点掩码
            rgb_values: RGB值元组 (r, g, b)
        """
        colored_count = np.sum(final_mask)
        print(f"正在创建着色点云 (共 {colored_count} 点)...")

        # 创建新LAS对象
        new_las = laspy.LasData(self.las.header)

        # 复制有效点的所有属性
        new_las.points = self.las.points[final_mask]

        # 设置RGB颜色
        new_las.red = rgb_values[0]
        new_las.green = rgb_values[1]
        new_las.blue = rgb_values[2]

        # 保存文件
        print(f"保存到: {self.output_path}")
        new_las.write(str(self.output_path))

    def process(self):
        """执行完整的着色流程"""
        try:
            # 1. 验证输入
            self.validate_inputs()

            # 2. 加载数据
            self.load_las()
            self.load_geotiff()

            # 3. 映射坐标
            cols, rows = self.map_points_to_pixels()

            # 4. 获取有效点
            valid_mask = self.get_valid_mask(cols, rows)

            # 5. 提取RGB值
            rgb_r, rgb_g, rgb_b = self.extract_rgb_values(cols, rows, valid_mask)

            # 6. 过滤零值颜色
            final_mask, filtered_rgb = self.filter_zero_colors(rgb_r, rgb_g, rgb_b, valid_mask)

            # 7. 创建并保存着色点云
            self.create_colored_las(final_mask, filtered_rgb)

            return True

        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            return False


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="从GeoTIFF影像中提取RGB值并赋予LAS点云",
    )

    parser.add_argument(
        '-i', '--las-file',
        help='输入LAS文件路径',
        default="D:/04-Datasets/8-6_laz/8-6_trans_trans.las"
    )

    parser.add_argument(
        '-t', '--tiff-file',
        help='输入GeoTIFF文件路径',
        default="D:/04-Datasets/8-6_laz/EOSDOMImage.tif"
    )

    parser.add_argument(
        '-o', '--output',
        help='输出LAS文件路径',
        default="D:/04-Datasets/8-6_laz/8-6_trans_trans_colored2.las"
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()

    colorizer = PointCloudColorizer(
        las_path=args.las_file,
        tiff_path=args.tiff_file,
        output_path=args.output
    )

    success = colorizer.process()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()