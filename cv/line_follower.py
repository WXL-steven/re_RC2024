import logging
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import cv2
import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import CubicSpline

from const import AUTO_PILOT


class AutoPilot:
    def __init__(self,
                 loop: asyncio.AbstractEventLoop = None,
                 imshow=None
                 ) -> None:
        self.logger = logging.getLogger("re_RC2024.CV.AutoPilot")
        self.executor = ThreadPoolExecutor(max_workers=AUTO_PILOT.MAX_WORKERS)
        self.loop = loop or asyncio.get_event_loop()
        self.imshow = imshow

    async def play_ground(self, frame: np.ndarray):
        upper = 255
        lower = 127

        # 裁剪掉底部100px
        frame = frame[:-100, :]

        frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)

        # 手动均值平滑
        # frame = await self.loop.run_in_executor(
        #     self.executor,
        #     cv2.blur,
        #     frame,
        #     (5, 5)
        # )

        thresh1 = await self.loop.run_in_executor(
            self.executor,
            cv2.adaptiveThreshold,
            frame,
            upper,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            35,
            25
        )

        # 开处理
        # open1 = await self.loop.run_in_executor(
        #     self.executor,
        #     cv2.morphologyEx,
        #     thresh1,
        #     cv2.MORPH_OPEN,
        #     np.ones((5, 5), np.uint8)
        # )

        # 闭处理
        close1 = await self.loop.run_in_executor(
            self.executor,
            cv2.morphologyEx,
            thresh1,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.uint8)
        )

        # 膨胀处理
        # dilate1 = await self.loop.run_in_executor(
        #     self.executor,
        #     cv2.dilate,
        #     thresh1,
        #     np.ones((3, 3), np.uint8),
        #     1
        # )

        # pre_orp = await self.remove_small_black_areas(close1, 1500)
        filtered_labels = await self.filter_components_by_size(close1, 800)

        # 转换为彩色图像
        mark_img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # 扫描图像
        # points = await self.loop.run_in_executor(
        #     self.executor,
        #     self.scan_image,
        #     filtered_labels,
        #     20,
        #     20
        # )
        #
        # for row_index in points:
        #     print(row_index)
        #     left_col_index, right_col_index = row_index[1], row_index[2]
        #
        #     # 绘制左侧结果（如果有，用黄色）
        #     if left_col_index is not None:
        #         cv2.circle(mark_img, left_col_index, radius=2, color=(255, 255, 0), thickness=-1)
        #
        #     # 绘制右侧结果（如果有，用黄色）
        #     if right_col_index is not None:
        #         cv2.circle(mark_img, right_col_index, radius=2, color=(255, 255, 0), thickness=-1)
        #
        #     # 绘制起点（绿色）
        #     # 转换为Tuple[int, int]
        #     pos = tuple(row_index[0].astype(int))
        #     cv2.circle(mark_img, pos, radius=2, color=(0, 255, 0), thickness=-1)

        # height, width = filtered_labels.shape[:2]
        # center_x = width // 2
        # center_y = height
        #
        # # 初始点
        # p1 = np.array([center_x, center_y + 10])  # 底部中心点
        # p2 = np.array([center_x, center_y - 1])  # 向上偏移的点
        # cv2.circle(mark_img, tuple(p1), radius=2, color=(0, 255, 0), thickness=-1)
        # cv2.circle(mark_img, tuple(p2), radius=2, color=(0, 255, 0), thickness=-1)
        #
        # # t1 = time.time_ns()
        # c, edge1, edge2 = self.calculate_extended_perpendicular_points(p1, p2, 20, width, height)
        # # print(f"edge Time: {(time.time_ns() - t1) / 1e6:.2f}ms")
        # cv2.circle(mark_img, tuple(c), radius=2, color=(255, 0, 0), thickness=-1)
        #
        # # t2 = time.time_ns()
        # curb1, curb2 = self.find_nearest_diff(filtered_labels, c, edge1, edge2)
        # # print(f"curb Time: {(time.time_ns() - t2) / 1e6:.2f}ms")
        # if curb1 is not None:
        #     cv2.circle(mark_img, curb1, radius=2, color=(255, 255, 0), thickness=-1)
        # if curb2 is not None:
        #     cv2.circle(mark_img, curb2, radius=2, color=(255, 255, 0), thickness=-1)
        #
        # p1 = p2
        # p2 = tuple(np.mean([curb1, curb2], axis=0).astype(int))
        # cv2.circle(mark_img, p2, radius=2, color=(0, 255, 0), thickness=-1)
        #
        # c, edge1, edge2 = self.calculate_extended_perpendicular_points(p1, p2, 20, width, height)
        # cv2.circle(mark_img, tuple(c), radius=2, color=(255, 0, 0), thickness=-1)
        #
        # curb1, curb2 = self.find_nearest_diff(filtered_labels, c, edge1, edge2)
        # if curb1 is not None:
        #     cv2.circle(mark_img, curb1, radius=2, color=(255, 255, 0), thickness=-1)
        # if curb2 is not None:
        #     cv2.circle(mark_img, curb2, radius=2, color=(255, 255, 0), thickness=-1)

        height, width = filtered_labels.shape
        col_index = width // 2  # 初始使用水平中点作为起始点

        failed_count = 0
        result = []
        for row_index in range(height - 1 - (height - 1) % 20, -1, -20):  # 每隔20像素
            left_col_index, right_col_index = self.find_nearest_diff_horizontal(filtered_labels, row_index, col_index)

            if left_col_index is None:
                failed_count += 1

            if right_col_index is None:
                failed_count += 1

            # 根据左右侧结果更新下一次扫描的起始点
            if left_col_index is not None and right_col_index is not None:
                col_index = (left_col_index + right_col_index) // 2

            # 记录结果
            result.append((row_index, left_col_index, right_col_index,
                           col_index if left_col_index is not None and right_col_index is not None else None))

            if failed_count > 3:
                break

        # 数据准备
        left_cols = []
        left_rows = []
        right_cols = []
        right_rows = []
        center_rows = []
        col_indices = []

        for row_index, left_col_index, right_col_index, col_index in result:
            if row_index is None:
                continue
            if left_col_index is not None:
                left_rows.append(row_index)
                left_cols.append(left_col_index)
                cv2.circle(mark_img, (left_col_index, row_index), radius=2, color=(255, 255, 0), thickness=-1)
            if right_col_index is not None:
                right_rows.append(row_index)
                right_cols.append(right_col_index)
                cv2.circle(mark_img, (right_col_index, row_index), radius=2, color=(255, 255, 0), thickness=-1)
            if col_index is not None:
                center_rows.append(row_index)
                col_indices.append(col_index)
                cv2.circle(mark_img, (col_index, row_index), radius=2, color=(0, 255, 0), thickness=-1)

        # 注意：rows列表可能因为left/right分别添加而有重复，需要去重
        # rows = list(set(rows))

        # 数据转换为numpy数组，方便处理
        left_rows_np = np.array(left_rows)
        right_rows_np = np.array(right_rows)
        center_rows_np = np.array(center_rows)
        left_cols_np = np.array(left_cols)
        right_cols_np = np.array(right_cols)
        col_indices_np = np.array(col_indices)

        dist = 0
        angle = 0

        # 使用CubicSpline拟合中心线
        if len(center_rows_np) > 3:
            # 分解为x和y坐标
            x = col_indices_np
            y = center_rows_np
            t = np.arange(len(center_rows_np))
            cs_x = CubicSpline(t, x)
            cs_y = CubicSpline(t, y)
            t_smooth = np.linspace(-10, 30, 300)
            x_smooth = cs_x(t_smooth)
            y_smooth = cs_y(t_smooth)
            curve_points = np.array([x_smooth, y_smooth]).T.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(mark_img, [curve_points], False, (0, 255, 0), 2)

            # 创建CubicSpline对象
            cs_x = CubicSpline(t, x)
            cs_y = CubicSpline(t, y)

            # 在t=0处计算切线斜率的导数
            dx_dt = cs_x.derivative()(1)
            dy_dt = cs_y.derivative()(1)

            # 计算切线向量的模（长度）
            length = np.sqrt(dx_dt ** 2 + dy_dt ** 2)

            # 标准化切线向量
            dx_dt_normalized = dx_dt / length
            dy_dt_normalized = dy_dt / length
            # print(f"vector1: {dx_dt_normalized}, {dy_dt_normalized}")

            # 确定箭头的长度
            arrow_length = 50  # 你可以根据需要调整这个长度

            # 确定箭头的起点（t=0时的点）
            start_point = (int(x[1]), int(y[1]))

            # 确定箭头的终点
            end_point = (int(start_point[0] + arrow_length * dx_dt_normalized),
                         int(start_point[1] + arrow_length * dy_dt_normalized))

            # 绘制箭头
            cv2.arrowedLine(mark_img, start_point, end_point, (255, 0, 0), 2)

            dist = width // 2 - end_point[0]
            cv2.putText(mark_img, f"Dist: {dist}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cos_theta = -dy_dt_normalized / np.sqrt(dx_dt_normalized ** 2 + dy_dt_normalized ** 2)

            # 计算角度，使用arccos并转换为度
            angle = np.arccos(cos_theta) * (180.0 / np.pi)

            # 判断方向（叉乘的符号）
            if -dx_dt_normalized < 0:
                angle = -angle  # 偏右为正，偏左为负

            cv2.putText(mark_img, f"Angle: {angle:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # # 多项式拟合
        # # 注意：拟合函数需要x和y的数组，这里我们的x是行（rows），y是列（left_cols, right_cols, col_indices）
        # if len(left_rows_np) > 3 and len(right_rows_np) > 3 and len(center_rows_np) > 3:
        #     left_fit = np.polyfit(left_rows_np, left_cols_np, 3)
        #     right_fit = np.polyfit(right_rows_np, right_cols_np, 3)
        #     col_fit = np.polyfit(center_rows_np, col_indices_np, 5)
        #
        #     # 生成拟合曲线
        #     # 为了绘制曲线，我们需要生成更多的点
        #     row_fit = np.linspace(0, height, 300)
        #     left_fit_curve = np.polyval(left_fit, row_fit)
        #     right_fit_curve = np.polyval(right_fit, row_fit)
        #     col_fit_curve = np.polyval(col_fit, row_fit)
        #     # print(f"Args: \n{left_fit}\n{right_fit}\n{col_fit}")
        #
        #     # 使用numpy将小于1e-5的值设置为0
        #     left_fit_curve[np.abs(left_fit_curve) < 1e-5] = 0
        #     right_fit_curve[np.abs(right_fit_curve) < 1e-5] = 0
        #     col_fit_curve[np.abs(col_fit_curve) < 1e-5] = 0
        #
        #     # 将曲线点转换为适合cv2.polylines的格式
        #     left_curve_points = np.array([left_fit_curve, row_fit]).T.reshape(-1, 1, 2).astype(np.int32)
        #     right_curve_points = np.array([right_fit_curve, row_fit]).T.reshape(-1, 1, 2).astype(np.int32)
        #     col_curve_points = np.array([col_fit_curve, row_fit]).T.reshape(-1, 1, 2).astype(np.int32)
        #
        #     # 绘制曲线
        #     cv2.polylines(mark_img, [left_curve_points], False, (255, 255, 0), 2)  # 左侧曲线，黄色
        #     cv2.polylines(mark_img, [right_curve_points], False, (255, 255, 0), 2)  # 右侧曲线，黄色
        #     cv2.polylines(mark_img, [col_curve_points], False, (0, 255, 0), 2)  # 中心曲线，绿色
        #
        #     # 创建多项式函数
        #     poly_col = np.poly1d(col_fit)
        #
        #     # 计算导数
        #     poly_col_deriv = poly_col.deriv()
        #
        #     # 在特定点（比如 height-1 行）计算导数值
        #     t = height - 30
        #     dx_dt = poly_col_deriv(t)  # x的变化率是常数
        #     dy_dt = 1  # y的变化率由导数给出
        #
        #     # 计算切线向量的模（长度）
        #     length = np.sqrt(dx_dt ** 2 + dy_dt ** 2)
        #
        #     # 标准化切线向量
        #     dx_dt_normalized = dx_dt / length
        #     dy_dt_normalized = dy_dt / length
        #     # print(f"vector2: {dx_dt_normalized}, {dy_dt_normalized}")
        #
        #     # 确定箭头的长度和起点终点
        #     arrow_length = 50  # 可根据需要调整长度
        #     start_point = (int(poly_col(t)), int(t))
        #     end_point = (int(start_point[0] - arrow_length * dx_dt_normalized),
        #                  int(start_point[1] - arrow_length * dy_dt_normalized))
        #
        #     # 绘制箭头，mark_img 应已定义
        #     cv2.arrowedLine(mark_img, start_point, end_point, (255, 0, 0), 2)
        #
        #     # 计算距离和角度
        #     dist = width // 2 - end_point[0]
        #     cv2.putText(mark_img, f"Dist: {dist}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        #
        #     cos_theta = -dy_dt_normalized / np.sqrt(dx_dt_normalized ** 2 + dy_dt_normalized ** 2)
        #     angle = np.arccos(cos_theta) * (180.0 / np.pi)
        #     if dx_dt_normalized < 0:  # 注意这里的条件改变了，因为dx_dt总是正的
        #         angle = -angle  # 偏右为正，偏左为负
        #
        #     # curbs = self.scan_image(filtered_labels, 20, 20)
        #
        # for curb in curbs:
        #     c, curb1, curb2 = curb
        #     cv2.circle(mark_img, c, radius=2, color=(255, 0, 0), thickness=-1)
        #     if curb1 is not None:
        #         cv2.circle(mark_img, curb1, radius=2, color=(255, 255, 0), thickness=-1)
        #     if curb2 is not None:
        #         cv2.circle(mark_img, curb2, radius=2, color=(255, 255, 0), thickness=-1)

        #
        # left_point, right_point = self.find_nearest_diff(filtered_labels, c, line_waypoints)
        # if left_point is not None:
        #     cv2.circle(mark_img, left_point, radius=2, color=(255, 255, 0), thickness=-1)
        #
        # if right_point is not None:
        #     cv2.circle(mark_img, right_point, radius=2, color=(255, 255, 0), thickness=-1)

        titles = ['filtered_labels', 'mark_img']
        images = [filtered_labels, mark_img]

        if self.imshow is not None:
            for i in range(len(titles)):
                await self.imshow(titles[i], images[i])

        return dist, angle

    async def filter_components_by_size(
            self,
            binary_img: np.ndarray,
            min_size: int,
            max_size: int = np.inf,
            p: bool = False
    ) -> np.ndarray:
        """根据面积过滤连通区域"""
        retval, labels, stats, centroids = await self.loop.run_in_executor(
            self.executor,
            cv2.connectedComponentsWithStats,
            binary_img,
            4,
            cv2.CV_16U,
            cv2.CCL_SAUF
        )

        if p:
            print(f"Top 5 max size area: {np.sort(stats[:, 4])[-5:]}")

        # 获取符合面积条件的连通域的索引
        valid_indices = (stats[:, 4] >= min_size) & (stats[:, 4] <= max_size)

        # print(f"Top 3 max size after filter: {np.sort(stats[valid_indices, 4])[-3:]}")

        # 使用布尔索引过滤labels，将不符合条件的连通域设置为0
        # 这里直接在布尔数组上进行操作，避免了使用lambda表达式
        filtered_labels = np.zeros_like(labels, dtype=np.uint8)
        filtered_labels[labels > 0] = np.where(valid_indices[labels[labels > 0]], 255, 0)

        return filtered_labels

    async def remove_small_black_areas(self, binary_img: np.ndarray, min_size: int) -> np.ndarray:
        """反转图像颜色，过滤掉较小的黑色块（现在是白色），再反转回来"""
        # 反转图像颜色
        inverted_img = 255 - binary_img

        # 使用类似之前的函数过滤掉较小的连通区域
        filtered_inverted_img = await self.filter_components_by_size(inverted_img, min_size)

        # 再次反转图像颜色，恢复原始颜色分布
        result_img = 255 - filtered_inverted_img

        return result_img

    @staticmethod
    def find_nearest_diff_horizontal(binary_image, row_index, col_index):
        """
        使用Numpy优化的版本，针对OpenCV二值图像，从给定的行和列坐标开始两侧查找最近的像素值发生变化的水平坐标。

        参数:
        - binary_image: 二维Numpy数组，代表OpenCV的二值图像。
        - row_index: 垂直坐标（行索引）。
        - col_index: 水平像素坐标（列索引）。

        返回:
        - (left_col_index, right_col_index): 两侧最近的像素值发生变化的水平坐标。
          如果在某一侧找不到这样的元素，则该侧返回None。
        """
        # 获取指定行
        row_pixels = binary_image[row_index]
        preset_value = row_pixels[col_index]

        # 创建一个与row_pixels相同长度的布尔数组，标记与预设点值不同的位置为True
        diff_mask = row_pixels != preset_value

        # 处理左侧
        left_diff_indices = np.flatnonzero(diff_mask[:col_index][::-1])
        left_col_index = None if left_diff_indices.size == 0 else col_index - 1 - left_diff_indices[0]

        # 处理右侧
        right_diff_indices = np.flatnonzero(diff_mask[col_index + 1:])
        right_col_index = None if right_diff_indices.size == 0 else col_index + 1 + right_diff_indices[0]

        return left_col_index, right_col_index

    @staticmethod
    def find_nearest_diff_vertical(binary_image, row_index, col_index):
        """
        使用Numpy优化的版本，针对OpenCV二值图像，从给定的行和列坐标开始上下查找最近的像素值发生变化的垂直坐标。

        参数:
        - binary_image: 二维Numpy数组，代表OpenCV的二值图像。
        - row_index: 垂直坐标（行索引）。
        - col_index: 水平像素坐标（列索引）。

        返回:
        - (upper_row_index, lower_row_index): 上下两侧最近的像素值发生变化的垂直坐标。
          如果在某一侧找不到这样的元素，则该侧返回None。
        """
        # 获取指定列的所有像素
        col_pixels = binary_image[:, col_index]
        preset_value = col_pixels[row_index]

        # 创建一个与col_pixels相同长度的布尔数组，标记与预设点值不同的位置为True
        diff_mask = col_pixels != preset_value

        # 处理上侧
        upper_diff_indices = np.flatnonzero(diff_mask[:row_index][::-1])
        upper_row_index = None if upper_diff_indices.size == 0 else row_index - 1 - upper_diff_indices[0]

        # 处理下侧
        lower_diff_indices = np.flatnonzero(diff_mask[row_index + 1:])
        lower_row_index = None if lower_diff_indices.size == 0 else row_index + 1 + lower_diff_indices[0]

        return upper_row_index, lower_row_index

    # @staticmethod
    # def create_line_waypoint(p1, p2, img):
    #     """
    #     生成一个数组，包含两点之间直线上每个像素的坐标和强度
    #
    #     参数:
    #         - P1: 一个包含第一个点坐标(x,y)的numpy数组
    #         - P2: 一个包含第二个点坐标(x,y)的numpy数组
    #         - img: 正在处理的图像
    #
    #     返回值:
    #         - it: 一个numpy数组，包含直线上每个像素的坐标和强度（形状: [numPixels, 2], 行 = [x,y]）
    #
    #     引用来源: https://stackoverflow.com/a/32857432
    #     """
    #     # 为了可读性定义局部变量
    #     image_h = img.shape[0]
    #     image_w = img.shape[1]
    #     p1_x = p1[0]
    #     p1_y = p1[1]
    #     p2_x = p2[0]
    #     p2_y = p2[1]
    #
    #     # 计算点之间的差异和绝对差异
    #     # 用于计算斜率和点之间的相对位置
    #     d_x = p2_x - p1_x
    #     d_y = p2_y - p1_y
    #     d_xa = np.abs(d_x)
    #     d_ya = np.abs(d_y)
    #
    #     # 根据点之间的距离预定义输出的numpy数组
    #     it_buffer = np.empty(shape=(np.maximum(d_ya, d_xa), 2), dtype=np.float32)
    #     it_buffer.fill(np.nan)
    #
    #     # 使用Bresenham算法的一种形式获取直线上的坐标
    #     neg_y = p1_y > p2_y
    #     neg_x = p1_x > p2_x
    #     if p1_x == p2_x:  # 垂直线段
    #         it_buffer[:, 0] = p1_x
    #         if neg_y:
    #             it_buffer[:, 1] = np.arange(p1_y - 1, p1_y - d_ya - 1, -1)
    #         else:
    #             it_buffer[:, 1] = np.arange(p1_y + 1, p1_y + d_ya + 1)
    #     elif p1_y == p2_y:  # 水平线段
    #         it_buffer[:, 1] = p1_y
    #         if neg_x:
    #             it_buffer[:, 0] = np.arange(p1_x - 1, p1_x - d_xa - 1, -1)
    #         else:
    #             it_buffer[:, 0] = np.arange(p1_x + 1, p1_x + d_xa + 1)
    #     else:  # 对角线段
    #         steep_slope = d_ya > d_xa
    #         if steep_slope:
    #             slope = d_x.astype(np.float32) / d_y.astype(np.float32)
    #             if neg_y:
    #                 it_buffer[:, 1] = np.arange(p1_y - 1, p1_y - d_ya - 1, -1)
    #             else:
    #                 it_buffer[:, 1] = np.arange(p1_y + 1, p1_y + d_ya + 1)
    #             it_buffer[:, 0] = (slope * (it_buffer[:, 1] - p1_y)).astype(np.int8) + p1_x
    #         else:
    #             slope = d_y.astype(np.float32) / d_x.astype(np.float32)
    #             if neg_x:
    #                 it_buffer[:, 0] = np.arange(p1_x - 1, p1_x - d_xa - 1, -1)
    #             else:
    #                 it_buffer[:, 0] = np.arange(p1_x + 1, p1_x + d_xa + 1)
    #             it_buffer[:, 1] = (slope * (it_buffer[:, 0] - p1_x)).astype(np.int8) + p1_y
    #
    #     # 移除图像外的点
    #     col_x = it_buffer[:, 0]
    #     col_y = it_buffer[:, 1]
    #     it_buffer = it_buffer[(col_x >= 0) & (col_y >= 0) & (col_x < image_w) & (col_y < image_h)]
    #
    #     # 从img ndarray获取强度值
    #     # it_buffer[:, 2] = img[it_buffer[:, 1].astype(np.int8), it_buffer[:, 0].astype(np.int8)]
    #
    #     return it_buffer.astype(int)
    #
    # @staticmethod
    # def find_line_border_intersections(p, v, width, height):
    #     """
    #     找到由点p和向量v定义的直线与图像边界的交点。
    #
    #     参数:
    #         - p: 一个包含点坐标(x,y)的numpy数组。
    #         - v: 一个包含向量坐标(x,y)的numpy数组。
    #         - width: 图像的宽度。
    #         - height: 图像的高度。
    #
    #     返回:
    #         - 一个包含两个有效交点的列表，每个交点都是一个包含坐标(x,y)的numpy数组。
    #     """
    #     x0, y0 = p
    #     vx, vy = v
    #
    #     # 防止除以零
    #     vx = np.where(vx == 0, 1e-10, vx)
    #
    #     m = vy / vx  # 斜率
    #     m = np.where(m == 0, 1e-10, m)
    #     b = y0 - m * x0  # y轴截距
    #
    #     # 计算所有潜在交点
    #     x_intersects = np.array([0, width, -b / m, (height - b) / m])
    #     y_intersects = np.array([b, m * width + b, 0, height])
    #
    #     # 筛选有效交点
    #     valid_mask = (x_intersects >= 0) & (x_intersects <= width) & (y_intersects >= 0) & (y_intersects <= height)
    #     valid_intersections = np.vstack((x_intersects[valid_mask], y_intersects[valid_mask])).T
    #
    #     # 理想情况下，应该有2个有效交点
    #     if len(valid_intersections) != 2:
    #         return [p - v, p + v]
    #
    #     return valid_intersections[:2]  # 返回前两个有效交点
    #
    # def calculate_extended_perpendicular_points(
    #         self,
    #         point_a: np.ndarray,
    #         point_b: np.ndarray,
    #         extension_distance: float,
    #         width: int,
    #         height: int
    # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     """
    #     计算并返回沿AB方向向量延长一定距离后的点C，以及过C点垂直于AB方向的两个点。
    #
    #     参数:
    #         - a: 点A的坐标，形如(x, y)。
    #         - b: 点B的坐标，形如(x, y)。
    #         - extension_distance: 从B点沿AB方向向量延长的距离。
    #         - perpendicular_distance: 从C点沿垂直于AB方向的距离，用于确定直线上的两个点。
    #
    #     返回:
    #         - c: 沿AB方向向量延长一定距离后的新点C。
    #         - point1: 过C点垂直于AB方向的一个点。
    #         - point2: 过C点垂直于AB方向的另一个点。
    #     """
    #     # 将输入坐标转换为NumPy数组
    #     a = np.array(point_a)
    #     b = np.array(point_b)
    #
    #     # 计算AB方向向量并单位化
    #     ab_vector = b - a
    #     # print(f"ab_vector: {ab_vector}")
    #     unit_ab_vector = ab_vector / np.linalg.norm(ab_vector)
    #     # print(f"unit_ab_vector: {unit_ab_vector}")
    #
    #     # 计算新的C点
    #     c = b + unit_ab_vector * extension_distance
    #     # print(f"c: {c}")
    #
    #     # 计算垂直于AB的方向向量并单位化
    #     perp_vector = np.array([-ab_vector[1], ab_vector[0]])
    #     # print(f"perp_vector: {perp_vector}")
    #     unit_perp_vector = perp_vector / np.linalg.norm(perp_vector)
    #     # print(f"unit_perp_vector: {unit_perp_vector}")
    #
    #     # 计算垂直点
    #     point1, point2 = self.find_line_border_intersections(c, unit_perp_vector, width, height)
    #     # point1 = c - unit_perp_vector * perpendicular_distance
    #     # point2 = c + unit_perp_vector * perpendicular_distance
    #
    #     c = c.astype(int)
    #     point1 = point1.astype(int)
    #     point2 = point2.astype(int)
    #
    #     return c, point1, point2
    #
    # def find_nearest_diff(
    #         self,
    #         labels: np.ndarray,
    #         index_point: np.ndarray,
    #         point1: np.ndarray,
    #         point2: np.ndarray
    # ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    #     def find_diff_point(waypoints, index_label):
    #         # 筛选出标签值不同的点
    #         diff_labels = labels[waypoints[:, 1].astype(int), waypoints[:, 0].astype(int)] != index_label
    #         diff_waypoints = waypoints[diff_labels]
    #
    #         if len(diff_waypoints) == 0:
    #             return None
    #
    #         # 计算曼哈顿距离
    #         manhattan_distances = np.abs(diff_waypoints[:, 0] - index_point[0]) + np.abs(
    #             diff_waypoints[:, 1] - index_point[1])
    #         # 找到曼哈顿距离最小的点
    #         nearest_index = np.argmin(manhattan_distances)
    #         return tuple(diff_waypoints[nearest_index])
    #
    #     # 获取index_point的标签值
    #     index_label = labels[index_point[1], index_point[0]]
    #
    #     # 为两条线分别获取像素点坐标
    #     waypoints1 = self.create_line_waypoint(index_point, point1, labels)
    #     waypoints2 = self.create_line_waypoint(index_point, point2, labels)
    #
    #     nearest_diff_point1 = find_diff_point(waypoints1, index_label)
    #     nearest_diff_point2 = find_diff_point(waypoints2, index_label)
    #
    #     return nearest_diff_point1, nearest_diff_point2
    #
    # def scan_image(self, frame: np.ndarray, max_iterations=20, extend_dis=20) -> list:
    #     height, width = frame.shape[:2]
    #     results = []
    #
    #     # 假设center_x和center_y是您希望开始扫描的初始位置
    #     center_x, center_y = width // 2, height // 2
    #     p1 = np.array([center_x, center_y + 10])  # 底部中心点
    #     p2 = np.array([center_x, center_y - 1])  # 向上偏移的点
    #
    #     for _ in range(max_iterations):
    #         c, edge1, edge2 = self.calculate_extended_perpendicular_points(p1, p2, extend_dis, width, height)
    #
    #         curb1, curb2 = self.find_nearest_diff(frame, c, edge1, edge2)
    #         if curb1 is not None and curb2 is not None:
    #             # 更新p1和p2为下一次迭代
    #             p1 = p2
    #             p2 = tuple(np.mean([curb1, curb2], axis=0).astype(int))
    #
    #             # 添加结果
    #             results.append((c, curb1, curb2))
    #
    #             # 检查c是否超出图片范围
    #             if c[0] < 0 or c[0] >= width or c[1] < 0 or c[1] >= height:
    #                 break
    #         else:
    #             results.append((c, None, None))
    #
    #     return results
    #
    # async def road_analysis(self, frame: np.ndarray) -> any:
    #     pass
