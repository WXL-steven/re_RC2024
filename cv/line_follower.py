import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from const import AUTO_PILOT


class AutoPilot:
    def __init__(
            self,
            loop: asyncio.AbstractEventLoop = None,
            corp_bottom: int = 100,
            adaptive_threshold_block_size: int = 35,
            adaptive_threshold_c: int = 25,
            close_kernel_size: int = 3,
            flitter_size: int = 500,
            imshow=None
    ) -> None:
        self.logger = logging.getLogger("re_RC2024.CV.AutoPilot")
        self.executor = ThreadPoolExecutor(max_workers=AUTO_PILOT.MAX_WORKERS)
        self.loop = loop or asyncio.get_event_loop()
        self.corp_bottom: int = corp_bottom
        self.adaptive_threshold_block_size: int = adaptive_threshold_block_size
        self.adaptive_threshold_c: int = adaptive_threshold_c
        self.close_kernel: np.ndarray = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        self.flitter_size: int = flitter_size
        self.imshow: callable = imshow

    async def get_correction(self, frame: np.ndarray):
        # 检查帧是否为空
        if frame is None:
            self.logger.warning("Frame is None")
            return 0, 0

        # 裁剪掉底部100px
        if 0 < self.corp_bottom < frame.shape[0]:
            frame = frame[:-self.corp_bottom, :]
        else:
            self.logger.warning("Invalid corp_bottom value, no crop applied")

        # 复制并转换为灰度图像
        frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)

        # 自适应阈值处理
        thresh = cv2.adaptiveThreshold(
            frame,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            self.adaptive_threshold_block_size,
            self.adaptive_threshold_c
        )

        # 闭处理
        close1 = cv2.morphologyEx(
            thresh,
            cv2.MORPH_CLOSE,
            self.close_kernel
        )

        # 额外膨胀一次
        close1 = cv2.dilate(
            close1,
            kernel=np.ones((5, 5), np.uint8),
            iterations=1
        )

        # 过滤掉较小的连通区域
        filtered_labels = await self.filter_components_by_size(close1, self.flitter_size)

        h, w = filtered_labels.shape

        # 转换为彩色图像, 并向下拓展200px灰色区域
        mark_img = np.zeros((filtered_labels.shape[0] + 200, filtered_labels.shape[1], 3), dtype=np.uint8)
        mark_img[:, :] = (128, 128, 128)
        mark_img[:frame.shape[0], :] = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # # 数据准备
        # left_cols = []
        # left_rows = []
        # right_cols = []
        # right_rows = []
        # center_rows = []
        # col_indices = []
        #
        # for row_index, left_col_index, right_col_index, col_index in result:
        #     if row_index is None:
        #         continue
        #     if left_col_index is not None:
        #         left_rows.append(row_index)
        #         left_cols.append(left_col_index)
        #         cv2.circle(mark_img, (left_col_index, row_index), radius=2, color=(255, 255, 0), thickness=-1)
        #     if right_col_index is not None:
        #         right_rows.append(row_index)
        #         right_cols.append(right_col_index)
        #         cv2.circle(mark_img, (right_col_index, row_index), radius=2, color=(255, 255, 0), thickness=-1)
        #     if col_index is not None:
        #         center_rows.append(row_index)
        #         col_indices.append(col_index)
        #         cv2.circle(mark_img, (col_index, row_index), radius=2, color=(0, 255, 0), thickness=-1)
        #
        # # 数据转换为numpy数组，方便处理
        # left_rows_np = np.array(left_rows)
        # right_rows_np = np.array(right_rows)
        # center_rows_np = np.array(center_rows)
        # left_cols_np = np.array(left_cols)
        # right_cols_np = np.array(right_cols)
        # col_indices_np = np.array(col_indices)
        #
        # dist = 0
        # angle = 0
        #
        # # 多项式拟合
        # # 注意：拟合函数需要x和y的数组，这里我们的x是行（rows），y是列（left_cols, right_cols, col_indices）
        # if len(left_rows_np) > 3 and len(right_rows_np) > 3 and len(center_rows_np) > 3:
        #     left_fit = np.polyfit(left_rows_np, left_cols_np, 2)
        #     right_fit = np.polyfit(right_rows_np, right_cols_np, 2)
        #     # col_fit = np.polyfit(center_rows_np[0:10], col_indices_np[0:10], 2)
        #     # 设置高斯权重分布的参数
        #     mu = center_rows_np[0]
        #     sigma = 50  # 标准差，控制下降速度
        #
        #     # 计算权重
        #     weights = np.exp(-((center_rows_np - mu) ** 2) / (2 * sigma ** 2))
        #
        #     # 使用计算出的权重进行加权多项式拟合
        #     col_fit = np.polyfit(center_rows_np, col_indices_np, 2, w=weights)
        #
        #     # 生成拟合曲线
        #     # 为了绘制曲线，我们需要生成更多的点
        #     row_fit = np.linspace(0, height + 200, 300)
        #     left_fit_curve = np.polyval(left_fit, row_fit)
        #     right_fit_curve = np.polyval(right_fit, row_fit)
        #     col_fit_curve = np.polyval(col_fit, row_fit)
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

        # 反色
        filtered_labels = 255 - filtered_labels

        target_pos = (300, 350)
        cv2.circle(mark_img, target_pos, 5, (0, 0, 255), -1)
        contours, hierarchy = cv2.findContours(filtered_labels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_areas = np.array([cv2.contourArea(contour) for contour in contours])
        top5_indices = np.argsort(-contour_areas)[:5]
        top5_contours = [contours[i] for i in top5_indices]
        max_contour = top5_contours[0]
        # hull = cv2.convexHull(max_contour)  # 寻找凸包

        contours_containing_target = []

        # 遍历这5个轮廓，检查它们是否包含目标点
        for contour in top5_contours:
            # cv2.pointPolygonTest 返回值：
            # -1：点在轮廓外部
            # 0：点在轮廓上
            # 1：点在轮廓内部
            # 第三个参数为False时，只检查点的位置关系，不计算最近距离
            if cv2.pointPolygonTest(contour, target_pos, False) >= 0:
                # 如果点在轮廓内部或轮廓上，将该轮廓添加到结果列表中
                contours_containing_target.append(contour)

        if contours_containing_target:
            if len(contours_containing_target) > 1:
                areas = [cv2.contourArea(contour) for contour in contours_containing_target]
                max_area_index = areas.index(max(areas))
                max_area_contour = contours_containing_target[max_area_index]
            else:
                max_area_contour = contours_containing_target[0]

            cv2.drawContours(mark_img, [max_area_contour], -1, (0, 255, 0), 2)

        # 画出轮廓
        # cv2.drawContours(mark_img, contours, -1, (255, 0, 0), 2)
        cv2.drawContours(mark_img, contours_containing_target, -1, (255, 0, 0), 1)

        scan_img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.drawContours(scan_img, [max_contour], -1, (255, 255, 255), 1)
        scan_img = cv2.cvtColor(scan_img, cv2.COLOR_BGR2GRAY)
        height, width = scan_img.shape
        scan_col_index = width // 2  # 从图像中心开始扫描

        result = []
        for row_index in range(height - 1, -1, -10):  # 反向扫描
            left_col_index, right_col_index = self.find_nearest_diff_horizontal(
                scan_img,
                row_index,
                scan_col_index
            )

            center_col_index = (left_col_index + right_col_index) // 2

            # 根据左右侧结果更新下一次扫描的起始点
            scan_col_index = center_col_index

            # 记录结果
            result.append(
                (row_index, left_col_index, center_col_index, right_col_index)
            )

        dist = 0
        angle = 0
        for row_index, left_col_index, center_col_index, right_col_index in result:
            if row_index is None:
                continue
            if left_col_index is not None:
                cv2.circle(mark_img, (left_col_index, row_index), radius=2, color=(255, 255, 0), thickness=-1)
            if right_col_index is not None:
                cv2.circle(mark_img, (right_col_index, row_index), radius=2, color=(255, 255, 0), thickness=-1)
            if center_col_index is not None:
                cv2.circle(mark_img, (center_col_index, row_index), radius=2, color=(0, 255, 0), thickness=-1)

            # if row_index == 300:
            #     # 绘出两个检测点
            #     cv2.circle(mark_img, (row_index, 150), radius=5, color=(0, 0, 255), thickness=-1)
            #     cv2.circle(mark_img, (row_index, 500), radius=5, color=(0, 0, 255), thickness=-1)
            #
            #     if left_col_index < 150:
            #         angle = -99
            #     elif right_col_index > 500:
            #         angle = 99

        check_point_left = (150, 300)
        check_point_right = (430, 300)

        res_left = cv2.pointPolygonTest(max_contour, check_point_left, False)
        color1 = (0, 255, 0)
        # 在之外
        if res_left < 0:
            angle = -99
            color1 = (0, 0, 255)
        res_right = cv2.pointPolygonTest(max_contour, check_point_right, False)
        color2 = (0, 255, 0)
        # 在之外
        if res_right < 0:
            angle = 99
            color2 = (0, 0, 255)

        # 绘出两个检测点
        cv2.circle(mark_img, check_point_left, radius=5, color=color1, thickness=-1)
        cv2.circle(mark_img, check_point_right, radius=5, color=color2, thickness=-1)

        titles = ['mark_img']
        images = [mark_img]

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

    # async def morphological_filter(
    #         self,
    #         binary_img: np.ndarray,
    #         kernel: np.ndarray,
    # ) -> np.ndarray:
    #     """使用形态学滤波器对二值图像进行处理"""
    #     result_img = await self.loop.run_in_executor(
    #         self.executor,
    #         cv2.findContours,
    #         binary_img,
    #         cv2.MORPH_CLOSE,
    #         kernel
    #     )
    #
    #     return result_img

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
    def find_nearest_diff_horizontal(
            binary_image: np.ndarray,
            row_index: int,
            col_index: int
    ) -> (int, int):
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
        row_pixels = binary_image[row_index]  # 提取到的扫描行
        preset_value = row_pixels[col_index]  # 基准像素值

        # 创建一个与row_pixels相同长度的布尔数组，标记与预设点值不同的位置为True
        diff_mask = row_pixels != preset_value

        # 处理左侧
        left_diff_indices = np.flatnonzero(diff_mask[:col_index][::-1])
        left_col_index = 0 if left_diff_indices.size == 0 else col_index - 1 - left_diff_indices[0]

        # 处理右侧
        right_diff_indices = np.flatnonzero(diff_mask[col_index + 1:])
        right_col_index = len(row_pixels) - 1 if right_diff_indices.size == 0 else col_index + 1 + right_diff_indices[0]

        return left_col_index, right_col_index
