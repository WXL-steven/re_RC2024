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
            corp_bottom: int = 75,
            adaptive_threshold_block_size: int = 35,
            adaptive_threshold_c: int = 25,
            close_kernel_size: int = 3,
            flitter_size: int = 500,
            target_base: tuple[int, int] = (315, 350),
            scan_height: int = 300,
            target_left: int = 150,
            target_right: int = 470,
            accept_inaccuracy: int = 20,
            max_inaccuracy: int = 100,
            mode: str = "center",
            debug: bool = False,
            imshow: callable = None,
    ) -> None:
        self.logger = logging.getLogger("re_RC2024.CV.AutoPilot")
        self.executor = ThreadPoolExecutor(max_workers=AUTO_PILOT.MAX_WORKERS)
        self.loop = loop or asyncio.get_event_loop()
        self.corp_bottom: int = corp_bottom
        self.adaptive_threshold_block_size: int = adaptive_threshold_block_size
        self.adaptive_threshold_c: int = adaptive_threshold_c
        self.close_kernel: np.ndarray = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        self.flitter_size: int = flitter_size

        self.target_base: tuple[int, int] = target_base
        self.scan_height: int = scan_height
        self.target_left: int = target_left
        self.target_right: int = target_right
        self.accept_inaccuracy: int = accept_inaccuracy
        self.max_inaccuracy: int = max_inaccuracy

        if mode.lower() not in ["left", "center", "right"]:
            self.logger.warning("Invalid mode, using center mode")
            self.mode = "center"
        else:
            self.mode = mode

        if debug and imshow is None:
            self.logger.warning("Debug mode is enabled but imshow is not provided, debug mode is disabled")
            self.debug = False
        else:
            self.debug = debug
            self.imshow = imshow

    async def get_correction(self, frame: np.ndarray):
        # 检查帧是否为空
        if frame is None:
            self.logger.warning("Frame is None")
            return 0, 0

        # 裁剪掉底部的部分
        if 0 < self.corp_bottom < frame.shape[0]:
            frame = frame[:-self.corp_bottom, :]
        else:
            self.logger.warning("Invalid corp_bottom value, no crop applied")

        # 复制并转换为灰度图像
        gary = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)

        # 自适应阈值处理
        thresh = cv2.adaptiveThreshold(
            gary,
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

        # 过滤掉较小的连通区域
        filtered_labels = await self.filter_components_by_size(close1, self.flitter_size)

        # 额外膨胀一次
        filtered_labels = cv2.dilate(
            filtered_labels,
            kernel=np.ones((5, 5), np.uint8),
            iterations=1
        )

        h, w = filtered_labels.shape
        # 反色
        filtered_labels = 255 - filtered_labels

        contours, hierarchy = cv2.findContours(filtered_labels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_areas = np.array([cv2.contourArea(contour) for contour in contours])
        top_indices = np.argsort(-contour_areas)[:3]
        top_contours = [contours[i] for i in top_indices]

        contours_containing_target = []

        # 遍历轮廓的凸包，检查它们是否包含目标点
        for contour in top_contours:
            hull = cv2.convexHull(contour)
            # cv2.pointPolygonTest 返回值：
            # -1：点在轮廓外部
            # 0：点在轮廓上
            # 1：点在轮廓内部
            # 第三个参数为False时，只检查点的位置关系，不计算最近距离
            if cv2.pointPolygonTest(hull, self.target_base, False) >= 0:
                # 如果点在轮廓内部或轮廓上，将该轮廓添加到结果列表中
                contours_containing_target.append(contour)

        # 寻找位于预期区域的最大面积的轮廓
        if contours_containing_target:
            if len(contours_containing_target) > 1:
                areas = [cv2.contourArea(contour) for contour in contours_containing_target]
                max_area_index = areas.index(max(areas))
                max_area_contour = contours_containing_target[max_area_index]
            else:
                max_area_contour = contours_containing_target[0]
        else:
            # 无效帧
            return 0, 0

        left_inaccuracy, right_inaccuracy = self.get_inaccuracy(max_area_contour, w, h)

        left_need_fix = False
        right_need_fix = False
        dist = 0
        angle = 0
        if self.mode == "center":
            angle = self.center_patrol(left_inaccuracy, right_inaccuracy)
            left_need_fix = True if angle == -1 else False
            right_need_fix = True if angle == 1 else False
        elif self.mode == "left":
            angle = self.left_patrol(left_inaccuracy)
            left_need_fix = True if angle != 0 else False
            right_need_fix = False
        elif self.mode == "right":
            angle = self.right_patrol(right_inaccuracy)
            left_need_fix = False
            right_need_fix = True if angle != 0 else False

        if self.debug:
            mark_img = self.visualize(
                frame,
                max_area_contour,
                left_inaccuracy,
                right_inaccuracy,
                left_need_fix,
                right_need_fix
            )
            if self.imshow is not None:
                await self.imshow("ap_visualize", mark_img)

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

        # 使用布尔索引过滤labels，将不符合条件的连通域设置为0
        # 这里直接在布尔数组上进行操作，避免了使用lambda表达式
        filtered_labels = np.zeros_like(labels, dtype=np.uint8)
        filtered_labels[labels > 0] = np.where(valid_indices[labels[labels > 0]], 255, 0)

        return filtered_labels

    @staticmethod
    def find_nearest_diff_horizontal(
            binary_image: np.ndarray,
            row_index: int,
            col_index: int
    ) -> tuple[int, int]:
        """
        使用Numpy优化的版本，针对OpenCV二值图像，从给定的行和列坐标开始两侧查找最近的像素值发生变化的水平坐标。

        参数:
        - binary_image: 二维Numpy数组，代表OpenCV的二值图像。
        - row_index: 垂直坐标（行索引）。
        - col_index: 水平像素坐标（列索引）。

        返回:
        - tuple(left_col_index, right_col_index): 两侧最近的像素值发生变化的水平坐标。
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

    def get_inaccuracy(
            self,
            counter: np.ndarray,
            width: int,
            height: int
    ) -> tuple[int, int]:
        if counter is None:
            return 0, 0
        target_img = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(target_img, [counter], -1, (255,), 2)
        height, width = target_img.shape
        scan_col_index = width // 2
        left_col_index, right_col_index = self.find_nearest_diff_horizontal(
            target_img,
            self.scan_height,
            scan_col_index
        )
        if left_col_index is not None and right_col_index is not None:
            left_inaccuracy = left_col_index - self.target_left
            right_inaccuracy = self.target_right - right_col_index
            if left_inaccuracy > self.max_inaccuracy:
                left_inaccuracy = 0
            if right_inaccuracy > self.max_inaccuracy:
                right_inaccuracy = 0
            return left_inaccuracy, right_inaccuracy
        return 0, 0

    def center_patrol(
            self,
            left_inaccuracy: int,
            right_inaccuracy: int
    ) -> int:
        left_overflown = left_inaccuracy > self.accept_inaccuracy
        right_overflown = right_inaccuracy > self.accept_inaccuracy
        if left_overflown and right_overflown:
            return 0
        elif left_overflown:
            return -1
        elif right_overflown:
            return 1
        return 0

    def left_patrol(
            self,
            left_inaccuracy: int,
    ) -> int:
        if left_inaccuracy > self.accept_inaccuracy:
            return -1
        if left_inaccuracy < -self.accept_inaccuracy:
            return 1
        return 0

    def right_patrol(
            self,
            right_inaccuracy: int,
    ) -> int:
        if right_inaccuracy > self.accept_inaccuracy:
            return 1
        if right_inaccuracy < -self.accept_inaccuracy:
            return -1
        return 0

    def switch_mode(self, mode: str) -> None:
        if mode.lower() not in ["left", "center", "right"]:
            self.logger.warning("Invalid mode, using center mode")
            self.mode = "center"
        else:
            self.mode = mode
        self.logger.info(f"Switching mode to {mode}")

    def visualize(
            self,
            frame: np.ndarray,
            target_counter: np.ndarray,
            left_inaccuracy: int,
            right_inaccuracy: int,
            left_need_fix: bool = False,
            right_need_fix: bool = False
    ) -> np.ndarray:
        # 复制原图像
        mark_img = frame.copy()

        # 绘制基准点
        cv2.circle(mark_img, self.target_base, radius=5, color=(255, 255, 0), thickness=-1)

        # 绘制期望位置
        cv2.circle(mark_img, (self.target_left, self.scan_height), radius=5, color=(15, 200, 250), thickness=2)
        cv2.circle(mark_img, (self.target_right, self.scan_height), radius=5, color=(15, 200, 250), thickness=2)

        # 绘制轮廓
        cv2.drawContours(mark_img, [target_counter], -1, (40, 0, 100), 1)

        # 绘制左右扫描结果点
        color_left = (0, 0, 255) if left_need_fix else (0, 255, 0)
        color_right = (0, 0, 255) if right_need_fix else (0, 255, 0)
        cv2.circle(
            mark_img,
            (self.target_left + left_inaccuracy, self.scan_height),
            radius=5,
            color=color_left,
            thickness=-1
        )
        cv2.circle(
            mark_img,
            (self.target_right - right_inaccuracy, self.scan_height),
            radius=5,
            color=color_right,
            thickness=-1
        )

        return mark_img
