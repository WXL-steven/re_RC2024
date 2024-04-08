import csv
import os

import cv2
import numpy as np
from rich.progress import track
from scipy.interpolate import CubicSpline


class ImageAnnotator:
    def __init__(self, predict_distance=100):
        self.img = None
        self.img_shape = None
        self.img_vis = None
        self.points = []
        self.poly_coeffs = None
        self.predict_distance = predict_distance
        self.vector_pos = None
        self.vector_dir = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 左键点击事件，添加点到列表
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and len(self.points) >= 3:
            # 右键点击事件，且点的数量大于等于3，进行拟合
            self.img_vis = self.img.copy()
            self.fit_polynomial()

    def fit_polynomial(self):
        # 将点分解为x和y坐标，但这次y作为自变量，x作为因变量
        x = np.array([p[0] for p in self.points])
        y = np.array([p[1] for p in self.points])

        # 拟合5次多项式，y作为自变量
        self.poly_coeffs, loss, _, _, _ = np.polyfit(y, x, 5, full=True)

        if loss is []:
            return

        # 使用get_predict_vector方法获取预测的位置和方向
        self.vector_pos, self.vector_dir = self.get_predict_vector(self.img_shape[0] + self.predict_distance)

    def get_predict_vector(self, height):
        # 创建多项式函数
        poly = np.poly1d(self.poly_coeffs)
        # 计算x值
        width = poly(height)
        # 求导
        dx_dy = poly.deriv()
        # 计算导数值（x相对于y的变化率）
        dx_dy_val = dx_dy(height)
        dy_dx_val = 1  # y相对于x的变化率，这里假定为1，表示y以固定速率变化
        # 标准化切线向量
        norm = np.sqrt(dx_dy_val ** 2 + dy_dx_val ** 2)
        dx_dy_val_normalized = dx_dy_val / norm
        dy_dx_val_normalized = dy_dx_val / norm
        return (width, height), (dx_dy_val_normalized, dy_dx_val_normalized)

    def fit_spline(self):
        # 将点分解为x和y坐标
        x = np.array([p[0] for p in self.points])
        y = np.array([p[1] for p in self.points])

        # 定义参数t，这里我们使用点的索引
        t = np.arange(len(self.points))

        # 使用CubicSpline分别对x和y进行拟合
        cs_x = CubicSpline(t, x)
        cs_y = CubicSpline(t, y)

        # 生成平滑的t值
        t_smooth = np.linspace(-10, 10, 300)

        # 使用拟合的样条曲线计算平滑的x和y值
        x_smooth = cs_x(t_smooth)
        y_smooth = cs_y(t_smooth)

        # 绘制曲线
        curve_points = np.array([x_smooth, y_smooth]).T.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(self.img_vis, [curve_points], isClosed=False, color=(255, 0, 0), thickness=2)

    def annotate_image(self, img, data=None):
        self.img = img
        self.img_shape = img.shape
        if data is not None:
            try:
                self.poly_coeffs = np.array(data[0:6], dtype=np.float64)
                self.vector_pos = np.array(data[8:10], dtype=np.float64)
                self.vector_dir = np.array(data[10:12], dtype=np.float64)
            except ValueError:
                self.poly_coeffs = None
                self.vector_pos = None
                self.vector_dir = None
                self.points.clear()
                print("数据格式错误，将重新标注")
        else:
            self.poly_coeffs = None
            self.vector_pos = None
            self.vector_dir = None
            self.points.clear()

        # 向下填充200px灰色像素
        self.img = cv2.copyMakeBorder(self.img, 0, 200, 0, 0, cv2.BORDER_CONSTANT, value=(128, 128, 128))

        if self.img_vis is None:
            self.img_vis = self.img.copy()

        cv2.namedWindow('Image Annotation')
        cv2.setMouseCallback('Image Annotation', self.mouse_callback)

        while True:
            for point in self.points:
                cv2.circle(self.img_vis, point, 3, (0, 255, 0), -1)

            if self.poly_coeffs is not None:
                # 使用拟合的多项式绘制曲线，这次是在y的范围内绘制
                y_vals = np.linspace(0, self.img_vis.shape[0], 300).astype(int)
                x_vals = np.polyval(self.poly_coeffs, y_vals).astype(int)
                curve_points = np.array([x_vals, y_vals]).T.reshape((-1, 1, 2))
                cv2.polylines(self.img_vis, [curve_points], isClosed=False, color=(255, 0, 0), thickness=2)
                for i in range(6):
                    text = f"{np.format_float_scientific(self.poly_coeffs[i], precision=2)}"
                    if not text.startswith("-"):
                        text = " " + text
                    cv2.putText(
                        self.img_vis,
                        text,
                        (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

            if self.vector_pos is not None and self.vector_dir is not None:
                start_pos = (self.vector_pos[0], self.vector_pos[1])
                end_pos = (self.vector_pos[0] - int(self.vector_dir[0] * self.predict_distance),
                           self.vector_pos[1] - int(self.vector_dir[1] * self.predict_distance))
                # 类型转换
                start_pos = (int(start_pos[0]), int(start_pos[1]))
                end_pos = (int(end_pos[0]), int(end_pos[1]))

                cv2.arrowedLine(self.img_vis, start_pos, end_pos, (0, 0, 255), 3)

            cv2.imshow('Image Annotation', self.img_vis)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # 按Esc退出
                break
            elif key == 32:  # 空格键，清空所有点和拟合结果
                self.points.clear()
                self.img_vis = self.img.copy()
                self.poly_coeffs = None
                self.vector_pos = None
                self.vector_dir = None
            elif key == 13:  # 回车键，结束并返回多项式结果
                cv2.destroyAllWindows()
                self.img_vis = None
                self.points.clear()
                if self.vector_pos is None or self.vector_dir is None:
                    self.vector_pos = (None, None)
                    self.vector_dir = (None, None)
                return self.poly_coeffs, (*self.vector_pos, *self.vector_dir)

        cv2.destroyAllWindows()
        self.img_vis = None
        return None

def process_dataset(dataset_folder_path, target_csv_path):
    image_paths = [os.path.join(dataset_folder_path, f) for f in os.listdir(dataset_folder_path) if f.endswith('.jpg')]
    annotator = ImageAnnotator()

    # 尝试读取现有的CSV文件
    existing_data = {}
    try:
        with open(target_csv_path, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # 跳过头部
            for row in csv_reader:
                if len(row) > 0:
                    existing_data[row[0]] = row[1:]
    except FileNotFoundError:
        print("CSV文件不存在，将创建一个新文件。")

    # 处理图片并更新数据
    updated_data = {}
    with open(target_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ['Image Name', 'Coeff0', 'Coeff1', 'Coeff2', 'Coeff3', 'Coeff4', 'Coeff5', 'Left', 'Right', 'Vector_x',
             'Vector_y', 'Vector_dx', 'Vector_dy']
        )
        idx = 0
        while idx < len(image_paths):
            img_path = image_paths[idx]
            img_name = os.path.basename(img_path)
            if img_name in existing_data:
                # 如果图片已存在于CSV中，则使用存储的数据
                data = existing_data[img_name]
                print(f"数据加载为: {data}")
            else:
                data = None

            img = cv2.imread(img_path)
            if img is not None:
                poly_coeffs, vector_args = annotator.annotate_image(img, data=data)
                if poly_coeffs is not None and len(poly_coeffs) == 6:
                    data = list(poly_coeffs) + [0, 0] + list(vector_args)
                else:
                    continue
            else:
                idx += 1
                continue

            csv_writer.writerow([img_name] + data)
            updated_data[img_name] = data
            idx += 1

# def process_dataset(dataset_folder_path, target_csv_path):
#     # 获取数据集文件夹中所有的.jpg图片路径
#     image_paths = [os.path.join(dataset_folder_path, f) for f in os.listdir(dataset_folder_path) if f.endswith('.jpg')
#     ]
#     # 创建ImageAnnotator实例
#     annotator = ImageAnnotator()
#
#     # 准备写入CSV文件
#     with (open(target_csv_path, 'w', newline='') as csvfile):
#         csv_writer = csv.writer(csvfile)
#         # 写入CSV头部
#         csv_writer.writerow(
#             ['Image Name', 'Coeff0', 'Coeff1', 'Coeff2', 'Coeff3', 'Coeff4', 'Coeff5', 'Left', 'Right', 'Vector_x',
#              'Vector_y', 'Vector_dx', 'Vector_dy']
#         )
#
#         pos = 0
#         while pos < len(image_paths):
#             img_path = image_paths[pos]
#             img = cv2.imread(img_path)
#             if img is not None:
#                 poly_coeffs, vector_args = annotator.annotate_image(img)
#                 if poly_coeffs is not None:
#                     if len(poly_coeffs) != 6:
#                         print(f"多项式系数数量错误: {len(poly_coeffs)}")
#                         continue
#                     # 获取文件名
#                     img_name = os.path.basename(img_path)
#                     # 写入CSV
#                     csv_writer.writerow([img_name] + list(poly_coeffs) + [0, 0] + list(vector_args))
#                     pos += 1


# 示例使用
# if __name__ == "__main__":
#     # 加载一张图片
#     img_path = r"C:\Users\Steven\PycharmProjects\re_rc2024\dataset\original\dataset_2024-04-02_13-33-31.jpg"  \
# 替换为你的图片路径
#     img = cv2.imread(img_path)
#     if img is not None:
#         annotator = ImageAnnotator()
#         poly_coeffs = annotator.annotate_image(img)
#         if poly_coeffs is not None:
#             print("多项式系数:", poly_coeffs)
#         else:
#             print("没有多项式结果")
#     else:
#         print("图片加载失败，请检查路径")


if __name__ == "__main__":
    dataset_folder_path = r"C:\Users\Steven\Downloads\QQ\pic"  # 替换为你的数据集文件夹路径
    target_csv_path = r"C:\Users\Steven\Downloads\QQ\data.csv"  # 替换为你的目标CSV文件路径
    try:
        process_dataset(dataset_folder_path, target_csv_path)
    except KeyboardInterrupt:
        print("用户取消操作")
