import cv2

from debug_bridge.img_ransfer import WebSocketServer

if __name__ == "__main__":
    server = WebSocketServer()


    # 读取图像
    image = cv2.imread("example.jpg")

    # 创建窗口并显示图像
    cv2.imshow("Example Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
