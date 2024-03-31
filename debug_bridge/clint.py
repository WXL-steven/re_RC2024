import asyncio
import struct

import websockets
import cv2
import numpy as np

control_keys_map = {
    ord('w'): "w",  # 前进
    ord('s'): "s",  # 后退
    ord('a'): "a",  # 左平移
    ord('d'): "d",  # 右平移
    ord('q'): "q",  # 左转
    ord('e'): "e",  # 右转
    ord(' '): " ",  # 停止
    ord('8'): "8",  # 舵机抬升
    ord('2'): "2",  # 舵机下降
    ord('4'): "4",  # 舵机左转
    ord('6'): "6",  # 舵机右转
    ord('5'): "5",  # 舵机复位
}


async def receive_images(uri):
    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket server")

        last_ord = " "

        while True:  # 持续接收图像
            # 接收整个消息
            data = await websocket.recv()

            # 先解析描述文本长度（假设描述文本长度占用1字节）
            description_length = struct.unpack("!B", data[:1])[0]

            # 确定头部的总长度：描述文本长度的字节（1字节） + 描述文本 + 图像数据长度（4字节）
            header_total_length = 1 + description_length + 4

            # 解析头部信息，获取描述文本和图像数据长度
            header_format = f"!{description_length}sI"
            description, image_length = struct.unpack(header_format, data[1:header_total_length])
            description = description.decode()

            # print(f"Received image: {description} ({image_length} bytes)")

            # 提取图像数据
            image_data = data[header_total_length:]

            # 如果接收到的数据少于预期长度，继续接收直到足够
            while len(image_data) < image_length:
                more_data = await websocket.recv()
                image_data += more_data

            # 解析图片数据
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            # 显示图片
            if image is not None:
                cv2.imshow(description, image)
                key = cv2.waitKey(1)  # 短暂等待，处理GUI事件
                if key & 0xFF == ord('p'):
                    await websocket.send(" ")
                    break
                if key in control_keys_map:
                    ord_ = control_keys_map[key]
                    if ord_ != last_ord or key in [ord('8'), ord('2'), ord(' '), ord('4'), ord('6'), ord('5')]:
                        last_ord = ord_
                        print(f"Sending order: \"{ord_}\"")
                        await websocket.send(ord_)
            else:
                print("Failed to decode image")


if __name__ == "__main__":
    # WebSocket服务器地址
    uri = "ws://192.168.251.74:22335"

    # 运行客户端
    asyncio.run(receive_images(uri))
