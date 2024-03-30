import asyncio
import struct

import websockets
import cv2
import numpy as np


async def receive_images(uri):
    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket server")

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

            print(f"Received image: {description} ({image_length} bytes)")

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
                if key & 0xFF == ord('q'):
                    break
            else:
                print("Failed to decode image")


# WebSocket服务器地址
uri = "ws://localhost:22335"

# 运行客户端
asyncio.run(receive_images(uri))
