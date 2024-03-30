import asyncio
import struct
from typing import Set

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import websockets


class WebSocketServer:
    """
    ### 开发文档：图像数据传输结构

    本文档描述了通过WebSocket进行图像数据传输时使用的数据结构。该结构旨在高效地传输图像及其描述信息，以便接收方能够准确地解析和显示图像。

    #### 数据包结构

    每个数据包由两个主要部分组成：**头部**和**图像数据**。

    1. **头部**：包含图像描述文本的长度、图像描述文本本身，以及图像数据的长度。头部的结构设计如下：
        - **描述文本长度**：1字节，表示描述文本的长度（以字节为单位），允许的最大长度为255个字符。
        - **描述文本**：变长，根据描述文本长度字段指定的长度确定。描述文本采用UTF-8编码。
        - **图像数据长度**：4字节，无符号整数，表示图像数据的长度（以字节为单位），允许的最大长度为\(2^{32}-1\)字节。

    2. **图像数据**：变长，根据头部中的图像数据长度字段指定的长度确定。图像数据是经过编码的，编码方式根据实际应用可以灵活选择（如JPEG、PNG等）。

    #### 数据包示例

    假设有一个描述文本为"Example Image"的图像，其编码后的图像数据长度为1024字节。描述文本"Example Image"的UTF-8编码长度为13字节（因为它是ASCII文本，所以每个字符占用一个字节）。

    - **描述文本长度**：`0x0D`（13的十六进制表示）
    - **描述文本**：`45 78 61 6D 70 6C 65 20 49 6D 61 67 65`（"Example Image"的ASCII码）
    - **图像数据长度**：`00 00 04 00`（1024的十六进制表示，采用大端字节序）

    整个数据包的头部部分将会是这样的（以十六进制表示）：

    ```
    0D 45 78 61 6D 70 6C 65 20 49 6D 61 67 65 00 00 04 00
    ```

    紧跟着头部的是图像数据部分，长度为1024字节，具体内容取决于图像数据。

    #### 发送和接收流程

    - **发送方**需要按照上述结构构造数据包，首先将描述文本长度、描述文本和图像数据长度打包成头部，然后将头部和图像数据拼接起来形成完整的消息，最后通过WebSocket发送给接收方。
    - **接收方**收到数据包后，首先解析头部，提取描述文本长度、描述文本和图像数据长度，然后根据图像数据长度读取相应长度的图像数据。接收方可以根据描述文本和图像数据展示或处理图像。

    #### 注意事项

    - 描述文本长度限制为255个字符，这意味着描述文本必须在255个字符以内。
    - 图像数据长度字段允许的最大长度为(2^{32}-1)字节，这应足够包含任何合理大小的图像数据。
    - 由于使用了UTF-8编码，描述文本可以包含任意Unicode字符，但发送方和接收方都需要确保正确处理UTF-8编码。
    """
    _instance = None
    _initialized = False  # 用于检查实例是否已初始化

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(WebSocketServer, cls).__new__(cls)
        return cls._instance

    def __init__(self, max_edge_length: int = 512, jpg_quality: int = 90) -> None:
        if self._initialized:  # 如果已经初始化过，直接返回
            return
        self.server: websockets.server = None
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.max_edge_length: int = max_edge_length
        self.jpg_quality: int = jpg_quality
        self._initialized = True  # 标记为已初始化

    async def _register(self, websocket: websockets.WebSocketServerProtocol) -> None:
        self.clients.add(websocket)

    async def _unregister(self, websocket: websockets.WebSocketServerProtocol) -> None:
        self.clients.remove(websocket)

    async def handler(self, websocket: websockets.WebSocketServerProtocol) -> None:
        await self._register(websocket)
        try:
            async for message in websocket:
                print(f"Received message from client: {message}")
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"Connection closed with error: {e}")
        finally:
            await self._unregister(websocket)

    async def _task(self) -> None:
        try:
            await self.server.wait_closed()
        except (asyncio.CancelledError, asyncio.TimeoutError, KeyboardInterrupt):
            await self.stop()

    async def start(self, host: str = '0.0.0.0', port: int = 22335) -> asyncio.Task:
        print(f"WebSocket Server started at ws://{host}:{port}")
        self.server = await websockets.serve(self.handler, host, port)
        task: asyncio.Task = asyncio.create_task(self._task())
        return task

    def status(self) -> bool:
        return self.server.is_serving()

    async def stop(self) -> None:
        self.server.close()
        await self.server.wait_closed()

    async def _broadcast(self, data: bytes) -> None:
        if self.clients:
            await asyncio.wait([self._safe_send(client, data) for client in self.clients])

    @staticmethod
    async def _safe_send(client: websockets.WebSocketServerProtocol, data: bytes) -> None:
        try:
            await client.send(data)
        except websockets.exceptions.ConnectionClosed:
            pass

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        scale: float = self.max_edge_length / max(h, w)
        if scale < 1:
            new_size = (int(w * scale), int(h * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return image

    def _encode_image(self, image: np.ndarray) -> bytes:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_quality]
        result, encimg = cv2.imencode('.jpg', image, encode_param)
        if result:
            return encimg.tobytes()
        else:
            raise ValueError("Failed to encode image")

    async def imshow(self, description: str, image: np.ndarray) -> None:
        if self.server is None or not self.server.is_serving():
            return
        if len(description) > 254:
            print("Description length exceeds 255 characters")
            return
        resized_image = self._resize_image(image)
        encoded_image = self._encode_image(resized_image)
        description_bytes = description.encode()
        description_length = len(description_bytes)
        header = struct.pack(f"!B{description_length}sI", description_length, description_bytes, len(encoded_image))
        message = header + encoded_image
        await self._broadcast(message)


# 主函数
async def main():
    server = WebSocketServer()
    executor = ThreadPoolExecutor()
    server_task = await server.start('localhost', 22335)

    while not server.status():
        await asyncio.sleep(0.1)

    print("WebSocket server started")

    cap = await asyncio.get_event_loop().run_in_executor(executor, cv2.VideoCapture, 0)
    print("Camera started")
    while cap.isOpened():
        ret, frame = await asyncio.get_event_loop().run_in_executor(executor, cap.read)
        if not ret:
            break
        await server.imshow("Camera0", frame)
        await server.imshow("Camera1", frame)
        await asyncio.sleep(0.1)

    await server_task  # 等待服务器任务结束


# 运行服务器
asyncio.run(main())
