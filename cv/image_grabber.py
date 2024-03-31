from concurrent.futures import ThreadPoolExecutor
import asyncio
import logging
from typing import Optional, Tuple

import numpy as np
import cv2


class AsyncCamera:
    """异步摄像头类，用于异步捕获视频帧。

    Attributes:
        task (Optional[asyncio.Task]): 异步任务，用于执行帧捕获。
        camera_id (int): 摄像头ID。
        cap (cv2.VideoCapture): OpenCV视频捕获对象。
        queue (asyncio.Queue): 存储最新帧的队列。
        running (bool): 控制帧捕获循环的运行状态。
        logger (logging.Logger): 日志记录器。
        executor (ThreadPoolExecutor): 线程池执行器，用于异步执行阻塞操作。
    """

    def __init__(self,
                 camera_id: int = 0,
                 loop: Optional[asyncio.AbstractEventLoop] = None
                 ) -> None:
        """初始化异步摄像头对象。

        Args:
            camera_id (int): 摄像头ID。
            loop (Optional[asyncio.AbstractEventLoop]): 事件循环对象。
        """
        self.task: Optional[asyncio.Task] = None
        self.camera_id: int = camera_id
        self.cap: Optional[cv2.VideoCapture] = None
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        self.running: bool = False
        self.logger: logging.Logger = logging.getLogger("re_RC2024.CV.AsyncCamera")
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
        self.loop: asyncio.AbstractEventLoop = loop or asyncio.get_event_loop()

    async def init(self) -> None:
        """
        初始化摄像头。
        """
        self.cap = await self.loop.run_in_executor(self.executor, cv2.VideoCapture, self.camera_id)
        if not self.cap.isOpened():
            self.logger.error("Failed to open camera")
            raise RuntimeError("Failed to open camera")
        self.logger.info(f"Camera {self.camera_id} opened")

    async def capture_frames(self) -> None:
        """异步捕获视频帧，并将其存储在队列中。"""
        loop = self.loop
        while self.running:
            ret, frame = await loop.run_in_executor(self.executor, self.cap.read)
            if not ret:
                self.logger.error("Failed to grab frame")
                break
            if not self.queue.empty():
                # 如果队列不为空，则清空队列
                await self.queue.get()
            await self.queue.put(frame)
            await asyncio.sleep(0)  # 让出控制权

    def is_opened(self) -> bool:
        """检查摄像头是否打开。

        Returns:
            bool: 摄像头是否打开。
        """
        return self.cap.isOpened()

    async def start(self) -> None:
        """启动视频帧捕获任务。"""
        if not self.cap:
            await self.init()
        self.running = True
        self.task = asyncio.create_task(self.capture_frames())

    async def stop(self) -> None:
        """停止视频帧捕获任务，并释放资源。"""
        if not self.cap:
            return
        self.running = False
        if self.task:
            await self.task
        self.cap.release()

    async def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """从队列中获取最新的视频帧。

        Returns:
            Tuple[bool, Optional[np.ndarray]]: 成功标志和帧数据。
        """
        return await self.queue.get()
