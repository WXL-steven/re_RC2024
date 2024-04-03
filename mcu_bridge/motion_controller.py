import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import serial

from const import MCU_BRIDGE, SERIAL_ORD as ORD


class Chassis:
    _instance: Optional['Chassis'] = None  # 类变量，用于存储单例实例
    _initialized: bool = False  # 类变量，用于标识是否已经进行了初始化
    executor: ThreadPoolExecutor
    loop: asyncio.AbstractEventLoop

    def __new__(cls, loop: Optional[asyncio.AbstractEventLoop] = None, *args, **kwargs) -> 'Chassis':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        if not Chassis._initialized:
            self.serial_initialized: bool = False
            self.serial_port: Optional[serial.Serial] = None  # 用于存储串口连接的实例
            self._init_async_executor(loop)
            Chassis._initialized = True
            self.serial_logger: Optional[logging.Logger] = logging.getLogger("re_RC2024.MCUBridge.Serial")
            self.engine_logger: Optional[logging.Logger] = logging.getLogger("re_RC2024.MCUBridge.Engine")

    def _init_async_executor(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        self.executor = ThreadPoolExecutor(max_workers=MCU_BRIDGE.MAX_WORKERS)
        self.loop = loop or asyncio.get_event_loop()

    async def run_in_executor(self, func, *args) -> any:
        """在异步执行器中运行给定的同步函数"""
        return await self.loop.run_in_executor(self.executor, func, *args)

    async def init_serial(self,
                          port: str = MCU_BRIDGE.SERIAL_PORT,
                          baudrate: int = MCU_BRIDGE.SERIAL_BAUDRATE,
                          timeout: Optional[float] = None
                          ) -> None:
        """初始化串口"""
        # 使用run_in_executor来异步化同步的串口操作
        await self.run_in_executor(self._init_serial_sync, port, baudrate, timeout)

    def _init_serial_sync(self, port: str, baudrate: int, timeout: Optional[float]) -> None:
        """同步方式初始化串口，这个方法将在执行器中被调用"""
        try:
            self.serial_port = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
            self.serial_initialized = True
            self.serial_logger.info(f"Serial port {port} initialized successfully")
        except serial.SerialException as e:
            self.serial_logger.error(f"Failed to initialize serial port {port}: {e}")
            raise

    async def close_serial(self) -> None:
        """关闭串口"""
        await self.run_in_executor(self._close_serial_sync)
        self.serial_logger.info("Serial port closed")

    def _serial_write_sync(self, data: bytes) -> None:
        """同步方式向串口写入数据"""
        if self.serial_port is not None:
            self.serial_port.write(data)

    async def serial_write(self, data: bytes) -> None:
        """向串口写入数据"""
        await self.run_in_executor(self._serial_write_sync, data)
        self.serial_logger.debug(f"Data written to serial port: {data}")

    @staticmethod
    def _gen_body(data):
        """将输入的数据转为16进制的高8位低8位"""
        data_body = []
        for i in range(3):
            if data[i] > 127:
                data[i] = 127
            elif data[i] < -128:
                data[i] = -128
            data_body.append(int(data[i]) & 0xff)  # 低8位
        return data_body

    def _close_serial_sync(self) -> None:
        """同步方式关闭串口，这个方法将在执行器中被调用"""
        if self.serial_port is not None:
            self.serial_port.close()
            self.serial_initialized = False

    async def open_engine(self) -> None:
        """开启引擎"""
        data = bytes(ORD.HEAD + [ORD.ARM & 0xff] + [0x00, 0x00, 0x00] + ORD.TAIL)
        await self.serial_write(data)
        self.engine_logger.info("Engine Start")

    async def close_engine(self) -> None:
        """关闭引擎"""
        data = bytes(ORD.HEAD + [ORD.DISARM & 0xff] + [0x00, 0x00, 0x00] + ORD.TAIL)
        await self.serial_write(data)
        self.engine_logger.info("Engine Stop")

    async def set_velocity(self, x: int, y: int, rotation: int) -> None:
        """设置速度"""
        # 限制和转换速度值
        vel = [x, y, rotation]
        body = self._gen_body(vel)
        data = bytes(ORD.HEAD + [ORD.VEL_MODE & 0xff] + body + ORD.TAIL)
        await self.serial_write(data)
        self.engine_logger.debug(f"Velocity set to x: {x}/128, y: {y}/128, rotation: {rotation}/128")

    async def auto_line_following(self) -> None:
        """执行自动巡线（灰度模式）"""
        data = bytes(ORD.HEAD + [ORD.GRAY_MODE & 0xff] + [0x00, 0x00, 0x00] + ORD.TAIL)
        await self.serial_write(data)
        self.engine_logger.info("Auto line following started")


if __name__ == '__main__':
    from rich.logging import RichHandler

    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )

    chassis = Chassis()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(chassis.init_serial())
    loop.run_until_complete(chassis.open_engine())
    try:
        while True:
            ipt = input("输入速度以测试底盘控制（格式：x y rotation），输入q/回车退出：")
            if ipt in ['q', 'Q'] or ipt == '':
                break
            x, y, r = map(int, ipt.split())
            loop.run_until_complete(chassis.set_velocity(x, y, r))
    except Exception as e:
        print(e)
    finally:
        print("Shutting down...")
        loop.run_until_complete(chassis.close_engine())
        loop.run_until_complete(chassis.close_serial())
        loop.close()
        print("Test finished")
