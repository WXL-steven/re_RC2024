from typing import Optional


class MCU_BRIDGE:
    """
    MCU_BRIDGE是一个用于存储与下位机通讯相关的常量和配置的类。这些常量和配置包括串口号、波特率、超时设置等。
    """
    MAX_WORKERS: int = 1  # 最大工作线程数
    SERIAL_PORT: str = "/dev/ttyUSB0"  # 串口号
    SERIAL_BAUDRATE: int = 9600  # 串口波特率
    SERIAL_TIMEOUT: Optional[float] = None  # 串口超时设置，单位秒。这里给出一个示例值，None表示永远等待


class SERIAL_ORD:
    HEAD: list[int] = [0xa5]
    TAIL: list[int] = [0x00, 0x5a]
    DISARM: int = 0
    ARM: int = 1
    VEL_MODE: int = 2
    GRAY_MODE: int = 3
