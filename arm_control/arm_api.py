import concurrent.futures
import asyncio
import logging
from typing import Optional

import oem_lib.Arm_Lib as Arm_Lib


class AsyncArm:
    _instance: Optional['AsyncArm'] = None
    _initialized: bool = False
    _default_angle: list[int] = [90, 110, 0, 0, 90, 0]

    def __new__(cls, *args, **kwargs) -> 'AsyncArm':
        if cls._instance is None:
            cls._instance = super(AsyncArm, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not self._initialized:
            self.logger: logging.Logger = logging.getLogger("re_RC2024.Arm.AsyncArm")
            self.executor: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self.arm: Optional[Arm_Lib.Arm_Device] = None
            self._initialized = True

    async def run_in_executor(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, lambda: func(*args))

    async def init(self, bus: int = 1) -> None:
        self.arm = await self.run_in_executor(Arm_Lib.Arm_Device, bus)
        self.logger.info(f"Arm initialized on bus {bus}")

    async def close(self) -> None:
        if self.arm is None:
            return
        await self.run_in_executor(self.arm.close)
        self.logger.info("Arm closed")

    async def set_all(
            self,
            angle1: int = 90,
            angle2: int = 110,
            angle3: int = 0,
            angle4: int = 0,
            angle5: int = 90,
            angle6: int = 0,
            time: int = 800
    ) -> None:
        if self.arm is None:
            raise ValueError("Arm not initialized")

        check_list = [angle1, angle2, angle3, angle4, angle5, angle6]
        for i in range(6):
            if not 0 <= check_list[i] <= 180 and i != 4:
                self.logger.error(f"Angle {i + 1} exceeds expectations({check_list[i]} of [0, 180])")
                check_list[i] = self._default_angle[i]
            if not 0 <= check_list[i] <= 270 and i == 4:
                self.logger.error(f"Angle 5 exceeds expectations({check_list[i]} of [0, 270])")
                check_list[i] = self._default_angle[i]

        await self.run_in_executor(
            self.arm.Arm_serial_servo_write6,
            angle1, angle2, angle3, angle4, angle5, angle6, time
        )

    async def lock(self) -> None:
        if self.arm is None:
            raise ValueError("Arm not initialized")
        self.logger.info("Arm locked")
        await self.run_in_executor(self.arm.Arm_serial_set_torque, 1)

    async def unlock(self) -> None:
        if self.arm is None:
            raise ValueError("Arm not initialized")
        self.logger.info("Arm unlocked")
        await self.run_in_executor(self.arm.Arm_serial_set_torque, 0)

    async def read(self, idx: int) -> int:
        if self.arm is None:
            raise ValueError("Arm not initialized")
        return await self.run_in_executor(self.arm.Arm_serial_servo_read, idx)

    async def ping(self, idx: int) -> int:
        if self.arm is None:
            raise ValueError("Arm not initialized")
        return await self.run_in_executor(self.arm.Arm_ping_servo, idx)

    async def write(self, idx: int, angle: int, time: int) -> None:
        if self.arm is None:
            raise ValueError("Arm not initialized")

        if not 0 <= angle <= 180 and idx != 5:
            self.logger.error(f"Angle {idx} exceeds expectations({angle} of [0, 180])")
            angle = self._default_angle[idx - 1]

        if not 0 <= angle <= 270 and idx == 5:
            self.logger.error(f"Angle 5 exceeds expectations({angle} of [0, 270])")
            angle = self._default_angle[idx - 1]

        await self.run_in_executor(self.arm.Arm_serial_servo_write, idx, angle, time)


if __name__ == '__main__':
    async def main():
        arm = AsyncArm()
        await arm.init()
        await arm.unlock()
        await asyncio.sleep(1)
        while True:
            for i in range(1, 7):
                print(f"Servo {i}: {await arm.read(i)}", end=" ")
            print()

    asyncio.run(main())
