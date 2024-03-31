import asyncio
import logging

from rich.logging import RichHandler

from debug_bridge.img_ransfer import WebSocketServer
from cv.image_grabber import AsyncCamera
from mcu_bridge.motion_controller import Chassis
from arm_control.arm_api import AsyncArm


class Ch_ctl:
    def __init__(self, chassis: Chassis, arm: AsyncArm, speed: int = 20):
        self.chassis = chassis
        self.rudder2_angle = 180
        self.rudder1_angle = 90
        self.speed = speed
        self.control_keys_map = {
            "w": [speed, 0, 0],
            "s": [-speed, 0, 0],
            "a": [0, speed, 0],
            "d": [0, -speed, 0],
            "q": [0, 0, speed],
            "e": [0, 0, -speed],
            " ": [0, 0, 0]
        }
        self.Arm = arm

    async def ws_ord_ctl(self, order: str):
        if order in self.control_keys_map:
            await self.chassis.set_velocity(*self.control_keys_map[order])

        if order == "8" and self.rudder2_angle < 170:
            self.rudder2_angle += 5
            await self.Arm.write(2, self.rudder2_angle, 500)

        if order == "2" and self.rudder2_angle > 90:
            self.rudder2_angle -= 5
            await self.Arm.write(2, self.rudder2_angle, 500)

        if order == "4" and self.rudder1_angle < 170:
            self.rudder1_angle += 5
            await self.Arm.write(1, self.rudder1_angle, 500)

        if order == "6" and self.rudder1_angle > 10:
            self.rudder1_angle -= 5
            await self.Arm.write(1, self.rudder1_angle, 500)

        if order == "5":
            self.rudder1_angle = 90
            await self.Arm.write(2, 180, 500)

    async def close(self):
        await self.chassis.close_engine()


async def main(cam_num: int = 0, speed: int = 20):
    cam = AsyncCamera(cam_num)
    await cam.start()
    print("Camera started")

    arm = AsyncArm()
    await arm.init()
    print("Arm started")

    chassis = Chassis()
    await chassis.init_serial()
    print("Chassis started")
    await chassis.open_engine()

    ch_ctl = Ch_ctl(chassis, arm, speed)

    server = WebSocketServer(message_callback=ch_ctl.ws_ord_ctl, close_callback=ch_ctl.close)
    await server.start('0.0.0.0', 22335)

    while not server.status():
        await asyncio.sleep(0.1)
    print("WebSocket server started")

    try:
        while cam.cap.isOpened():
            frame = await cam.queue.get()
            await server.imshow("Camera0", frame)
            await asyncio.sleep(0)
    except KeyboardInterrupt:
        pass
    finally:
        await cam.stop()
        await server.stop()
        await chassis.close_engine()
        await chassis.close_serial()
        await arm.close()


if __name__ == "__main__":
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )

    cam_num = input("请输入摄像头id：")
    speed = input("请输入速度：")

    try:
        if 0 <= int(cam_num) <= 1 and 0 <= int(speed) <= 50:
            asyncio.run(main(int(cam_num), int(speed)))

    except KeyboardInterrupt:
        print("程序结束")
        pass
