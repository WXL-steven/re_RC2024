import asyncio
import logging

from rich.logging import RichHandler

from cv.line_follower import AutoPilot
from debug_bridge.img_ransfer import WebSocketServer
from cv.image_grabber import AsyncCamera
from mcu_bridge.motion_controller import Chassis
from arm_control.arm_api import AsyncArm


class Ch_ctl:
    def __init__(
            self,
            chassis: Chassis,
            arm: AsyncArm,
            speed: int = 20,
            fix_speed: int = 10,
            switch_mod: callable = None
    ):
        self.chassis = chassis
        self.rudder2_angle = 110
        self.rudder1_angle = 90
        self.speed = speed
        self.fix_speed = fix_speed
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
        self.switch_mod = switch_mod

        self.ap = False

    def set_switch_mode(self, switch_mod: callable):
        self.switch_mod = switch_mod

    async def ws_ord_ctl(self, order: str):
        if order == "ap":
            self.ap = True
        elif order in ['z', 'x', 'c']:
            pass
        else:
            self.ap = False

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
            await self.Arm.write(1, self.rudder1_angle, 500)
            print(f"a2: {self.rudder2_angle}")

        if order == "z" and self.switch_mod is not None:
            self.switch_mod("left")

        if order == "x" and self.switch_mod is not None:
            self.switch_mod("center")

        if order == "c" and self.switch_mod is not None:
            self.switch_mod("right")

    async def close(self):
        self.ap = False
        await self.chassis.set_velocity(0, 0, 0)

    async def auto_pilot(self, dist, angle):
        if not self.ap:
            return
        if dist is not None and angle is not None and self.ap:
            if abs(dist) > 20:
                horizontal_speed = (-1 if dist < 0 else 1)*self.fix_speed
            else:
                horizontal_speed = 0

            if angle != 0:
                rotation_speed = (-1 if angle < 0 else 1)*self.fix_speed
            else:
                rotation_speed = 0

            await self.chassis.set_velocity(self.speed, horizontal_speed, rotation_speed)


async def main(cam_num: int = 0, speed: int = 20, debug: bool = False):
    logger = logging.getLogger("re_RC2024.Playground")

    cam = AsyncCamera(cam_num)
    await cam.start()
    logger.info("Camera started")
    await asyncio.sleep(.1)

    arm = AsyncArm()
    await arm.init()
    await asyncio.sleep(.1)
    await arm.lock()
    logger.info("Arm started")
    await arm.set_all()

    await asyncio.sleep(.1)

    chassis = Chassis()
    await chassis.init_serial()
    await asyncio.sleep(.1)
    logger.info("Chassis started")
    await chassis.open_engine()

    ch_ctl = Ch_ctl(chassis, arm, speed)

    server = WebSocketServer(message_callback=ch_ctl.ws_ord_ctl, close_callback=ch_ctl.close, max_edge_length=640)
    await server.start('0.0.0.0', 22335)

    while not server.status():
        await asyncio.sleep(0.1)
    logger.info("WebSocket server started")

    cv_playground = AutoPilot(
        debug=True,
        imshow=server.imshow,
        mode="left",
    )

    ch_ctl.set_switch_mode(cv_playground.switch_mode)

    windows_name = "Dataset" if debug else "Playground"
    try:
        while cam.cap.isOpened():
            frame = await cam.queue.get()
            if frame is None:
                logger.warning("Frame is None")
            await server.imshow(windows_name, frame)
            dist, angle = await cv_playground.get_correction(frame)
            if dist is not None and angle is not None:
                await ch_ctl.auto_pilot(dist, angle)
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
    if cam_num == "":
        cam_num = 0

    speed = input("请输入速度：")
    if speed == "":
        speed = 20

    debug = input("是否开启调试模式？(y/[n])")
    if debug == "y":
        debug_flag = True
    else:
        debug_flag = False

    try:
        if 0 <= int(cam_num) <= 1 and 0 <= int(speed) <= 50:
            print("=============== Program Start ===============")
            asyncio.run(main(int(cam_num), int(speed), debug_flag))

    except KeyboardInterrupt:
        print("程序结束")
        pass
