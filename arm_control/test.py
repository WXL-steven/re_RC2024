import time

import oem_lib.Arm_Lib as Arm_Lib

Arm = Arm_Lib.Arm_Device()

Arm.Arm_serial_servo_write6(90, 180, 0, 0, 90, 0, 800)


Arm.Arm_serial_set_torque(0)
print("\n机械臂舵机力矩已关闭 可以掰动\n")
input("\n当机械臂掰动到位后请输按回车\n")
Arm.Arm_serial_set_torque(1)
print("\n机械臂舵机力矩已启动\n")
time.sleep(.2)
for i in range(6):
    aa = Arm.Arm_serial_servo_read(i + 1)
    print("第" + str(i + 1) + "号舵机角度为：", aa)
