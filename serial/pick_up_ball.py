import serial_test as VC
import serial
import time

def pick_up_ball():
    portx = "COM5"
    bps = "115200"
    timex = 5
    ser = serial.Serial(portx, bps, timeout=timex)
    cmd_vel = VC.Cmdvel(ser)
    cmd_vel(0.5, 0, 0)
    time.sleep(2)
    cmd_vel(0, 0, 0.5)
    time.sleep(2)
    cmd_vel(0, 0.5, 0)
    time.sleep(2)
    cmd_vel(0, 0, 0)
    ser.close()
if __name__ == '__main__':
    pick_up_ball()