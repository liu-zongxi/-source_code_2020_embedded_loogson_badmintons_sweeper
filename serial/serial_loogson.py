import serial
import time
import cv2
class Cmdvel():
    def __init__(self, ser):
        self.ser = ser

    def int2bin16(self, num):
        return (bin(((1 << 16) - 1) & num)[2:]).zfill(16)

    def get_bcc(self, inputStr):
        bcc = 0
        for i in inputStr.split(' '):
            bcc = bcc ^ int(i, 16)
        bcc = self.int2bin16(bcc)
        bcc = str(bcc)
        bcc = hex(int(bcc, 2))
        bcc = bcc[2:].zfill(2)
        return str(bcc)

    def str_to_hex(self, str):
        return ' '.join([hex(ord(c)).replace('0x', '') for c in str])
    def __call__(self, v_x, v_y, v_z):
        v_x *= 1000
        v_x = self.int2bin16(int(v_x))
        v_x_send = str(v_x)
        v_x_send = hex(int(v_x_send, 2))
        v_x_send = v_x_send[2:].zfill(4)
        v_x_high = v_x_send[0:2]
        v_x_low = v_x_send[2:4]
        v_y *= 1000
        v_y = self.int2bin16(int(v_y))
        v_y_send = str(v_y)
        v_y_send = hex(int(v_y_send, 2))
        v_y_send = v_y_send[2:].zfill(4)
        v_y_high = v_y_send[0:2]
        v_y_low = v_y_send[2:4]
        v_z *= 1000
        v_z = self.int2bin16(int(v_z))
        v_z_send = str(v_z)
        v_z_send = hex(int(v_z_send, 2))
        v_z_send = v_z_send[2:].zfill(4)
        v_z_high = v_z_send[0:2]
        v_z_low = v_z_send[2:4]
        bcc_list = ['7B', '00', '00', v_x_high, v_x_low, v_y_high, v_y_low, v_z_high, v_z_low]
        bcc_input = ' '.join(bcc_list)
        bcc = self.get_bcc(bcc_input)
        print("start")
        self.ser.write(chr(0x7B).encode("utf-8"))
        self.ser.write(chr(0x00).encode("utf-8"))
        self.ser.write(chr(0x00).encode("utf-8"))
        self.ser.write(bytes.fromhex(v_x_high))
        self.ser.write(bytes.fromhex(v_x_low))
        self.ser.write(bytes.fromhex(v_y_high))
        self.ser.write(bytes.fromhex(v_y_low))
        self.ser.write(bytes.fromhex(v_z_high))
        self.ser.write(bytes.fromhex(v_z_low))
        self.ser.write(bytes.fromhex(bcc))
        self.ser.write(chr(0x7D).encode("utf-8"))
        print("end")
        return


portx = "/dev/ttyUSB0"
bps = "115200"
timex = 5
result1 = cv2.imread("/home/cjj/fake/1.png")
result2 = cv2.imread("/home/cjj/fake/2.png")
ser = serial.Serial(portx, bps, timeout=timex)
cmd_vel = Cmdvel(ser)
cv2.imshow("result", result1)
cv2.waitKey(3000)
cmd_vel(0.1, 0, 0)
time.sleep(5)
cmd_vel(0, -0.1,0)
time.sleep(2.5)
cmd_vel(0, 0, 0)
time.sleep(1)
cmd_vel(0, 0, 0.5)
time.sleep(3.2)
cmd_vel(0, 0, 0)
time.sleep(1.5)
cv2.imshow("result", result2)
cv2.waitKey(3000)
cmd_vel(0.1,0, 0)
time.sleep(5)
cmd_vel(0, -0.1, 0)
time.sleep(2.5)
cmd_vel(0, 0, 0)


ser.close()
