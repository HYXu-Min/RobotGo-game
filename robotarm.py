import serial

ser = serial.Serial()
ser.baudrate = 9600
ser.port = 'COM5'

ser.open()
# # set the arm to default centered position
ser.write("#0 P900\r".encode())
ser.write("#1 P1300\r".encode())
ser.write("#2 P2000\r".encode())
ser.write("#3 P900\r".encode())
ser.write("#4 P900\r".encode()) #1100，1400
