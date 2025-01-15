# Obtain required libraries
from time import sleep

import serial

from lib_al5_2D_IK import al5_2D_IK, al5_moveMotors

# Constants - Speed in µs/s, 4000 is roughly equal to 360°/s or 60 RPM
#           - A lower speed will most likely be more useful in real use, such as 100 µs/s (~9°/s)
CST_SPEED_MAX = 4000
CST_SPEED_DEFAULT = 300

# Create and open a serial port
sp = serial.Serial()
sp.baudrate = 9600
sp.port = 'COM5'

sp.open()

# Set default values
AL5_DefaultPos = 1500;
cont = True
defaultTargetX = 4
defaultTargetY = 4
defaultTargetZ = 1
defaultTargetG = 30
defaultTargetWA = 0
defaultTargetShoulder = 90
defaultTargetElbow = 90
targetX = defaultTargetX
targetY = defaultTargetY
targetZ = defaultTargetZ
targetG = defaultTargetG
targetWA = defaultTargetWA
index_X = 0
index_Y = 1
index_Z = 2
index_G = 3
index_WA = 4
targetXYZGWAWR = (targetX, targetY, targetZ, targetG, targetWA)
targetQ = "y"
motors_SEWBZWrG = (90, 90, 90, 90, 90)
speed_SEWBZWrG = (
CST_SPEED_DEFAULT, CST_SPEED_DEFAULT, CST_SPEED_DEFAULT, CST_SPEED_DEFAULT, CST_SPEED_DEFAULT)

# Set the arm to default centered position (careful of sudden movement)
print("Default position is " + str(AL5_DefaultPos) + ".")
for i in range(0, 5):
    print(("#" + str(i) + " P" + str(AL5_DefaultPos) + "\r").encode())
    sp.write(("#" + str(i) + " P" + str(AL5_DefaultPos) + "\r").encode())


while cont:

    # Get X/Y position of end effector and perform IK on it
    print("")
    print("--- --- --- --- --- --- --- --- --- ---")
    print("< Set X/Y position of end effector >")
    print("")

    # Get X position
    targetInput = input("X position (last X = " + str(targetXYZGWAWR[index_X]) + ") ? ")
    if (targetInput == ""):
        targetX = targetXYZGWAWR[index_X];  # defaultTargetX;
    else:
        targetX = float(targetInput);
    targetXYZGWAWR = (
    targetX, targetXYZGWAWR[1], targetXYZGWAWR[2], targetXYZGWAWR[3], targetXYZGWAWR[4])

    # Get Y position
    targetInput = input("Y position (last Y = " + str(targetXYZGWAWR[index_Y]) + ") ? ")
    if (targetInput == ""):
        targetY = targetXYZGWAWR[index_Y];  # defaultTargetY;
    else:
        targetY = float(targetInput);
        if targetY <0.3:
            sp.write("#2 P1400 S300\r".encode())
            sp.write("#1 P1600 S300\r".encode())
    targetXYZGWAWR = (
    targetXYZGWAWR[0], targetY, targetXYZGWAWR[2], targetXYZGWAWR[3], targetXYZGWAWR[4])

    # Get Z position
    targetInput = input("Z position (last Z = " + str(targetXYZGWAWR[index_Z]) + ") ? ")
    if (targetInput == ""):
        targetZ = targetXYZGWAWR[index_Z];  # defaultTargetZ;
    else:
        targetZ = float(targetInput);
    targetXYZGWAWR = (
    targetXYZGWAWR[0], targetXYZGWAWR[1], targetZ, targetXYZGWAWR[3], targetXYZGWAWR[4])




    # Perform IK
    errorValue = al5_2D_IK(targetXYZGWAWR)
    if isinstance(errorValue, tuple):
        motors_SEWBZWrG = errorValue
    else:
        print(errorValue)
        motors_SEWBZWrG = (
        defaultTargetShoulder, defaultTargetElbow, defualtTargetWA, defaultTargetZ, defaultTargetG)

    # Move motors
    errorValue = al5_moveMotors(motors_SEWBZWrG, speed_SEWBZWrG, sp)


    # Quit? (quit on "y", continue on any other input)
    targetQ = str(input("Quit ? (Y/N) "))
    if targetQ == "y":
        cont = False
    else:
        sp.write("#4 P1100 S1000\r".encode())
        sleep(1)
        sp.write("#2 P1500 S400\r".encode())
        sp.write("#3 P1800 S300\r".encode())
        sp.write("#0 P700 S300\r".encode())
        sp.write("#1 P1400 S100\r".encode())

        sleep(3)

        sp.write("#2 P2000 S300\r".encode())
        sleep(3)
        sp.write("#4 P900 S300\r".encode())




# Set all motors to idle/unpowered (pulse = 0)
print("< Idling motors... >");
for i in range(0, 5):
    print(("#" + str(i) + " P" + str(0) + "\r").encode())
    sp.write(("#" + str(i) + " P" + str(0) + "\r").encode())
print("< Done >")