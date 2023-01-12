import cv2
import numpy as np
import datetime
import pygame

# Initialize webcam
camera = cv2.VideoCapture(0)

# Create background subtractor object
background_subtractor = cv2.createBackgroundSubtractorMOG2()

# Set threshold for motion detection
threshold = 10
alarm_status = False

# Set motion threshold
motion_threshold = 2
min_area = 500

# Set time threshold (seconds)
time_threshold = 2

#initialize pygame mixer
pygame.init()
alarm_sound = pygame.mixer.Sound("alarm.wav")
alarm_channel = pygame.mixer.Channel(1)

def disarm_alarm(x):
    global alarm_status
    alarm_channel.stop()
    alarm_status = False
    print("Alarm Disarmed")

cv2.namedWindow("Alarm")
cv2.createTrackbar("Disarm", "Alarm", 0, 1, disarm_alarm)

def alarm_control(contours):
    global alarm_status
    # initialize variables to keep track of motion
    motion = 0
    total_area = 0
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            (x, y, w, h) = cv2.boundingRect(contour)
            total_area += w*h
            motion += 1
    if motion > motion_threshold and total_area > min_area:
        if not alarm_status:
            alarm_channel.play(alarm_sound)
            alarm_status = True
            print("Motion detected, alarm ON!")
    else:
        alarm_channel.stop()
        alarm_status = False
        print("No motion detected, alarm OFF")
    return alarm_status

# keep track of time when motion was last detected
last_motion_time = datetime.datetime.now()

# Loop over frames from the video stream
while True:
    # Read a frame from the video stream
    _, frame = camera.read()

    # Apply background subtraction to frame
    foreground_mask = background_subtractor.apply(frame)

    # threshold the mask to identify motion regions
    thresholded_mask = cv2.threshold(foreground_mask, threshold, 255, cv2.THRESH_BINARY)[1]

    # check if any contours in the mask
    contours, _ = cv2.findContours(thresholded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    alarm_status = alarm_control(contours)
    if alarm_status:
        last_motion_time = datetime.datetime.now()
    else:
        # check if alarm has been on for more than time_threshold seconds
        if (datetime.datetime.now() - last_motion_time).seconds > time_threshold:
            alarm_channel.stop()
            alarm_status = False
            print("Alarm OFF")

        # show alarm status on the frame
    if alarm_status:
        cv2.putText(frame, "Alarm ON!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Alarm OFF", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Camera", frame)
    cv2.imshow("Foreground", thresholded_mask)
    cv2.imshow("Alarm", np.zeros((50,640,3), np.uint8))

    # check if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release camera
camera.release()

# Close all windows
cv2.destroyAllWindows()

