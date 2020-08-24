import cv2

#face classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Grab webcam feed
webcam = cv2.VideoCapture(0)

while True:
    # if frame is succesful
    successful_frame_read, frame = webcam.read()

    # if there's an error , abort
    if not successful_frame_read:
        break

    # change to grayscale
    frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 

    #show the current frame
    cv2.imshow('why so serious', frame_grayscale)

    cv2.waitKey(1)


webcam.release() 
cv2.destroyAllWindows()       

"""
#show the current frame
while True:

    # read current frame from webcam
    successful_frame_read, frame = webcam.read()

    cv2.imshow('why so serious', frame)

webcam.release() 
cv2.destroyAllWindows()   
"""

print("What's up")