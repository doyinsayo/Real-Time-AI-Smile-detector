import cv2

#face classifier
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


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

    # detect faces
    faces = face_detector.detectMultiScale(frame_grayscale)
    
    #print(faces)

    # run smile detection for each of those faces

    for (x,y,w,h) in faces:
        # draw a rectangle around the face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(100,250,50),4)
        
        # get the sub frame (using numpy N-dimensional array slicing)
        the_face = frame[y:y+h , x:x+w]
        
        # change to grayscale
        face_grayscale = cv2.cvtColor(the_face,cv2.COLOR_BGR2GRAY)

        smiles = smile_detector.detectMultiScale(face_grayscale,scaleFactor=1.7,
          minNeighbors=20)

        #  Find all smiles in the face
        """for (x_,y_,w_,h_) in smiles:
            # draw a rectangle around the smile
            cv2.rectangle(the_face ,(x_,y_),(x_+w_,y_+h_),(50,50,200),4)
            """
                 
        # label the face as smiling         
        if len(smiles) > 0:
            cv2.putText(frame,'smiling',(x,y+h+40), fontScale=3,
            fontFace=cv2.FONT_HERSHEY_PLAIN,color=(255,255,255))
        
    

    #show the current frame
    cv2.imshow('Smile detector', frame)

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