import tensorflow as tf
import numpy as np
import cv2

cam = cv2.VideoCapture(0)

while(True):
    ret, frame = cam.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    #Move to next frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#Now find the face in the frame, if it exists.

#Run the frame through the neural net




# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()




