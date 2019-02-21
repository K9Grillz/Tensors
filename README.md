# Tensors

The method used is based of this directory: https://github.com/Tony607/Keras_face_identification_realtime

The main two files are 'precompute' and 'face_identify' The other two files, 'flow' and 'webcam' are just test files.

To get the program working:

Satisfy the requirements
  1. Insert a video of yourself in /data/videos/{yourname} folder
  2. Run the 'precompute' program
  3. Run the 'face_identify_demo' program
  4. The program should be able to successfully identify your face
  
Note, there is a bug in one of the imported libraries. 
It will come up the first time 'precompute' is run. 
To fix this, follow the error to its source and replace 
keras.application.vggface ...... to 
keras_application.vggface

This should fix everything.
