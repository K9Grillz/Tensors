import os
import cv2
import tensorflow as tf
import pathlib

#import functions from other files
from fr_utils import *
from inception_blocks_v2 import *


def triplet_loss(y_true, y_pred, alpha=0.3): 
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss 


def create_model():
    global FRmodel
    #call the function to create a face recognition model
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    #compile the model
    FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
    #load weights for the model
    load_weights_from_FaceNet(FRmodel)


def check_face(face, model):
    #Get path to my image
    myimage = Path("/images/me.jpg")
    #Get the embeding of the input image
    testembed = img_to_embedding(face, model)
    #Get embed of my image
    myembed = img_path_to_encoding(myimage, model)
    #if they match, return true
    if(testembed == myembed):
        return True
    #otherwise false
    return False

def draw_verf():
    print("Yup it matches.")


def capture_face():
    global FRmodel
#Capture video with opencv
    cam = cv2.VideoCapture(0)
#read in the frame
    ret, frame = cam.read()
#convert frame to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#setup the cascade classifier
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#find all the faces in the image
    faces = faceCascade.detectMultiscale(frame)

#Lets pass the faces through the network
    for face in faces:
        if(check_face(face, FRmodel)):
            draw_verf(face)


def main():
    #Create the model 
    create_model()
    #Run forever
    while(True):
        capture_face()



main()
