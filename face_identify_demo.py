from keras_vggface.vggface import VGGFace
from multiprocessing import Pool
from PIL import ImageTk, Image
from tkinter import ttk
import tkinter as tk
import numpy as np
import scipy as sp
import cv2
import pickle
import smtplib

#This program performs the actual facial recognition

def load_stuff(filename):
    saved_stuff = open(filename, "rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff


subject = "Safe Home"
msg = "Hello there, There maybe a possible new guest at your home."

def send_email(subject, msg):
        try:
            server = smtplib.SMTP('smtp.gmail.com:587')
            server.ehlo()
            server.starttls()
            server.login("edpNM06@gmail.com", "OMPRules7")
            message = 'Subject: {}\n\n {}'.format(subject, msg)
            server.sendmail("edpNM06@gmail.com", "edpNM06@gmail.com", message)
            server.quit()
            print("Success: Email sent!")
        except:
            print("Email failed to send.")


class FaceIdentify(object):
    """
    Singleton class for real time face identification
    """

    CASE_PATH = ".\\pretrained_models\\haarcascade_frontalface_alt.xml"

    def __new__(cls, precompute_features_file=None):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceIdentify, cls).__new__(cls)
        return cls.instance

    def __init__(self, precompute_features_file=None):
        self.face_size = 224
        self.precompute_features_map = load_stuff(precompute_features_file)
        print("Loading VGG Face model...")
        self.model = VGGFace(model='resnet50',
                             include_top=False,
                             input_shape=(224, 224, 3),
                             pooling='avg')  # pooling: None, avg or max
        print("Loading VGG Face model done")

    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    def crop_face(self, imgarray, section, margin=20, size=224):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w, h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w - 1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h - 1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)



    def identify_face(self, features, threshold=100):
        distances = []
        for person in self.precompute_features_map:
            person_features = person.get("features")
            distance = sp.spatial.distance.euclidean(person_features, features)
            distances.append(distance)
        min_distance_value = min(distances)
        min_distance_index = distances.index(min_distance_value)
        if min_distance_value < threshold:
            return self.precompute_features_map[min_distance_index].get("name") + ' ' + str(min_distance_value)
        else:
           #uncomment the following line for full functionality. However this will slow down the video application.
           # send_email(subject, msg)
            return "?"

    def detect_face(self):
        face_cascade = cv2.CascadeClassifier(self.CASE_PATH)

        # 0 means the default video capture device in OS
        video_capture = cv2.VideoCapture(0)

        # infinite loop, break by key ESC
        while True:
            if not video_capture.isOpened():
                sleep(5)
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=10,
                minSize=(64, 64)
            )
            # placeholder for cropped faces
            face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))

            # This can be made parallel. A thread for each face crop
            for i, face in enumerate(faces):
                face_img, cropped = self.crop_face(frame, face, margin=10, size=self.face_size)
                (x, y, w, h) = cropped
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                face_imgs[i, :, :, :] = face_img
                #This is where we would combine everything

            #It is possible to make this parallel but given how intense neural nets are,
            #probably not worth it
            if len(face_imgs) > 0:
                # generate features for each face
                features_faces = self.model.predict(face_imgs)
                #This part however, is perfect candidate for parallel
                predicted_names = [self.identify_face(features_face) for features_face in features_faces]
            # draw results
            #Again, another parallel option
            for i, face in enumerate(faces):
                label = "{}".format(predicted_names[i])
                self.draw_label(frame, (face[0], face[1]), label)

            cv2.imshow('Safe Home Camera', frame)  # shows the frame
            if cv2.waitKey(5) == 27:  # ESC key press
                break
        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()



def start():
    face = FaceIdentify(precompute_features_file="./data/precompute_features.pickle")
    face.detect_face()


def main():
    window = tk.Tk()
    #window.title("Join")
    window.geometry("800x630")
    window.configure(background='white')
    window.iconbitmap(default="SafeHomeIcon.ico")
    window.wm_title("Safe Home")

    path = "background5.jpg"

    #Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
    img = ImageTk.PhotoImage(Image.open(path))

    #The Label widget is a standard Tkinter widget used to display a text or image on the screen.
    panel = tk.Label(window, image = img)

    #The Pack geometry manager packs widgets in rows or columns.
    panel.pack(side = "top", fill = "both", expand = "yes")

    button1 = ttk.Button(window,text="Start Monitoring Now",command= start)
    button1.pack()

    #Start the GUI
    window.mainloop()

if __name__ == "__main__":
    main()


######
# Performance Analysis
# The 'init' is actually very light on system
# Capture frames takes up about 25% utilization on each core
# Actually identifying the frame uses 100% on all cores

# Even with just serial, it feels like the system is already maxed out, the gains
# to be had from making it parallel feel small.
