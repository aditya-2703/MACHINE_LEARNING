# importing usefull modules
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
import cv2
import dlib

# open vdo capturing stream
vdo = cv2.VideoCapture(0)

# initialize dlib face detector
detector = dlib.get_frontal_face_detector()

# face landmark from this file
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# loading data for training
data =np.load("mood_detect_data.npy")

# fetching features and lables
features = data[:,1:].astype(int)
lable = data[:,0]


# building model
model = KNeighborsClassifier()

# train model with these data
model.fit(features,lable)
        
# initialize the loop which capture multiple frames so it act like vdo
while True:
    # taking flag and image from this stream
    # flag is just boolean variable which return true if there is no problem with vdo capturing else false
    flag, image = vdo.read()
        
    # so if there is no problem with vdo streaming then will do our work
    if flag:
        # convert image to gray scale
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        
        # detecting face
        face = detector(image)

        # flatten all image frame
        flatten_img = gray.flatten()

        # getting prediction from model
        prediction = model.predict([flatten_img])

        # print(prediction)

        # show this prediction as text to window 
        cv2.putText(image,prediction[0],(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
        
        # showing the window         
        cv2.imshow("image",image)
        
        # for quite the windo        
        key = cv2.waitKey(1)
        if key==ord("q"):
            break

vdo.release()
cv2.destroyallwindows()
                