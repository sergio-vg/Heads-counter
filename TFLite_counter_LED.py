# Import packages
from clibs.centroidtracker import CentroidTracker
from clibs.trackableobject import TrackableObject
from clibs.VideoStream import VideoStream
from clibs.del_folder import rmFolder
from clibs.gsm import sim
import dlib
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
from datetime import datetime
import RPi.GPIO as GPIO 
from concurrent.futures import ThreadPoolExecutor

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    default = 'model')
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x480')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')
parser.add_argument('--max-distance',help="maximum distance of centroid movement per frame",
                     type=int, default=260)
parser.add_argument('--max-lost', help="maximum frames of object dissapeared",
                     type=int, default=10)
parser.add_argument('--skip-frames', help="# of skip frames between detections",
                    type=int, default=1)

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'   

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

#Variable parameter
tinfo = 2 #Time to send info in minutes
DeviceID = "0000"
APN = '"internet.itelcel.com","webgprs","webgprs2002"'
MQTTHost = "soldier.cloudmqtt.com"
MQTTPort = "10463"
MQTTUsername = "bulyartj"
MQTTPassword = "Y0TjhD7Mco7c"
MQTTTopic = "try"


# Variables of the counting lines orientation
#LimOr = args.way
w = None
dir1 = "Entrada"
dir2 = "Salida"

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared = args.max_lost,maxDistance = args.max_distance) 
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDir1 = 0
totalDir2 = 0
totalDir1b = 0
totalDir2b = 0
c1 = 0
c2 = 0
c3 = 0
packetid = 0

#flags
sf = 70 #time flag(to avoid print several times the same time value due the miliseconds value)
tf = 70

#Set up videostream
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream and set video writer
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()


#Create folder where will allocate the saved images and delete old folders
now = datetime.now()
fname = str(now.strftime("%d-%m-%Y"))

path0 ="/home/pi/Desktop/images/image_detection/"
path1 = "/home/pi/Desktop/images/image/"
dirName0 = path0 + fname
dirName1 = path1 + fname

if not os.path.exists(dirName0):
    os.mkdir(dirName0)
    print("Directory " , dirName0 ,  " Created ")
else:    
    print("Directory " , dirName0 ,  " already exists")

if not os.path.exists(dirName1):
    os.mkdir(dirName1)
    print("Directory " , dirName1 ,  " Created ")
else:    
    print("Directory " , dirName1 ,  " already exists")
    
remove = rmFolder(dir1 = path0, dir2 = path1)
remove.remove()

count = 0 #counter of image enumeration

#Setup GPIO pins
GPIO.setmode(GPIO.BCM)     # set up BCM GPIO numbering  
GPIO.setup(25, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)    # Orientation selector (slide switch)
GPIO.setup(24, GPIO.OUT)  #Running indicator LED

#Set the counting orientation with a fisical switch
if GPIO.input(25):
    LimOr = "h"
else:
    LimOr = "v"
    
#turn on the running indicator LED
GPIO.output(24, 1)

#GSM modulue
GSM = sim(DeviceID, APN, MQTTHost, MQTTPort, MQTTUsername, MQTTPassword, MQTTTopic)
GSM.sleep()

executor = ThreadPoolExecutor(max_workers = 4)

def Count_blink():
    GPIO.output(24, 0)
    time.sleep(.25)
    GPIO.output(24, 1)


while(True):

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    framec = frame1.copy()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)


    # box rectangles returned by either (1) our object detector or
    # (2) the correlation trackers
    status = "Esperando"
    rects = []
    

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):

        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            # set the status
            status = "Detectando"

            # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
                
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 255, 0), 2)

            # construct a dlib rectangle object from the bounding
            # box coordinates and then start the dlib correlation
            # tracker
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(xmin, ymin, xmax, ymax)
            tracker.start_track(frame, rect)

            # add the bounding box coordinates to the rectangles list
            rects.append((xmin, ymin, xmax, ymax))


    # draw two vertical/horizontal lines in the center of the frame and other two in the borders -- once an
    # object crosses these center llines we will determine whether they were
    # moving 'up'/'right' or 'down'/'left', and once it crosses the border line, can be deregister
    if(LimOr == "h"):
        p1 = 0, int(imH/2)-50
        p2 = int(imW), int(imH/2)-50
        p3 = 0, int(imH/2)+50
        p4 = int(imW), int(imH/2)+50

        p5 = 0, 50
        p6 = int(imW), 50
        p7 = 0, int(imH)-50
        p8 = int(imW), int(imH)-50

        p9 = 0, int(imH/2)
        p10 = int(imW), int(imH/2)

        # orientation variables
        w = 1

    elif(LimOr == "v"):
        p1 = int(imW/2)-50, 0
        p2 = int(imW/2)-50, int(imH)
        p3 = int(imW/2)+50, 0
        p4 = int(imW/2)+50, int(imH)

        p5 = 80, 0
        p6 = 80, int(imH)
        p7 = int(imW)-80, 0
        p8 = int(imW)-80, int(imH)

        p9 = int(imW/2), 0
        p10 = int(imW/2), int(imH)

        # orientation variables
        w = 0

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)
        
        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)
        
        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving 
            y = [c[w] for c in to.centroids]
            direction = centroid[w] - np.mean(y)
            to.centroids.append(centroid)
    
            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is positive (indicating the object
                # is moving right/up) AND the centroid is above right limit
                # line, count the object
                if direction > 0 and centroid[w] > p3[w]: #and centroid[w] <= p7[w]:
                    totalDir1 += 1
                    to.counted = True
                    executor.submit(Count_blink)

                # if the direction is negative (indicating the object
                # is moving left/down) AND the centroid is below the left limit
                # line, count the object
                elif direction < 0 and centroid[w] < p1[w]: #and centroid[w] >= p5[w]:
                    totalDir2 += 1
                    to.counted = True
                    executor.submit(Count_blink)
            # If the object was already counted and it passed the counting limits, 
            # will be deleated to no assign the same id to a new object
            elif (to.counted == True and ct.disappeared[objectID] >= 1) and ((direction < 0 and centroid[w] < p1[w]) or (direction > 0 and centroid[w] > p3[w])):
                ct.disappeared[objectID] = args.max_lost

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to
    
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        
    # construct a tuple of information we will be displaying on the
    # frame
    info = [
    (dir1, totalDir1),
    (dir2, totalDir2),
    ("Status", status),
    ]    
    
    now = datetime.now()

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (20, int(imH) - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (20, int(imH) - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw framerate and time in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(imW-120,imH-20),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame,str(now.strftime("%d/%m/%Y %H:%M:%S")),(20,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(imW-120,imH-20),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(frame,str(now.strftime("%d/%m/%Y %H:%M:%S")),(20,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
        
    cv2.putText(framec,str(now.strftime("%d/%m/%Y %H:%M:%S")),(20,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(framec,str(now.strftime("%d/%m/%Y %H:%M:%S")),(20,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
        
    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    #increment the count for image enumeration
    while os.path.exists(dirName1+f"/{count:08d}.jpg"):
        count += 1


    #Draw the lines, the border a center lines for visualization and de midle one for saved frames
    framed=frame.copy()

    cv2.line(frame, p1, p2, (0, 0, 255), 2)
    cv2.line(frame, p3, p4, (0, 0, 255), 2)

    cv2.line(frame, p5, p6, (255, 0, 0), 2)
    cv2.line(frame, p7, p8, (255, 0, 0), 2)

    cv2.line(framed, p9, p10, (0, 255, 0), 2)
    
    #Get second value
    s=int(now.strftime("%S"))
    t=int(now.strftime("%M")) 


    if (t%tinfo == 0) and(t != tf): 
        c1 = totalDir1 - totalDir1b
        c2 = totalDir2 - totalDir2b
        c3 += c1 - c2

        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M")
        infoPrint = [dt_string, "{}: {}".format(dir1,c1), "{}: {}".format(dir2,c2), "Pasajeros: {}".format(c3), "Total {}: {}".format(dir1,totalDir1), "Total {}: {}".format(dir2,totalDir2)]
        infoSend = str(DeviceID) + ";" + dt_string + ";" + str(c1) + ";" + str(c2) + ";" + str(c3) + ";" + str(totalDir1) + ";" + str(totalDir2)
        print(infoPrint[0]) #Hora
        print(infoPrint[1]) #entrada
        print(infoPrint[2]) #salida
        print(infoPrint[3]) #pasajeros
        print(infoPrint[4]) #Total entrada
        print(infoPrint[5]) #Total salida
        print("\n")

        mqttThread = Thread(name = "hilo_1", target = GSM.mqttPublish, args = (infoSend, packetid))
        mqttThread.start()

        totalDir1b = totalDir1
        totalDir2b = totalDir2

        tf = t 
        packetid += 1       

        

    #if(s != sf):
    # When run the script at boot the frame can't be displayed, so just will save the frames each second 
    cv2.imwrite(dirName0+f"/{count:08d}.jpg",framed)
    cv2.imwrite(dirName1+f"/{count:08d}.jpg",framec)
    sf = s


    # increment the total number of frames processed thus far and
    # then update the FPS counter
    totalFrames += 1   

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
videostream.stop()

cv2.destroyAllWindows()

# Turn Off LED
GPIO.output(24, 0)




