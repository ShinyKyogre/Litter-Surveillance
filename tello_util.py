import cv2 as cv
from djitellopy import tello
import numpy as np
import argparse
from matplotlib import pyplot as plt

# initializing tello | may put in main file
tello_instance = tello.Tello()
tello_instance.connect()
print(tello_instance.get_battery())
tello_instance.streamon()

object_count = 0
# Will store coordinates and count for each detection.
litters = [[]]

def log_litter(arr2d, num, dist):
    if num == 0: 
        return
    arr2d.append([dist, num])
    

def record_litter(distance, amount):
    return

def initialize_model():
    #weights_path = 
    #obj_path = 
    net = cv.dnn.readNet("C:/Users/sgupt/Desktop/CS 22-23/Shaun Projects/litterbug/model/litterbug.weights", "C:/Users/sgupt/Desktop/CS 22-23/Shaun Projects/litterbug/model/litterbug.cfg")
    with open("C:/Users/sgupt/Desktop/CS 22-23/Shaun Projects/litterbug/model/obj.names", "r") as f:
        # Gets every class, however we only have one
        classes = [line.strip() for line in f.readlines()] 
    output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
    colors = np.random.uniform(0, 255, size = (len(classes), 3))
    return net, classes, colors, output_layers

def display_blob(blob):
    for b in blob:
        for n, imgb in enumerate(b):
            cv.imshow(str(n), imgb)
            
# This is where we forward propogate to get the detection.
def detect(img, net, output_layers):
    blob = cv.dnn.blobFromImage(img, scalefactor=0.00392, size = (320, 320), mean = (0, 0, 0), swapRB = True, crop = False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    # Get the amount of current detections on the screen
    count = len(outputs)
    return blob, outputs

# Could be some issues here with terrible naming technique
def get_box_dimensions(outputs, height, width):
    boxes = []
    confidences = []
    classids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]
            if confidence > 0.3:
                center_x = int(detect[0]*width)
                center_y = int(detect[1]*height)
                w = int(detect[2]*width)
                h = int(detect[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classids.append(classid)
    return boxes, confidences, classids
                
def draw_labels(boxes, confidences, colors, classids, classes, img):
    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv.FONT_HERSHEY_SIMPLEX
    for i in range (len(boxes)):
        for i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[classids[i]])
            color = (255, 0, 0)
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv.putText(img, label, (x, y-5), font, 1, color, 1)
    cv.imshow("Image", img)

# Don't use this functiom - using videocap is extremely slow
def tello_detect_videocap():
    model, classes, colors, output_layers = initialize_model()
    cap = tello_instance.get_video_cap()
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect(frame, model, output_layers)
        boxes, confidences, classids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confidences, colors, classids, classes, frame)
        print("Count " + object_count)
        print("Objects " + outputs)
        key = cv.waitKey(1)
        if key == 27:
            break
        cap.release
        
# Should hopefully be faster and I can apply a frame skip. UPdate: it is definitely faster
def tello_detect_frame():
    model, classes, colors, output_layers = initialize_model()
    while True:
        
        frame_read = tello_instance.get_frame_read()
        frame = frame_read.frame
        height, width, channels = frame.shape
        blob, outputs = detect(frame, model, output_layers)
        boxes, confidences, classids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confidences, colors, classids, classes, frame)
        print("Net Distance: " + str(tello_instance.get_distance_tof()))
        print("Count: " + str(len(confidences)))
        log_litter(litters, len(confidences), tello_instance.get_distance_tof())
        print(litters)
        key = cv.waitKey(1)
        if key == 27:
            tello_instance.land()
            plot_litter(litters)
            log_coordinates("litters.txt", litters)
            break
    
# Log all coordinates to a txt file.
def log_coordinates(filename, arr2d):
    f = open(filename, 'x')
    for i in range(len(arr2d)):
        if i == 0:
            continue
        f.write(str(arr2d[i][0]) + " : " + str(arr2d[i][1]) + "\n")
    f.close()
    
def plot_litter(arr2d):
    coordinates = []
    litter_count = []
    
    for i in range(len(arr2d)):
        if i == 0:
            i = 1
        if coordinates.count(arr2d[i][0]) == 0:
             coordinates.append(arr2d[i][0])
             litter_count.append(arr2d[i][1])
                 
    plt.scatter(coordinates, litter_count)
    plt.savefig("plot.png")
       
     
