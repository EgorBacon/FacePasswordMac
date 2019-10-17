#Face Training
import cv2
import os
import numpy as np

def collect_dataset():
    images = []
    labels = []
    labels_dic = {}
    people = [person for person in os.listdir("Faces/")]
    for i, person in enumerate(people):
            labels_dic[i] = person
            for image in os.listdir("Faces/" + person):
                    images.append(cv2.imread("Faces/" + person + '/' + image, 0))
                    labels.append(person)
       
    return (images, np.array(labels), labels_dic)

images, labels, labels_dic = collect_dataset()
class FaceDetector(object):
    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)
    
    def detect(self, image, biggest_only=True):
        scale_factor = 1.2
        min_neighbors = 5
        min_size = (30, 30)
        biggest_only = True
        faces_coord = self.classifier.detectMultiScale(image,
                                                       scaleFactor=scale_factor,
                                                       minNeighbors=min_neighbors,
                                                       minSize=min_size,
                                                       flags=cv2.CASCADE_SCALE_IMAGE)
        return faces_coord

def cut_faces(image, faces_coord):
    faces = []
    
    for (x, y, w, h) in faces_coord:
        w_rm = int(0.3 * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])
         
    return faces

def resize(images, size=(224, 224)):
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size, 
                                    interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, 
                                    interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)

    return images_norm



def normalize_faces(image, faces_coord):

    faces = cut_faces(image, faces_coord)
    faces = resize(faces)
    return faces
count = 0

dir_path = os.path.dirname(os.path.realpath(__file__))
Unix_path_detector = str(dir_path) + "\\data\\" + "haarcascade_frontalface_default.xml"
mac_path_detector = str(dir_path) + "/data/" + "haarcascade_frontalface_default.xml"
for image in images:
        dir_path
        detector = FaceDetector("""{put in here your variable (Unix/not Unix)}""") 
        faces_coord = detector.detect(image, True)
        faces = normalize_faces(image ,faces_coord)
        for i, face in enumerate(faces):
                cv2.imwrite('%s.jpeg' % (count), faces[i])
                count += 1    

#Comments
"""str(dir_path) + "data/" + "haarcascade_frontalface_default.xml"""
"""C:\\Users\\MiniM\\Documents\\Github\\FacePasswordMac\\data\\haarcascade_frontalface_default.xml"""