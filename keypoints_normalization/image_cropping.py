from cv2 import cv2
import mediapipe as mp
import json
import glob
from pathlib import Path
import pprint as pp
from turtle import right
import numpy as np

class ImageCroppingNormalization():
    def __init__(self, holistic):
        self.holistic = holistic
        self.global_boundingbox_coord = None
        self.global_distance = None
    
    def normalize(self, image):
        image = self.cropping(image)
        data = self.extract_data(image)

        return data

    def cropping(self, image):
        mp_drawing = mp.solutions.drawing_utils
        # drawing = {"mp_drawing": mp_drawing}
        result = self.holistic.process(image)
        # if result.pose_landmarks == None:
        #     return []

        image = self.body_tracking(result.pose_landmarks, image)

        return image

    def drawing(self, image):
        mp_drawing = mp.solutions.drawing_utils
        drawing = {"mp_drawing": mp_drawing}
        result = self.holistic.process(image)
        drawing["image"] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        face_drawing = drawing.copy()
        pose_drawing = drawing.copy()
        left_hand_drawing = drawing.copy()
        right_hand_drawing = drawing.copy()
        face_drawing["landmarks"] = result.face_landmarks
        face_drawing["connections"] = self.holistic.FACEMESH_TESSELATION
        left_hand_drawing["landmarks"] = result.left_hand_landmarks
        right_hand_drawing["landmarks"] = result.right_hand_landmarks
        left_hand_drawing["connections"] = self.holistic.HAND_CONNECTIONS
        right_hand_drawing["connections"] = left_hand_drawing["connections"]
        pose_drawing["landmarks"] = result.pose_landmarks
        pose_drawing["connections"] = self.holistic.POSE_CONNECTIONS
        im = self.draw_landmarks(image, face_drawing)
        im = self.draw_landmarks(im, left_hand_drawing)
        im = self.draw_landmarks(im, right_hand_drawing)
        im = self.draw_landmarks(image, pose_drawing)

        return im

    def extract_data(self, image):
        result = self.holistic.process(image)
        face_landmarks = result.face_landmarks
        left_hand_landmarks = result.left_hand_landmarks
        right_hand_landmarks = result.right_hand_landmarks
        pose_landmarks = result.pose_landmarks
        mouth_coordinates = self.face_landmarks_data(face_landmarks)
        left_hand_coordinates = self.hand_landmarks_data(left_hand_landmarks)
        right_hand_coordinates = self.hand_landmarks_data(right_hand_landmarks)
        pose_coordinates = self.pose_landmarks_data(pose_landmarks)
        coordinates_collection = mouth_coordinates + left_hand_coordinates + right_hand_coordinates + pose_coordinates

        return coordinates_collection

    def face_landmarks_data(self, landmarks):
        coordinates = []
        if landmarks:
            for i in self.mouth_indices():
                landmark = landmarks.landmark[i]
                coordinates.append(landmark.x)
                coordinates.append(landmark.y)
                coordinates.append(landmark.z)
        else:
            for i in range(0, len(self.mouth_indices())):
                for i in range(0, 3):
                    coordinates.append(0)

        return coordinates

    def hand_landmarks_data(self, landmarks):
        coordinates = []
        if landmarks:
            for i in landmarks.landmark:
                coordinates.append(i.x)
                coordinates.append(i.y)
                coordinates.append(i.z)
        else:
            for i in range(0, 21):
                for i in range(0, 3):
                    coordinates.append(0)
        
        return coordinates

    def pose_landmarks_data(self, landmarks):
        coordinates = []
        if landmarks:
            for i in landmarks.landmark:
                coordinates.append(i.x)
                coordinates.append(i.y)
                coordinates.append(i.z)
        else:
            for i in range(0, 33):
                for i in range(0, 3):
                    coordinates.append(0)
        
        return coordinates

    def mouth_indices(self):
        return [0,13,14,17,37,39,40,61,78,80,81,82,84,87,88,91,95,146,178,181,185,191,267,269,270,291,308,310,311,312,314,317,318,321,324,375,402,405,409,415]

    def body_tracking(self, pose, image):
        centroid_indices = [0, 11, 12, 23, 24]
        im_h = image.shape[0]
        im_w = image.shape[1]
        if pose is None and self.global_boundingbox_coord is None:
            return image
        elif pose is None and self.global_boundingbox_coord is not None:
            x, y, z = self.global_boundingbox_coord
        else:
            x, y, z = self.find_body_centroid(pose, centroid_indices)

        distance = self.find_distance(pose, im_w, im_h)
        center_x = int(x*im_w)
        center_y = int(y*im_h)
        w, h = int(distance), int(distance)
        box_coord = [center_y-h, center_y+h, center_x-w, center_x+w]
        top, bottom, left, right = self.padding(box_coord, im_h, im_w)
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value=0)
        img_cpy = image.copy()
        return img_cpy[center_y - h + top:center_y + h + bottom + top, center_x-w + left:center_x + w + right + left]

    def padding(self, box_coord, im_h, im_w):
        paddings = np.zeros((4)).astype(dtype=np.int32)
        if box_coord[0] < 0:
            paddings[0] = abs(box_coord[0])
        if box_coord[1] > im_h:
            paddings[1] = abs(box_coord[1] - im_h)
        if box_coord[2] < 0:
            paddings[2] = abs(box_coord[2])
        if box_coord[3] > im_w:
            paddings[3] = abs(box_coord[3] - im_w)

        return paddings

    def find_body_centroid(self, landmarks, indices):
        main_body = indices
        if landmarks:
            x_bodies = []
            y_bodies = []
            z_bodies = []
            for i in main_body:
                x_bodies.append(landmarks.landmark[i].x)
                y_bodies.append(landmarks.landmark[i].y)
                z_bodies.append(landmarks.landmark[i].z)
            self.global_boundingbox_coord = [np.average(x_bodies), np.average(y_bodies), np.average(z_bodies)]
            return np.average(x_bodies), np.average(y_bodies), np.average(z_bodies)

    def find_distance(self, landmarks, im_w, im_h):
        indices_a = [11, 12]
        indices_b = [23, 24]
        centroid_a = np.array(self.find_body_centroid(landmarks, indices_a))
        centroid_b = np.array(self.find_body_centroid(landmarks, indices_b))
        point_a = self.pixel_coordinate_convertion(centroid_a, im_w, im_h)
        point_b = self.pixel_coordinate_convertion(centroid_b, im_w, im_h)
        # print(point_a, point_b)
        sum_sq = np.sum(np.square(point_a - point_b))
        euclidean = np.sqrt(sum_sq)
        return euclidean

    def pixel_coordinate_convertion(self, coordinates, w, h):
        x = int(coordinates[0] * w)
        y = int(coordinates[1] * h)
        coordinates[0] = x
        coordinates[1] = y

        return coordinates

    def draw_landmarks(self, image, params):
        mp_drawing = params["mp_drawing"]
        landmarks = params["landmarks"]
        connections = params["connections"]
        landmarks_drawing_spec = mp_drawing.DrawingSpec(
            color=[255, 0, 0],
            thickness=2,
            circle_radius=2,
        )
        connection_drawing_spec = mp_drawing.DrawingSpec(
            color=[0, 255, 00],
            thickness=1,
            circle_radius=2,
        )
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=landmarks,
            connections=connections,
            landmark_drawing_spec=landmarks_drawing_spec,
            connection_drawing_spec=connection_drawing_spec
        )

        return image