from cv2 import cv2
import mediapipe as mp
import json
import glob
from pathlib import Path
import pprint as pp
from turtle import right
import numpy as np

class RecalculateNormalization():
    def __init__(self, holistic):
        self.holistic = holistic
        self.global_boundingbox_coord = None
        self.global_distance = None
    
    def normalize(self, image):
        result = self.holistic.process(image)
        params = { "pose_landmarks": result.pose_landmarks, "image": image, "result": result }
        params = self.body_tracking(params)
        coordinates = self.extract_data(params)
        return coordinates


    def extract_data(self, params):
        result = params["result"]
        face_landmarks = result.face_landmarks
        left_hand_landmarks = result.left_hand_landmarks
        right_hand_landmarks = result.right_hand_landmarks
        pose_landmarks = params["pose_landmarks"]
        shoulders_centroid = params["shoulders_centroid"]
        hips_centroid = params["hips_centroid"]
        image = params["image"]
        im_h = image.shape[0]
        im_w = image.shape[1]
        point_a = self.pixel_coordinate_convertion(shoulders_centroid.copy(), im_w, im_h)
        point_b = self.pixel_coordinate_convertion(hips_centroid.copy(), im_w, im_h)
        px_radius = int(self.euclidean(point_a, point_b))
        params["px_radius"] = px_radius
        if params["radius"] == 0:
            params["px_radius_multiplier"] = 0
        else:
            params["px_radius_multiplier"] = px_radius / params["radius"]
        if px_radius == 0:
            img = np.zeros((im_w, im_h, 3), dtype = np.uint8)
        else:
            size = (px_radius * 2, px_radius * 2, 3)
            img = np.zeros(size, dtype = np.uint8)
        
        # img, left_hand_coordinates = self.hand_landmarks_data(img, left_hand_landmarks, params)
        # img, right_hand_coordinates = self.hand_landmarks_data(img, right_hand_landmarks, params)
        # img, pose_coordinates = self.pose_landmarks_data(img, pose_landmarks, params)
        # coordinates_collection = left_hand_coordinates + right_hand_coordinates + pose_coordinates
        img, mouth_coordinates = self.landmarks_data(img, face_landmarks, params, "mouth")
        img, left_hand_coordinates = self.landmarks_data(img, left_hand_landmarks, params, "hand")
        img, right_hand_coordinates = self.landmarks_data(img, right_hand_landmarks, params, "hand")
        img, pose_coordinates = self.landmarks_data(img, pose_landmarks, params, "pose")
        coordinates_collection = mouth_coordinates + left_hand_coordinates + right_hand_coordinates + pose_coordinates
        return coordinates_collection
    
    def landmarks_data(self, img, landmarks, params, key):
        coordinates = []
        if landmarks:
            if key == "mouth":
                for i in self.mouth_indices():
                    landmark = landmarks.landmark[i]
                    img, x, y, z = self.coordinate_recalculation(img, landmark, params)
                    coordinates.append(x)
                    coordinates.append(y)
                    coordinates.append(z)
            elif key == "pose":
                for i in range(0, 33):
                    landmark = landmarks.landmark[i]
                    img, x, y, z = self.coordinate_recalculation(img, landmark, params)
                    coordinates.append(x)
                    coordinates.append(y)
                    coordinates.append(z)
            else:
                for i in landmarks.landmark:
                    img, x, y, z = self.coordinate_recalculation(img, i, params)
                    coordinates.append(x)
                    coordinates.append(y)
                    coordinates.append(z)
        else:
            if key == "hand":
                vertex_num = 21
            if key == "pose":
                vertex_num = 33
            if key == "mouth":
                vertex_num = len(self.mouth_indices())
            for i in range(0, vertex_num):
                for i in range(0, 3):
                    coordinates.append(0)

        return img, coordinates
    
    def coordinate_recalculation(self, img, landmark, params):
        radius = params["radius"] # radius ternormalisasi
        px_radius = params["px_radius"] # radius dalam pixel
        centroid = params["centroid"] # centroid ternormalisasi
        coordinates = [] # x y z
        centroid_x = centroid[0]
        centroid_y = centroid[1]
        original_h = params["image"].shape[0]
        original_w = params["image"].shape[1]
        px_centroid_x, px_centroid_y = self.pixel_coordinate_convertion([centroid_x, centroid_y], original_w, original_h)
        left = px_centroid_x - px_radius
        top = px_centroid_y - px_radius
        px_x, px_y = self.pixel_coordinate_convertion([landmark.x, landmark.y], original_w, original_h)
        w_h = np.min([original_h, original_w])
        radius_width_ratio = w_h / (px_radius * 2)

        # normalized_px_x = px_x - left
        # normalized_px_y = px_y - top
        if px_x >= 0:
            normalized_px_x = abs(left - px_x)
        else:
            normalized_px_x = -abs(left - px_x)
        if px_y >= 0:
            normalized_px_y = abs(top - px_y)
        else:
            normalized_px_y = -abs(top - px_y)
        normalized_x = normalized_px_x / (2 * px_radius)
        normalized_y = normalized_px_y / (2 * px_radius)
        img = cv2.circle(img, (normalized_px_x, normalized_px_y), 4, (0, 255, 0), -1)
        coordinates.append(img)
        coordinates.append(normalized_x)
        coordinates.append(normalized_y)
        coordinates.append(landmark.z * radius_width_ratio)

        return coordinates

    def mouth_indices(self):
        return [0,13,14,17,37,39,40,61,78,80,81,82,84,87,88,91,95,146,178,181,185,191,267,269,270,291,308,310,311,312,314,317,318,321,324,375,402,405,409,415]


    def hand_landmarks_data(self, img, landmarks, params):
        radius = params["radius"]
        px_radius = params["px_radius"]
        centroid = params["centroid"]
        coordinates = []
        centroid_x = centroid[0]
        centroid_y = centroid[1]
        original_h = params["image"].shape[0]
        original_w = params["image"].shape[1]
        px_centroid_x, px_centroid_y = self.pixel_coordinate_convertion([centroid_x, centroid_y], original_w, original_h)
        x = 0
        if landmarks:
            for i in landmarks.landmark:
                left = px_centroid_x - px_radius
                top = px_centroid_y - px_radius
                px_x, px_y = self.pixel_coordinate_convertion([i.x, i.y], original_w, original_h)
                if px_x >= 0:
                    normalized_px_x = abs(left - px_x)
                else:
                    normalized_px_x = -abs(left - px_x)
                if px_y >= 0:
                    normalized_px_y = abs(top - px_y)
                else:
                    normalized_px_y = -abs(top - px_y)
                normalized_x = normalized_px_x / (2 * px_radius)
                normalized_y = normalized_px_y / (2 * px_radius)
                img = cv2.circle(img, (normalized_px_x, normalized_px_y), 2, (0, 0, 255), -1)
                coordinates.append(normalized_x)
                coordinates.append(normalized_y)
                coordinates.append(i.z)
                x += 1
        else:
            for i in range(0, 21):
                for i in range(0, 3):
                    coordinates.append(0)

        return img, coordinates

    def pose_landmarks_data(self, img, landmarks, params):
        coordinates = []
        radius = params["radius"]
        px_radius = params["px_radius"]
        centroid = params["centroid"]
        coordinates = []
        centroid_x = centroid[0]
        centroid_y = centroid[1]
        original_h = params["image"].shape[0]
        original_w = params["image"].shape[1]
        px_centroid_x, px_centroid_y = self.pixel_coordinate_convertion([centroid_x, centroid_y], original_w, original_h)
        x = 0
        if landmarks:
            for i in landmarks.landmark:
                left = px_centroid_x - px_radius
                top = px_centroid_y - px_radius
                px_x, px_y = self.pixel_coordinate_convertion([i.x, i.y], original_w, original_h)
                if px_x >= 0:
                    normalized_px_x = abs(left - px_x)
                else:
                    normalized_px_x = -abs(left - px_x)
                if px_y >= 0:
                    normalized_px_y = abs(top - px_y)
                else:
                    normalized_px_y = -abs(top - px_y)
                normalized_x = normalized_px_x / (2 * px_radius)
                normalized_y = normalized_px_y / (2 * px_radius)
                img = cv2.circle(img, (normalized_px_x, normalized_px_y), 2, (0, 0, 255), -1)
                coordinates.append(normalized_x)
                coordinates.append(normalized_y)
                coordinates.append(i.z)
                x += 1
        else:
            for i in range(0, 33):
                for i in range(0, 3):
                    coordinates.append(0)

        return img, coordinates


    def body_tracking(self, params):
        pose = params["pose_landmarks"]
        centroid_indices = [0, 11, 12, 23, 24]
        if pose is None and self.global_boundingbox_coord is None:
            params["centroid"] = [0, 0, 0]
        elif pose is None and self.global_boundingbox_coord is not None:
            params["centroid"] = self.global_boundingbox_coord
        else:
            params["centroid"] = self.find_body_centroid(pose, centroid_indices)

        if pose is None and self.global_distance is not None:
            params["radius"] = self.global_distance
            params["shoulders_centroid"] = [0, 0, 0]
            params["hips_centroid"] = [0, 0, 0]
        else:
            params = self.find_distance(pose, params)
        return params

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
            self.global_boundingbox_coord = [x_bodies, y_bodies, z_bodies]
            return np.average(x_bodies), np.average(y_bodies), np.average(z_bodies)
        else:
            return 0, 0, 0

    def find_distance(self, landmarks, params):
        indices_a = [11, 12]
        indices_b = [23, 24]
        centroid_a = np.array(self.find_body_centroid(landmarks, indices_a))
        centroid_b = np.array(self.find_body_centroid(landmarks, indices_b))
        params["radius"] = self.euclidean(centroid_a, centroid_b)
        params["shoulders_centroid"] = centroid_a
        params["hips_centroid"] = centroid_b
        return params

    def euclidean(self, a, b):
        sum_sq = np.sum(np.square(a - b))
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