from cv2 import cv2
import mediapipe as mp
import json
import glob
from pathlib import Path
import pprint as pp
from turtle import right
import numpy as np

class RecalculateNormalization2():
    visualization = False
    def __init__(self, mp_holistic):
        self.mp_holistic = mp_holistic
        self.holistic = mp_holistic.Holistic()
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
        
        mouth_coordinates, mouth_coordinates_indexed = self.landmarks_data(face_landmarks, params, "mouth")
        left_hand_coordinates, left_hand_coordinates_indexed = self.landmarks_data(left_hand_landmarks, params, "hand")
        right_hand_coordinates, right_hand_coordinates_indexed = self.landmarks_data(right_hand_landmarks, params, "hand")
        pose_coordinates, pose_coordinates_indexed = self.landmarks_data(pose_landmarks, params, "pose")
        coordinates_collection = mouth_coordinates + left_hand_coordinates + right_hand_coordinates + pose_coordinates
        params["mouth_coordinates_indexed"] = mouth_coordinates_indexed
        params["left_hand_coordinates_indexed"] = left_hand_coordinates_indexed
        params["right_hand_coordinates_indexed"] = right_hand_coordinates_indexed
        params["pose_coordinates_indexed"] = pose_coordinates_indexed
        self.visualize(img, params)

        return coordinates_collection
    
    def landmarks_data(self, landmarks, params, key):
        coordinates = []
        indexed_coordinates = []
        if landmarks:
            if key == "mouth":
                for i in self.mouth_indices():
                    landmark = landmarks.landmark[i]
                    x, y, z = self.coordinate_recalculation(landmark, params)
                    coordinates.append(x)
                    coordinates.append(y)
                    coordinates.append(z)
                    indexed_coordinates.append([x, y, z])
            elif key == "pose":
                for i in range(0, 33):
                    landmark = landmarks.landmark[i]
                    x, y, z = self.coordinate_recalculation(landmark, params)
                    coordinates.append(x)
                    coordinates.append(y)
                    coordinates.append(z)
                    indexed_coordinates.append([x, y, z])
            else:
                for i in landmarks.landmark:
                    x, y, z = self.coordinate_recalculation(i, params)
                    coordinates.append(x)
                    coordinates.append(y)
                    coordinates.append(z)
                    indexed_coordinates.append([x, y, z])
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
                indexed_coordinates.append([0, 0, 0])

        return coordinates, indexed_coordinates

    def visualize(self, blank_img, params):
        image = params["image"]
        im_h = image.shape[0]
        im_w = image.shape[1]
        px_radius = params["px_radius"]
        mouth_coordinates_indexed = params["mouth_coordinates_indexed"]
        left_hand_coordinates_indexed = params["left_hand_coordinates_indexed"]
        right_hand_coordinates_indexed = params["right_hand_coordinates_indexed"]
        pose_coordinates_indexed = params["pose_coordinates_indexed"]
        hand_connections = self.mp_holistic.HAND_CONNECTIONS
        pose_connections = self.mp_holistic.POSE_CONNECTIONS
        face_mesh = self.mp_holistic.FACE_CONNECTIONS
        lips_connections = [i for i in face_mesh if i[0] in self.mouth_indices() and i[1] in self.mouth_indices()]
        self.draw_landmarks(blank_img, left_hand_coordinates_indexed, hand_connections)
        self.draw_landmarks(blank_img, right_hand_coordinates_indexed, hand_connections)
        self.draw_landmarks(blank_img, pose_coordinates_indexed, pose_connections)
        self.draw_lips_landmarks(blank_img, mouth_coordinates_indexed, lips_connections)

        dim1 = (500, 500)
        dim2 = (int(500 / im_h * im_w), 500)
        # resize image
        centroid = params["centroid"]
        centroid_x = centroid[0]
        centroid_y = centroid[1]
        center_x, center_y = self.pixel_coordinate_convertion([centroid_x, centroid_y], im_w, im_h)
        image = cv2.rectangle(image, (center_x-px_radius, center_y-px_radius),
                            (center_x+px_radius, center_y+px_radius), (0, 0, 255), 2)
        resized1 = cv2.resize(blank_img, dim1, interpolation = cv2.INTER_AREA)
        resized2 = cv2.resize(image.copy(), dim2, interpolation = cv2.INTER_AREA)
        resized2 = cv2.cvtColor(resized2, cv2.COLOR_RGB2BGR)
        merged = np.concatenate((resized1, resized2), axis=1)
        cv2.imshow("video", merged)

    def coordinate_recalculation(self, landmark, params):
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
        coordinates.append(normalized_x)
        coordinates.append(normalized_y)
        coordinates.append(landmark.z * radius_width_ratio)

        return coordinates

    def mouth_indices(self):
        return [0,13,14,17,37,39,40,61,78,80,81,82,84,87,88,91,95,146,178,181,185,191,267,269,270,291,308,310,311,312,314,317,318,321,324,375,402,405,409,415]

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

    def draw_landmarks(self, image, landmarks, connections):
        for connection in connections:
            x0, y0 = int(landmarks[connection[0]][0] * image.shape[1]), int(landmarks[connection[0]][1] * image.shape[0])
            x1, y1 = int(landmarks[connection[1]][0] * image.shape[1]), int(landmarks[connection[1]][1] * image.shape[0])
            if (x0 >= 0 or y0 >= 0) or (x0 >= 0 or y0 >= 0):
                cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 4)
                # print(x0, y0, x1, y1)
        for landmark in landmarks:
            x, y = int(landmark[0] * image.shape[1]), int(landmark[1] * image.shape[0])
            cv2.circle(image, (x, y), 1, (255, 0, 0), 3)

    def draw_lips_landmarks(self, image, lips_landmarks, connections):
        indices = np.array(self.mouth_indices())
        for connection in connections:
            get_index_x = np.where(indices == connection[0])[0][0]
            get_index_y = np.where(indices == connection[1])[0][0]
            x0, y0 = int(lips_landmarks[get_index_x][0] * image.shape[1]), int(lips_landmarks[get_index_x][1] * image.shape[0])
            x1, y1 = int(lips_landmarks[get_index_y][0] * image.shape[1]), int(lips_landmarks[get_index_y][1] * image.shape[0])
            cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 4)
        for landmark in lips_landmarks:
            x, y = int(landmark[0] * image.shape[1]), int(landmark[1] * image.shape[0])
            cv2.circle(image, (x, y), 1, (255, 0, 0), 3)


        return image