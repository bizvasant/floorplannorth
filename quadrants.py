import cv2
#from matplotlib import pyplot as plt
# import time
# import re
# import pandas as pd
import os
import numpy as np
# from PIL import Image
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import easyocr
from sympy import Polygon
from shapely.geometry import LineString, Polygon
import warnings
# import string
# import itertools
warnings.filterwarnings('ignore')
from contour import img_contour
from image_process import img_processing

class quadrants(img_contour,img_processing):
    def __init__(self):
        pass

    def find_rooms_with_lines(image,
               corners_threshold=0.1,
               gap_in_wall_threshold=500):
        """
        :param img: grey scale image of rooms, already eroded and doors removed etc.
        :param noise_removal_threshold: Minimal area of blobs to be kept.
        :param corners_threshold: Threshold to allow corners. Higher removes more of the house.
        :param room_closing_max_length: Maximum line length to add to close off open doors.
        :param gap_in_wall_threshold: Minimum number of pixels to identify component as room instead of hole in the wall.
        :return: rooms: list of numpy arrays containing boolean masks for each detected room
                 colored_house: A colored version of the input image, where each room has a random color.
        """
        contour, thresh = img_contour.find_contour(image)
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 8)
        assert 0 <= corners_threshold <= 1
        # preprocessing the image
        gray = img_processing.get_grayscale(image)
        th = img_processing.thresholding(gray)
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
        processed_img = closing.copy()
        edged = cv2.Canny(processed_img, 1, 1)
        lines = cv2.HoughLinesP(edged, # Input edge image
                                1, # Distance resolution in pixels
                                np.pi / 180, # Angle resolution in radians
                                100, # Min number of votes for valid line
                                minLineLength=200, # Min allowed length of line
                                maxLineGap=250) # Max allowed gap between line for joining them
        #getting boundry box co-ordinates from ocr
        reader = easyocr.Reader(['en'])
        result = reader.readtext(image, paragraph="False")
        l = []
        for i in result:
            polycoords = i[0]
            polygon = Polygon(polycoords)
            for j, k in enumerate(lines):
                if j < len(lines):
                    line = LineString([(k[0][0], k[0][1]), (k[0][2], k[0][3])])
                    if polygon.intersects(line):
                        l.append(j)
        l = list(set(l))
        f = np.delete(lines, l, 0)
        lines = f.copy()
        #drawing lines on images
        for i in range(len(lines)):
            if i < len(lines):
                cv2.line(processed_img,
                         (lines[i][0][0], lines[i][0][1]),
                         (lines[i][0][2], lines[i][0][3]), (0, 255, 0),
                         10)
        # Mark the outside of the house as black
        mask = np.zeros(thresh.shape[:2], np.uint8)
        cv2.fillPoly(mask, pts=[np.asarray(contour)], color=(1))
        processed_img[mask == 0] = 0
        # Find the connected components in the house
        # ret, labels = cv2.connectedComponents(processed_img)
        totalLabels, labels, values, centroid = cv2.connectedComponentsWithStats(processed_img,4,cv2.CV_32S)
        
        img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
        unique = np.unique(labels)
        room_mapping = {"room": [], "color": [] ,"area": []}
        for label in unique:
            component = labels == label
            if img[component].sum(
            ) == 0 or np.count_nonzero(component) < gap_in_wall_threshold:
                color = 0
            else:
                room_mapping["room"].append(component)
                color = np.random.randint(0, 255, size=3)
                room_mapping["color"].append(color)
                area = values[label, cv2.CC_STAT_AREA]
                room_mapping['area'].append(area)
            img[component] = color
        return img, room_mapping
        
    def find_rooms_wo_lines(image, noise_removal_threshold=40, corners_threshold=0.1,
                   room_closing_max_length=100, gap_in_wall_threshold=500):
        """
        :param img: grey scale image of rooms, already eroded and doors removed etc.
        :param noise_removal_threshold: Minimal area of blobs to be kept.
        :param corners_threshold: Threshold to allow corners. Higher removes more of the house.
        :param room_closing_max_length: Maximum line length to add to close off open doors.
        :param gap_in_wall_threshold: Minimum number of pixels to identify component as room instead of hole in the wall.
        :return: rooms: list of numpy arrays containing boolean masks for each detected room
                 colored_house: A colored version of the input image, where each room has a random color.
        """
        assert 0 <= corners_threshold <= 1

        #preprocessing the image
        gray = img_processing.get_grayscale(image)
    #         blur = img_processing.remove_noise(gray)
        th = img_processing.thresholding(gray)
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
        img = closing.copy()
        edged = cv2.Canny(img, 1, 1)

        lines = cv2.HoughLinesP(edged, 1, np.pi/180, 50, minLineLength=350, maxLineGap=380)
        for i in range(len(lines)):
            if i < len(lines):
                cv2.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0,255,0), 3)

        # Mark the outside of the house as black
        contour, thresh = img_contour.find_contour(image)
        mask = np.zeros(thresh.shape[:2], np.uint8)
        cv2.fillPoly(mask, pts =[np.asarray(contour)], color=(1))
        img[mask == 0] = 0

        # Find the connected components in the house
        # ret, labels = cv2.connectedComponents(img)
        totalLabels, labels, values, centroid = cv2.connectedComponentsWithStats(img,4,cv2.CV_32S)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        unique = np.unique(labels)
        room_mapping = {"room": [], "color": [] ,"area": []}
        for label in unique:
            component = labels == label
            if img[component].sum() == 0 or np.count_nonzero(component) < gap_in_wall_threshold:
                color = 0
            else:
                room_mapping["room"].append(component)
                color = np.random.randint(0, 255, size=3)
                room_mapping["color"].append(color)
                area = values[label, cv2.CC_STAT_AREA]
                room_mapping['area'].append(area)
            img[component] = color
        # print('quad_',img.shape)
        # print(room_mapping.keys())
        return img, room_mapping
        
        