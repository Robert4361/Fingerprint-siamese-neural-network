from itertools import product
import cv2 as cv
import os
import random
import numpy as np
from skimage import transform

 
def cartesian_product(arr1, arr2):
    return list(product(arr1, arr2)) 


class Image:
    def __init__(self, img_name):
        self.img = transform.rescale(cv.imread(f"processed_dataset/{img_name}").astype("float32"), 0.4)
        self.name = img_name
    def __str__(self):
        return self.name

def group_fingers(images):
    image_groups = {}
    for image_file in images:
        parts = image_file.name.split("_")
        person_id = parts[0]
        finger_id = parts[1]
        key = (person_id, finger_id)
        if key not in image_groups:
            image_groups[key] = [image_file]
        else:
            image_groups[key].append(image_file)
    return image_groups

def get_pairs():
    names = os.listdir("processed_dataset")

    images = []

    for name in names:
        images.append(Image(name))

    grouped = group_fingers(images)

    matching_pairs = []

    for key, images in grouped.items():
        matching_pairs = matching_pairs + cartesian_product(images, images)

    non_matching_pairs = []

    for _ in range(65):
        a,b = random.sample(range(0, 64), 2)
        first_group = list(grouped.items())[a][1]
        second_group = list(grouped.items())[b][1]
        non_matching_pairs = non_matching_pairs + cartesian_product(first_group, second_group)
    
    #for count in range(4160):
    #    matching_pairs[count] = (matching_pairs[count][0].img, matching_pairs[count][1].img)
    #    non_matching_pairs[count] = (non_matching_pairs[count][0].img, non_matching_pairs[count][1].img)

    return [matching_pairs, non_matching_pairs]


def remove_names(images):
    nameless_images = []
    for image in images:
        nameless_images.append([image[0].img, image[1].img])
    return np.array(nameless_images)


