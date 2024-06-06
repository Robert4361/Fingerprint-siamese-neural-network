from skimage.io import imread_collection
from preprocess import fingerprint_pipeline
import cv2 as cv
import os

class Image:
    def __init__(self, img_name):
        self.img = cv.imread(f"dataset/{img_name}")
        self.name = img_name
    def __str__(self):
        return self.name

names = os.listdir("dataset")

images = []

for name in names:
    images.append(Image(name))

for image in images:
  cv.imwrite(os.path.join('/home/robert/Documents/biometrija/processed_dataset', image.name), fingerprint_pipeline(image.img))
