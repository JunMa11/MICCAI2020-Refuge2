import cv2
import os
from utils import keep_large_area
import numpy as np
from skimage.measure import *
import matplotlib.pyplot as plt
import pandas as pd

img_dir = r'E:\competition\refuge2\GPU\data\challenge\val\img'
ImageName = []
Fovea_X = []
Fovea_Y = []

# for name in os.listdir(img_dir):
for i in range(400):
    name = 'V{}.jpg'.format(str(i + 1).zfill(4))
    print(name)
    img = cv2.imread(os.path.join(img_dir, name), 0)
    mask = cv2.imread(os.path.join(r'C:\Users\DELL\Desktop\Fovea-png', name.rstrip('.jpg') + '.png'), 0)
    mask[mask > 0] = 1
    mask = keep_large_area(mask, 1)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(np.uint8)
    connect_regions = label(mask, connectivity=1, background=0)
    props = regionprops(connect_regions)
    Fovea_y, Fovea_x = props[0].centroid
    Fovea_x = round(Fovea_x, 2)
    Fovea_y = round(Fovea_y, 2)
    ImageName.append(name)
    Fovea_X.append(Fovea_x)
    Fovea_Y.append(Fovea_y)
    print(Fovea_x, Fovea_y)
save = pd.DataFrame({'ImageName': ImageName, 'Fovea_X': Fovea_X, 'Fovea_Y': Fovea_Y})
save.to_csv(r'E:\competition\refuge2\GPU\data\challenge\Example-Refuge2/fovea_location_results.csv', index=False)

