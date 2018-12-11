# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 12:10:02 2018

@author: AyubQuadri
"""

import os
import cv2
import numpy as np


os.getcwd()

def filter_images():
    images= []
    for img in tqdm(os.listdir(TRAIN_DIR))