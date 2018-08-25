# https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial

##############################
# data_process.py is used to extract the data and label from the dicom files
##############################

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom as dicom
import os
import cv2
from glob import glob
import scipy.ndimage
from skimage.draw import polygon
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Some constants
INPUT_FOLDER = '/Users/apple/Desktop/VIPCUP/01-Dataset/01-finalDataSet/VIP_CUP18_ValidationData/'
patient_paths = glob(INPUT_FOLDER + '*/')
patient_paths.sort()


# Load the scans in given folder path
def load_scan(root):
    paths = glob(root + '*/')
    for path in paths:
        if len(glob(path + '*/*')) > 1:
            slice_paths = glob(path + '*/*')
        else:
            structure = dicom.read_file(glob(path + '*/*')[0])
            if len(structure.ROIContourSequence) < 1:
                print('error! wrong number of contours')
            contours = read_structure(structure)
    slices = [dicom.read_file(s) for s in slice_paths]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices, contours


def read_structure(structure):
    contours = []
    for i in range(len(structure.ROIContourSequence)):
        contour = {}
        contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
        contour['number'] = structure.StructureSetROISequence[i].ROINumber
        contour['name'] = structure.StructureSetROISequence[i].ROIName
        contour['contours'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
        contours.append(contour)
    return contours


def get_mask(contours, slices, image):
    z = [s.ImagePositionPatient[2] for s in slices]
    pos_r = slices[0].ImagePositionPatient[1]
    spacing_r = slices[0].PixelSpacing[1]
    pos_c = slices[0].ImagePositionPatient[0]
    spacing_c = slices[0].PixelSpacing[0]
    label = np.zeros_like(image, dtype=np.uint8)
    for i, contour in enumerate(contours):
        if contour['name'] == 'GTV-1':
            break
    for c in contours[i]['contours']:
        nodes = np.array(c).reshape((-1, 3))
        assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
        try:
            z_index = z.index(np.around(nodes[0, 2], 1))
            r = (nodes[:, 1] - pos_r) / spacing_r
            c = (nodes[:, 0] - pos_c) / spacing_c
            # r = (nodes[:, 1] - pos_r)
            # c = (nodes[:, 0] - pos_c)
            rr, cc = polygon(r, c)
            label[z_index, rr, cc] = 255
        except ValueError as e:
            print('value error, point z index expire')
    return label


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
        # print(slope, intercept)
    return np.array(image, dtype=np.int16)


def check_resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + [scan[0].PixelSpacing[0]] + [scan[0].PixelSpacing[1]], dtype=np.float32)
    if spacing[0] != 3:
        print('error! wrong thinkness')
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    if (new_shape[1:] != np.asarray([500., 500.])).any():
        print('error! wrong size', new_shape)
    # real_resize_factor = new_shape / image.shape
    # new_spacing = spacing / real_resize_factor
    # image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    # return image, new_spacing


def normalize_clip(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image[image > MAX_BOUND] = MAX_BOUND
    image[image < MIN_BOUND] = MIN_BOUND
    return image


def draw_overlay(inputs, masks, output_path):
    output_path_image = output_path.replace('overlay', 'image')
    output_path_mask = output_path.replace('overlay', 'mask')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(output_path_image):
        os.makedirs(output_path_image)
    if not os.path.exists(output_path_mask):
        os.makedirs(output_path_mask)
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    alpha = 0.5
    for i, (input, mask) in enumerate(zip(inputs, masks)):
        output = (input - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        output = np.uint8(output*255)
        #draw slice
        cv2.imwrite('{}{:04d}.png'.format(output_path_image, i), output)
        #draw mask
        cv2.imwrite('{}{:04d}.png'.format(output_path_mask, i), mask)
        #draw overlay
        cv2.addWeighted(mask, alpha, output, 1 - alpha, 0, output)
        cv2.imwrite('{}{:04d}.jpg'.format(output_path, i), output)


def draw_image(inputs, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    for i, input in enumerate(inputs):
        output = (input - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        output = np.uint8(output*255)
        #draw slice
        cv2.imwrite('{}{:04d}.png'.format(output_path, i), output)

def draw_mask(masks, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i, mask in enumerate(masks):
        # draw mask
        cv2.imwrite('{}{:04d}.png'.format(output_path, i), mask)

def draw_image_and_mask(inputs, masks, output_path_imgae, output_path_mask):
    if not os.path.exists(output_path_imgae):
        os.makedirs(output_path_imgae)
    if not os.path.exists(output_path_mask):
        os.makedirs(output_path_mask)
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    for i, (input, mask) in enumerate(zip(inputs, masks)):
        output = (input - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        output = np.uint8(output*255)
        #draw slice
        cv2.imwrite('{}{:04d}.png'.format(output_path_imgae, i), output)
        # draw mask
        cv2.imwrite('{}{:04d}.png'.format(output_path_mask, i), mask)


for patient_path in patient_paths:
    print(patient_path)
    patient, contours = load_scan(patient_path)
    patient_pixels = get_pixels_hu(patient)
    masks = get_mask(contours, patient, patient_pixels)
    check_resample(patient_pixels, patient, [1,1,1])
    patient_pixels_norm = normalize_clip(patient_pixels)
    patient_num = patient_path.split('/')[-2]
    print(patient_pixels_norm.shape, masks.shape)

    output_path_image = '/Users/apple/Desktop/VIPCUP/01-Dataset/01-finalDataSet/validation_data' + '/{}/{}/'.format(INPUT_FOLDER.split('/')[-2], patient_num)
    output_path_mask = '/Users/apple/Desktop/VIPCUP/01-Dataset/01-finalDataSet/validation_label' + '/{}/{}/'.format(INPUT_FOLDER.split('/')[-2], patient_num)

    draw_image_and_mask(patient_pixels_norm, masks, output_path_image, output_path_mask)

    # draw_image(patient_pixels_norm, '/Users/apple/Desktop/VIPCUP/01-Dataset/01-finalDataSet/data' + '/{}/{}/'.format(INPUT_FOLDER.split('/')[-2], patient_num))
    # draw_mask(masks, '/Users/apple/Desktop/VIPCUP/01-Dataset/01-finalDataSet/label' + '/{}/{}/'.format(INPUT_FOLDER.split('/')[-2], patient_num))
    #
    # draw_overlay(patient_pixels_norm, masks, '../../../Datasets_hdd/VIPCUP/' + 'preprocessed/{}/{}/overlay/'.format(INPUT_FOLDER.split('/')[-2], patient_num))
    # np.save('../../../Datasets_hdd/VIPCUP/' + 'preprocessed/{}/{}_slices.npy'.format(INPUT_FOLDER.split('/')[-2], patient_num), patient_pixels_norm)
    # np.save('../../../Datasets_hdd/VIPCUP/' + 'preprocessed/{}/{}_masks.npy'.format(INPUT_FOLDER.split('/')[-2], patient_num), masks)

