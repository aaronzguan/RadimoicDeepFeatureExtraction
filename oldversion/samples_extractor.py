import cv2
import os
import numpy as np
import scipy.misc
import random

##############################
# samples_extractor.py is used to extract the positive and
# negative training samples from the data and label images
##############################

def delete_ds_store(fpath):
    for root, dirs, files in os.walk(fpath):
        for file in files:
            if file.endswith('.DS_Store'):
                path = os.path.join(root, file)
                os.remove(path)

class samples_extractor():
    def save_samples(self, image, coord, save_path):
        sample_region = np.zeros_like(image)
        sample_region[coord[0]:coord[1], coord[2]:coord[3]] = image[coord[0]:coord[1], coord[2]:coord[3]]
        coord_tuple = np.where(sample_region != 0)
        if len(coord_tuple[0]) != 0:
            scipy.misc.imsave(save_path, sample_region)


    def get_samples(self, label_path, data_path, positive_save_path, negative_save_path, window_ratio, find_positive, find_negative):
        delete_ds_store(label_path)
        patient_list = os.listdir(label_path)
        patient_list.sort()
        tumor_coord = []
        for patient_name, i in zip(patient_list, range(len(patient_list))):
            patient_path = os.path.join(label_path, patient_name)
            dicom_path = os.path.join(data_path, patient_name)

            if find_positive:
                if not os.path.exists(positive_save_path):
                    os.mkdir(positive_save_path)
                positive_out_path = os.path.join(positive_save_path, patient_name)
                if not os.path.exists(positive_out_path):
                    os.mkdir(positive_out_path)
            if find_negative:
                if not os.path.exists(negative_save_path):
                    os.mkdir(negative_save_path)
                negative_out_path = os.path.join(negative_save_path, patient_name)
                if not os.path.exists(negative_out_path):
                    os.mkdir(negative_out_path)

            print(patient_name + ' - ' + str(i) + ' out of ' + str(len(patient_list)))
            delete_ds_store(patient_path)
            img_list = os.listdir(patient_path)

            for img_name in img_list:
                index = img_name.split('.')[0]
                img_label_path = os.path.join(patient_path, img_name)
                img_dicom_path = os.path.join(dicom_path, img_name)
                label = cv2.imread(img_label_path)
                image = cv2.imread(img_dicom_path)
                coord_tuple = np.where(label != 0)

                if len(coord_tuple[0]) == 0:

                    # Extract Negative Samples
                    if find_negative:
                        image_coord_tuple = np.where(image != 0)
                        x = random.randint(min(image_coord_tuple[0])+200, max(image_coord_tuple[0])-200)
                        y = random.randint(min(image_coord_tuple[1])+200, max(image_coord_tuple[1])-200)
                        square_length = random.randint(20, 70)

                        X_min = int(x - (square_length/2))
                        X_max = int(x + (square_length/2))
                        Y_min = int(y - (square_length/2))
                        Y_max = int(y + (square_length/2))

                        coord = [X_min, X_max, Y_min, Y_max]
                        save_path = os.path.join(negative_out_path, (str(index)+'_1.png'))
                        self.save_samples(image, coord, save_path)
                        if len(tumor_coord) != 0:
                            save_path = os.path.join(negative_out_path, (str(index)+'_2.png'))
                            self.save_samples(image, tumor_coord, save_path)
                    continue

                # Extract Positive Samples
                x_min = min(coord_tuple[0])
                x_max = max(coord_tuple[0])
                y_min = min(coord_tuple[1])
                y_max = max(coord_tuple[1])

                tumor_coord = [x_min, x_max, y_min, y_max]

                x_length = abs(x_max - x_min)
                y_length = abs(y_max - y_min)
                x_mid = x_min + (x_length / 2)
                y_mid = y_min + (y_length / 2)

                if x_length < y_length:
                    square_length = np.multiply(y_length, window_ratio)
                else:
                    square_length = np.multiply(x_length, window_ratio)

                X_min = int(x_mid - (square_length / 2))
                X_max = int(x_mid + (square_length / 2))
                Y_min = int(y_mid - (square_length / 2))
                Y_max = int(y_mid + (square_length / 2))

                if find_positive:

                    coord = [X_min, X_max, Y_min, Y_max]
                    save_path = os.path.join(positive_out_path, str(img_name))
                    self.save_samples(image, coord, save_path)

                    # coord = [X_min + 2, X_max + 2, Y_min, Y_max]
                    # save_path = os.path.join(positive_out_path, (str(index)+'_1.png'))
                    # self.save_samples(image, coord, save_path)
                    #
                    # coord = [X_min - 2, X_max - 2, Y_min, Y_max]
                    # save_path = os.path.join(positive_out_path, (str(index) + '_2.png'))
                    # self.save_samples(image, coord, save_path)
                    #
                    # coord = [X_min, X_max, Y_min + 2, Y_max + 2]
                    # save_path = os.path.join(positive_out_path, (str(index) + '_3.png'))
                    # self.save_samples(image, coord, save_path)
                    #
                    # coord = [X_min, X_max, Y_min - 2, Y_max - 2]
                    # save_path = os.path.join(positive_out_path, (str(index) + '_4.png'))
                    # self.save_samples(image, coord, save_path)

                    # coord = [X_min + 2, X_max + 2, Y_min + 2, Y_max + 2]
                    # save_path = os.path.join(positive_out_path, (str(index) + '_5.png'))
                    # self.save_samples(image, coord, save_path)
                    #
                    # coord = [X_min + 2, X_max + 2, Y_min - 2, Y_max - 2]
                    # save_path = os.path.join(positive_out_path, (str(index) + '_6.png'))
                    # self.save_samples(image, coord, save_path)
                    #
                    # coord = [X_min - 2, X_max - 2, Y_min - 2, Y_max - 2]
                    # save_path = os.path.join(positive_out_path, (str(index) + '_7.png'))
                    # self.save_samples(image, coord, save_path)
                    #
                    # coord = [X_min - 2, X_max - 2, Y_min + 2, Y_max + 2]
                    # save_path = os.path.join(positive_out_path, (str(index) + '_8.png'))
                    # self.save_samples(image, coord, save_path)

                if find_negative:
                    # Extract Negative Samples 1
                    X_min_1 = X_min
                    X_max_1 = X_max
                    Y_max_1 = int(y_min + (y_length/6))
                    Y_min_1 = int(Y_max_1 - square_length)
                    coord = [X_min_1, X_max_1, Y_min_1, Y_max_1]
                    save_path = os.path.join(negative_out_path, (str(index)+'_1.png'))
                    self.save_samples(image, coord, save_path)

                    # Extract Negative Samples 2
                    X_min_2 = X_min
                    X_max_2 = X_max
                    Y_min_2 = int(y_max - (y_length/6))
                    Y_max_2 = int(Y_min_2 + square_length)
                    coord = [X_min_2, X_max_2, Y_min_2, Y_max_2]
                    save_path = os.path.join(negative_out_path, (str(index)+'_2.png'))
                    self.save_samples(image, coord, save_path)

                    # Extract Negative Samples 3
                    X_min_3 = int(x_max - (x_length/6))
                    X_max_3 = int(X_min_3 + square_length)
                    Y_min_3 = Y_min
                    Y_max_3 = Y_max
                    coord = [X_min_3, X_max_3, Y_min_3, Y_max_3]
                    save_path = os.path.join(negative_out_path, (str(index)+'_3.png'))
                    self.save_samples(image, coord, save_path)

                    # Extract Negative Samples 4
                    X_max_4 = int(x_min + (x_length/6))
                    X_min_4 = int(X_max_4 - square_length)
                    Y_min_4 = Y_min
                    Y_max_4 = Y_max
                    coord = [X_min_4, X_max_4, Y_min_4, Y_max_4]
                    save_path = os.path.join(negative_out_path, (str(index)+'_4.png'))
                    self.save_samples(image, coord, save_path)


label_path = '/Users/apple/Desktop/VIPCUP/01-Dataset/01-finalDataSet/label/VIP_CUP18_TrainingData'
data_path = '/Users/apple/Desktop/VIPCUP/01-Dataset/01-finalDataSet/data/VIP_CUP18_TrainingData'
positive_save_path = '/Users/apple/Desktop/VIPCUP/01-Dataset/01-finalDataSet/Positive_Data_Set_new_1'
negative_save_path = '/Users/apple/Desktop/VIPCUP/01-Dataset/01-finalDataSet/Negative_Data_Set_tmp'
# label_path = '/Users/apple/Desktop/VIPCUP/01-Dataset/label/VIP-CUP18-Data'
# data_path = '/Users/apple/Desktop/VIPCUP/01-Dataset/data/VIP-CUP18-Data'
# positive_save_path = '/Users/apple/Desktop/VIPCUP/01-Dataset/Positive_Data_Set'
# negative_save_path = '/Users/apple/Desktop/VIPCUP/01-Dataset/Negative_Data_Set'

samples_extractor = samples_extractor()

find_positive = True
find_negative = False

window_ratio = 1
samples_extractor.get_samples(label_path, data_path, positive_save_path, negative_save_path, window_ratio, find_positive, find_negative)

window_ratio = 1.5
positive_save_path = '/Users/apple/Desktop/VIPCUP/01-Dataset/01-finalDataSet/Positive_Data_Set_new_1.5'
samples_extractor.get_samples(label_path, data_path, positive_save_path, negative_save_path, window_ratio, find_positive, find_negative)

window_ratio = 2
positive_save_path = '/Users/apple/Desktop/VIPCUP/01-Dataset/01-finalDataSet/Positive_Data_Set_new_2'
samples_extractor.get_samples(label_path, data_path, positive_save_path, negative_save_path, window_ratio, find_positive, find_negative)

window_ratio = 1
label_path = '/Users/apple/Desktop/VIPCUP/01-Dataset/02-initialDataSet/label/VIP-CUP18-Data'
data_path = '/Users/apple/Desktop/VIPCUP/01-Dataset/02-initialDataSet/data/VIP-CUP18-Data'
positive_save_path = '/Users/apple/Desktop/VIPCUP/01-Dataset/01-finalDataSet/Positive_Data_Set_old_1'
samples_extractor.get_samples(label_path, data_path, positive_save_path, negative_save_path, window_ratio, find_positive, find_negative)

window_ratio = 1.5
positive_save_path = '/Users/apple/Desktop/VIPCUP/01-Dataset/01-finalDataSet/Positive_Data_Set_old_1.5'
samples_extractor.get_samples(label_path, data_path, positive_save_path, negative_save_path, window_ratio, find_positive, find_negative)

window_ratio = 2
positive_save_path = '/Users/apple/Desktop/VIPCUP/01-Dataset/01-finalDataSet/Positive_Data_Set_old_2'
samples_extractor.get_samples(label_path, data_path, positive_save_path, negative_save_path, window_ratio, find_positive, find_negative)
