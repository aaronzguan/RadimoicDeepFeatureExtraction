import torch
import torchvision.models as models
from torchsummary import summary
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import os
from torch.optim.lr_scheduler import StepLR
import argparse
import random
import math


from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

import logging

import matplotlib.pyplot as plt


logging.basicConfig(filename='alexnet.log', level=logging.INFO)

parser = argparse.ArgumentParser(description='AlexNet')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--batch_size', default=10, type=int, help='')
parser.add_argument('--num_epochs', default=200, type=int, help='')
parser.add_argument('--model_root', default='/home/aaron/VIPCUP/Radiomic/model/alexnet_18_09_09.pkl')
args = parser.parse_args()

def delete_ds_store(fpath):
    for root, dirs, files in os.walk(fpath):
        for file in files:
            if file.endswith('.DS_Store'):
                path = os.path.join(root, file)
                os.remove(path)


# torch.cuda.set_device(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def read_samples(positive_data_path, positive_label, negative_data_path, negative_label):
    positive_data = []
    negative_data = []

    delete_ds_store(positive_data_path)
    patient_list = os.listdir(positive_data_path)
    for patient_folder in patient_list:
        patient_path = os.path.join(positive_data_path, patient_folder)
        # print('reading samples from: ' + str(patient_path))
        delete_ds_store(patient_path)
        img_list = os.listdir(patient_path)
        for img in img_list:
            positive_data.append(os.path.join(patient_path, img))

    delete_ds_store(negative_data_path)
    patient_list = os.listdir(negative_data_path)
    for patient_folder in patient_list:
        patient_path = os.path.join(negative_data_path, patient_folder)
        # print('reading samples from: ' + str(patient_path))
        delete_ds_store(patient_path)
        img_list = os.listdir(patient_path)
        for img in img_list:
            negative_data.append(os.path.join(patient_path, img))

    # ------------------------
    # Create Training Dataset
    # Positive_training: 3331
    # Negative_training: 3331 * 3 = 9993
    # ------------------------
    positive_training_num = len(positive_data) - 1000
    positive_data_train = random.sample(positive_data, positive_training_num)

    negative_training_num = len(positive_data)*3
    negative_data_train = random.sample(negative_data, negative_training_num)

    data_train = positive_data_train + negative_data_train
    label_train = [positive_label]*len(positive_data_train) + [negative_label]*len(negative_data_train)
    # ------------------------
    # Create Validation Dataset
    # Positive_val: 1000
    # Negative_val: 3000
    # ------------------------
    positive_data_val = [i for i in positive_data if i not in positive_data_train]
    negative_data_val = random.sample([i for i in negative_data if i not in negative_data_train], len(positive_data_val)*3)

    data_val = positive_data_val + negative_data_val
    label_val = [positive_label]*len(positive_data_val) + [negative_label]*len(negative_data_val)

    return data_train, label_train, data_val, label_val


class load_data(Dataset):
    def __init__(self, positive_data_path, positive_label, negative_data_path, negative_label, is_train=True):
        # Transforms
        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                             ])
        # Read the image files & label for train or validation
        if is_train:
            self.data, self.label, _, _ = read_samples(positive_data_path, positive_label, negative_data_path, negative_label)
        else:
            _, _, self.data, self.label = read_samples(positive_data_path, positive_label, negative_data_path, negative_label)

        # Calculate len
        self.data_len = len(self.data)

    def __getitem__(self, index):
        # Get label(class) of the image
        label = [self.label[index]]
        label = np.asarray(label).reshape(1, 1)
        # Get image path
        img_path = self.data[index]
        # Open image
        img = Image.open(img_path)
        # Transform image
        img = self.transform(img)

        return img, torch.LongTensor(label)

    def __len__(self):
        return self.data_len

# def performace_evalution(outputs, labels):


def train(net, loader, optimizer, criterion, epoch):

    net.train()

    print('Epoch {}/{}'.format(epoch+1, args.num_epochs))
    logging.info('Epoch {}/{}'.format(epoch + 1, args.num_epochs))
    print('-' * 10)
    logging.info('-' * 10)

    running_batch = 0
    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for i, (image, label) in enumerate(loader):

        image = image.to(device)
        label = label.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        output = net(image)
        _, label_pred = torch.max(output, 1)

        loss = criterion(output, label.view(-1))
        loss.backward()
        optimizer.step()

        running_batch += label.size(0)
        running_loss += loss.item() * image.size(0)
        running_corrects += torch.sum(label_pred == label.view(-1)).item()

        if (i+1) % 50 == 0:  # print every 50 mini-batches
            print('[%d, %5d] loss: %.3f correct: %.3f' %
                  (epoch + 1, i + 1, running_loss/running_batch, running_corrects/running_batch))
            logging.info('[%d, %5d] loss: %.3f correct: %.3f' %
                  (epoch + 1, i + 1, running_loss/running_batch, running_corrects/running_batch))


def validate(net, loader, criterion, epoch):

    net.eval()

    running_batch = 0
    running_loss = 0.0
    running_corrects = 0

    label_truth = []
    label_output = []

    with torch.no_grad():
        for i, (image, label) in enumerate(loader):
            image = image.to(device)
            label = label.to(device)
            output = net(image)

            label_pred = torch.nn.functional.softmax(output)[:, 1]
            # _, label_pred = torch.max(output, 1)
            loss = criterion(output, label.view(-1))
            running_batch += label.size(0)
            running_loss += loss.item() * image.size(0)
            # running_corrects += torch.sum(label_pred == label.view(-1)).item()

            # if (i+1) % 50 == 0:  # print every 50 mini-batches
            #     print('[%d, %5d] loss_val: %.3f correct_val: %.3f' %
            #           (epoch + 1, i + 1, running_loss/running_batch, running_corrects/running_batch))
            #     logging.info('[%d, %5d] loss_val: %.3f correct_val: %.3f' %
            #                  (epoch + 1, i + 1, running_loss/running_batch, running_corrects/running_batch))

            label_truth.extend(label.view(-1).cpu().numpy())
            label_output.extend(label_pred.cpu().numpy())

    precision, recall, thresholds = precision_recall_curve(label_truth, label_output)
    average_precision = average_precision_score(label_truth, label_output)

    plt.figure()
    plt.plot(recall, precision)
    # plt.step(recall, precision, color='b', alpha=0.2, where='post')
    # plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    # plt.show()
    plt.savefig('%d.png' % epoch)

    print('[%d, %5d] average_precision_val: %.3f' %
          (epoch + 1, i + 1, average_precision))
    logging.info('[%d, %5d] average_precision_val: %.3f' %
                 (epoch + 1, i + 1, average_precision))

    # # Precision
    # running_precision = precision_score(label_truth, label_output)
    # # Recall
    # running_recall = recall_score(label_truth, label_output)
    # # Average Precision
    # running_ap = average_precision_score(label_truth, label_output)

    # print('[%d, %5d] precision_val: %.3f recall_val: %.3f average_precision_val: %.3f' %
    #       (epoch + 1, i + 1, running_precision, running_recall, running_ap))
    # logging.info('[%d, %5d] precision_val: %.3f recall_val: %.3f average_precision_val: %.3f' %
    #              (epoch + 1, i + 1, running_precision, running_recall, running_ap))




if __name__ == '__main__':
    positive_data_path = '/home/aaron/VIPCUP_Data/Positive_Data_Set'
    positive_label = 1
    negative_data_path = '/home/aaron/VIPCUP_Data/Negative_Data_Set'
    negative_label = 0

    train_data = load_data(positive_data_path, positive_label, negative_data_path, negative_label, is_train=True)
    train_loader = DataLoader(dataset=train_data, num_workers=4, batch_size=args.batch_size, pin_memory=False, shuffle=True)

    val_data = load_data(positive_data_path, positive_label, negative_data_path, negative_label, is_train=False)
    val_loader = DataLoader(dataset=val_data, num_workers=4, batch_size=args.batch_size, pin_memory=False, shuffle=False)

    alexnet = models.alexnet(pretrained=True)
    # print(alexnet)
    alexnet.classifier[6] = nn.Linear(4096, 2)
    # Extract features from the first fc
    # new_classifier = nn.Sequential(*list(alexnet.classifier.children())[:-5])
    # alexnet.classifier = new_classifier
    # print(alexnet)
    alexnet.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(alexnet.parameters(),
                           lr=args.lr,
                           weight_decay=0.0005,
                           betas=(0.9, 0.999),
                           amsgrad=True)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # ------------------------
    # Start Training and Validating
    # ------------------------
    save_step = 50
    for epoch in range(args.num_epochs):
        scheduler.step()
        train(alexnet, train_loader, optimizer, criterion, epoch)
        print('Validation')
        logging.info('Validation')
        validate(alexnet, val_loader, criterion, epoch)
        if epoch % save_step == 0:
            torch.save(alexnet.state_dict(), args.model_root)