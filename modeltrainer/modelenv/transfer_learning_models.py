from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from .preprocess import Preprocess

class tf_Pretrained_Models:

    def __init__(self, model_name, num_classes, dataframe, batch_size, criterion, use_pretrained=True):
        self.model_name = model_name
        self.model = None
        self.input_size = 0
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained
        self.criterion = criterion
        self.device = "cpu"

        self.initialize_model()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        processed_data = Preprocess(dataframe, self.input_size, batch_size)
        self.dataloaders = processed_data.get_data()

    def run_transfer_learning(self):
        self.train(num_epochs=5, is_inception=(self.model_name=="inception"))
        self.set_parameter_requires_grad(True)
        _, val_acc = self.train(num_epochs=2, is_inception=(self.model_name=="inception"))

        return self.model, self.optimizer, val_acc

    def initialize_model(self):

        if self.model_name == "resnet":
            """ Resnet18
            """
            self.model = models.resnet18(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(False)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.num_classes)
            self.input_size = 224

        elif self.model_name == "alexnet":
            """ Alexnet
            """
            self.model = models.alexnet(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(False)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
            self.input_size = 224

        elif self.model_name == "vgg":
            """ VGG11_bn
            """
            self.model = models.vgg11_bn(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(False)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
            self.input_size = 224

        elif self.model_name == "squeezenet":
            """ Squeezenet
            """
            self.model = models.squeezenet1_0(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(False)
            self.model.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1,1), stride=(1,1))
            self.model.num_classes = self.num_classes
            self.input_size = 224

        elif self.model_name == "densenet":
            """ Densenet
            """
            self.model = models.densenet121(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(False)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, self.num_classes)
            self.input_size = 224

        elif self.model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            self.model = models.inception_v3(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(False)
            # Handle the auxilary net
            num_ftrs = self.model.AuxLogits.fc.in_features
            self.model.AuxLogits.fc = nn.Linear(num_ftrs, self.num_classes)
            # Handle the primary net
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs,self.num_classes)
            self.input_size = 299

    def set_parameter_requires_grad(self, feature_extracting):
        for param in self.model.parameters():
            param.requires_grad = feature_extracting

    def train(self, num_epochs=25, is_inception=False):
        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = self.model(inputs)
                            loss1 = self.criterion(outputs, labels)
                            loss2 = self.criterion(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(self.dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model, val_acc_history

    def save_model(self, model, optimizer, save_path, check_point_name):

        model_name = check_point_name + '.pth'
        model_path = save_path + model_name
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_net_state_dict': optimizer.state_dict(),
        }, model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_net_state_dict'])

        return model, optimizer
