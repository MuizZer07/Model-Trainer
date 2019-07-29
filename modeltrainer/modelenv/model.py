from dataloader import Data_Loader
from transfer_learning_models import tf_Pretrained_Models
import torch.optim as optim
import torch.nn as nn

dataloader = Data_Loader('/home/muiz/Downloads/idc_small.csv')
df, num_of_classes = dataloader.get_data()

model_name = 'resnet'
batch_size = 16
criterion = nn.CrossEntropyLoss()

model = tf_Pretrained_Models(model_name, num_of_classes, df, batch_size, criterion)
trained_model, optimizer = model.run_transfer_learning()

model.save_model(trained_model, optimizer, '/home/muiz/Downloads/', 'idc_resnet')
