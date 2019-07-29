from .dataloader import Data_Loader
from .transfer_learning_models import tf_Pretrained_Models
import torch.optim as optim
import torch.nn as nn

class Run_Model:

    def __init__(self, user, model_name, data_path):
        self.data_path = data_path
        self.dataloader = Data_Loader(data_path)
        self.df, self.num_of_classes = self.dataloader.get_data()

        self.model_name = model_name
        self.batch_size = 16
        self.criterion = nn.CrossEntropyLoss()
        self.model_path = '/home/muiz/Downloads/'
        self.trained_model_name = model_name + '_' + str(user)

        self.model = tf_Pretrained_Models(self.model_name, self.num_of_classes, self.df, self.batch_size, self.criterion)
        self.trained_model, self.optimizer, self.val_acc = self.model.run_transfer_learning()

        self.model.save_model(self.trained_model, self.optimizer, self.model_path, self.trained_model_name)

    def get_results(self):
        return self.val_acc
