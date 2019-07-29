import pandas as pd

class Data_Loader:

    def __init__(self, filename):
        self.df = pd.read_pickle(filename)
        self.num_of_classes = len(self.df.groupby("labels").count())

    def get_data(self):
        return self.df, self.num_of_classes
