import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CodeSnippetsDataset(Dataset):

    def __init__(self, file_names):

        pos_df = pd.read_csv(file_names[0],delimiter=';')
        neg_df = pd.read_csv(file_names[1], delimiter=';')
        if 'otalLinesOfCode' in pos_df.columns:
            pos_df=pos_df.rename(columns={'otalLinesOfCode':'TotalLinesOfCode'})

        if 'Score' not in pos_df.columns:
                pos_df['Score']=np.nan

        conc_df = pd.concat([pos_df.reset_index(drop=True), neg_df.reset_index(drop=True)
                             ], axis=0, ignore_index=True, )
        collnames=list(neg_df.columns)
        conc_df=conc_df[collnames]


        g = conc_df.groupby('label')
        conc_df=g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))

        self.x = conc_df.iloc[:, 0:78].values
        self.y = conc_df.iloc[:, 80].values
        self.x_train = torch.tensor(self.x, dtype=torch.float32)
        self.y_train = torch.tensor(self.y, dtype=torch.long)
    def get_all_data(self,model_type='cnn'):
        if model_type=='cnn':
            return torch.tensor(self.x, dtype=torch.float32),torch.tensor(self.y, dtype=torch.long)
        else:
            return self.x,self.y
    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]
