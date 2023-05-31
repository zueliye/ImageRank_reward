from torch import nn
import torch.nn.functional as F
from dataloader import *
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import numpy as np
class MyDataset(Dataset):
    def __init__(self,input_path,label_path):
        super(MyDataset,self).__init__()
        self.input_root=input_path
        self.label_root=label_path

        input = torch.load(self.input_root)
        label = torch.load(self.label_root)
        inputs=[]
        labels=[]
        shuffle_ix = np.random.permutation(len(input))
        shuffle_ix = list(shuffle_ix)
        for i in shuffle_ix:
            inputs.append(input[i])
            labels.append(label[i])
        # inputs=inputs[shuffle_ix,:]
        # labels = labels[shuffle_ix, :]
        print(labels)
        self.input=inputs
        self.label=labels

    def __len__(self):
        return len(self.label)

    def __getitem__(self,item):
        input = self.input[item]
        label = self.label[item]
        return input,label
input_path = r'E:/img-prompt/input_test_add1.pt'
label_path = r'E:/img-prompt/label_test1_add1.pt'
dataset=MyDataset(input_path,label_path)
