from torch.utils.data import Dataset,DataLoader
import torch

class MyDataset(Dataset):
    def __init__(self,input_path,label_path):
        super(MyDataset,self).__init__()
        self.input_root=input_path
        self.label_root=label_path

        input = torch.load(self.input_root)
        label = torch.load(self.label_root)
        print(input)
        print(label)
        inputs=[]
        labels=[]
        for i in range(len(input)):
            inputs.append(input[i])
            labels.append(label[i])

        self.input=inputs
        self.label=labels

    def __len__(self):
        return len(self.label)

    def __getitem__(self,item):
        input = self.input[item]
        label = self.label[item]
        return input,label