import torch
from torch.utils.data import Dataset


import pickle


class pickle_dataloader(Dataset):
      def __init__(self,img_path,transform=None):
          self.img_path=pickle.load(open(img_path,'rb'))
          self.transform=transform
          self.images=[]
          self.text=[]
          self.label=[]
          for i in (self.img_path.keys()):
              item=self.img_path[i]
              self.images.append(item.pixels)
              self.text.append(item.text)
              self.label.append(item.label)
              #self.text=self.text.append(self.img_path[i].text)
              #self.label=self.label.append(self.img_path[i].label)

      def __len__(self):
          return len(self.images)

      def __getitem__(self,index):
          return self.images[index],self.text[index],self.label[index]
               



data=pickle_dataloader('/Users/tanmoy/research/Fake_News/data/weibo_train.pkl')
from torch.utils.data import DataLoader

train_dataloader = DataLoader(data, batch_size=64, shuffle=True)

from sentence_transformers import SentenceTransformer 
model = SentenceTransformer('paraphrase-MiniLM-L6-v2') 

for (i,j,k) in train_dataloader:
    print (model.encode(j))

