import datetime
import os
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision import models
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, BertTokenizer,DistilBertModel, DistilBertConfig, DistilBertTokenizer
import wandb

import glob
import pdb

import argparse

from dataloader import pickle_dataloader


if __name__=="__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('--dataset', type=str, default="gossip", help='i/p data')
   parser.add_argument('--batch_size',default=32,type=int)
   parser.add_argument('--image_encoder_lr',default=1e-4,type=int)
   parser.add_argument('--text_encoder_lr',default=1e-5,type=int)
   parser.add_argument('--weight_decay',default=1e-3,type=int)
   parser.add_argument('--epochs',default=4,type=int)

   #Model parameters
   parser.add_argument('--vis_model',default='ResNet50',type=str)
   parser.add_argument('--image_embedding',default=2048,type=int)
   parser.add_argument('--text_model',default= "distilbert-base-uncased",type=str)
   parser.add_argument('--text_embedding',default = 768, type=int)
   parser.add_argument('--text_tokenizer',default = "distilbert-base-uncased",type=str)
   parser.add_argument('--max_length',default = 200, type=int)
   
   parser.add_argument('--pretrained', default = True,type=str,help=' # for both image encoder and text encoder')
   parser.add_argument('--trainable',default = True,type=str,help=' # for both image encoder and text encoder')
   parser.add_argument('--temperature',default = 1.0,type=str)
   
   parser.add_argument('--image_size',default=224,type=int)
   # for projection head; used for both image and text encoders
   parser.add_argument('--num_projection_layers',default = 1,type=int)
   parser.add_argument('--projection_dim',default = 256,type=int) 
   parser.add_argument('--dropout',default = 0.1,type=float)
   
   args=parser.parse_args()

   #if args.data=='gossip':
          
   if args.dataset=='gossip':
      res=glob.glob('/media/steven/WDD/research/data/Fake_Media/'+args.dataset+'_*.pkl')
      train_data=[i for i in res if 'train' in i][0]
      test_data=[i for i in res if 'test' in i][0]
   elif args.dataset=='politi':
        res=glob.glob('/media/steven/WDD/research/data/Fake_Media/'+args.dataset+'_*.pkl')
        train_data=[i for i in res if 'train' in i][0]
        test_data=[i for i in res if 'test' in i][0]
   else:
        res=glob.glob('/media/steven/WDD/research/data/Fake_Media/'+args.dataset+'_*.pkl')
        train_data=[i for i in res if 'train' in i][0]
        test_data=[i for i in res if 'test' in i][0]

   if args.vis_model=='ResNet50':
      from torchvision.models import resnet50, ResNet50_Weights
      vis_model=resnet50(weights="IMAGENET1K_V1")
   TEXT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # 471 MB
   IMAGE_MODEL = "convnext_tiny_384_in22ft1k"
   
   train_datas=pickle_dataloader(train_data)
   test_datas=pickle_dataloader(test_data)

   train_dataloader = DataLoader(train_datas, batch_size=args.batch_size, shuffle=True)
   test_dataloader = DataLoader(test_datas, batch_size=args.batch_size, shuffle=False)

   for (i,j,k) in train_dataloader:
       print (i)

       

