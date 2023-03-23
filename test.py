import torch
import torch.nn as nn
import torch.nn.functional as f
from models import *
import torch.optim as optim
import sys
import numpy as np
from musicautobot.numpy_encode import *
from torch.optim.lr_scheduler import StepLR

from musicautobot.utils.file_processing import process_all, process_file
from musicautobot.config import *
from musicautobot.music_transformer import *
from musicautobot.music_transformer.dataloader import MusicDataBunch
from pathlib import Path
from models import BeatPositionEncoder
from torch.nn import TransformerEncoder as TE
from torch.nn import TransformerEncoderLayer as TEL
from torch.nn import Transformer
import torch

def sample(model,test_item,path, max_len):
    model.eval()
    test_item.data = test_item.data[:1]
    for i in range(max_len):
        test = test_item.to_tensor(device).long().to(device)
        test_beats = test_item.get_pos_tensor(device).long().to(device)
        test_genre = torch.Tensor([0]).long().to(device)
        outs = model(test.unsqueeze(0),test_genre.unsqueeze(0),test_beats.unsqueeze(0))
        outs = outs.squeeze(0)
        outs = f.softmax(outs,dim=1)
        out = outs[-1]
        choice = torch.multinomial(out,1,True)
        test_item = MusicItem(np.append(test_item.data,choice.item()),MusicVocab.create())
    item = MusicItem(test_item.data,MusicVocab.create())
    item.to_stream(bpm = 120).write("midi",path + ".mid")
def sample_d(model,test_item,path, max_len,gen):
    model.eval()
    test_item.data = test_item.data[:1]
    for i in range(max_len):
        print(i/max_len)
        test = test_item.to_tensor(device).long().to(device)
        test_beats = test_item.get_pos_tensor(device).long().to(device)
        test_genre = torch.Tensor([gen]).long().to(device)
        test_genre = F.one_hot(test_genre,num_classes=18).float()
        outs = model(test.unsqueeze(0),test_genre.unsqueeze(0),test_beats.unsqueeze(0))
        outs = outs.squeeze(0)
        outs = f.softmax(outs,dim=1)
        out = outs[-1]
        choice = torch.multinomial(out,1,True)
        test_item = MusicItem(np.append(test_item.data,choice.item()),MusicVocab.create())
    item = MusicItem(test_item.data,MusicVocab.create())
    item.to_stream(bpm = 120).write("midi",path + ".mid")
def sample_d_int(model,test_item,path, max_len,gen1,w1,gen2,w2):
    model.eval()
    test_item.data = test_item.data[:1]
    for i in range(max_len):
        test = test_item.to_tensor(device).long().to(device)
        test_beats = test_item.get_pos_tensor(device).long().to(device)
        test_genre1 = torch.Tensor([gen1]).long().to(device)
        test_genre1 = F.one_hot(test_genre1,num_classes=18).float()
        test_genre2 = torch.Tensor([gen2]).long().to(device)
        test_genre2 = F.one_hot(test_genre2,num_classes=18).float()
        test_genre = w1*test_genre1+w2*test_genre2
        outs = model(test.unsqueeze(0),test_genre.unsqueeze(0),test_beats.unsqueeze(0))
        outs = outs.squeeze(0)
        outs = f.softmax(outs,dim=1)
        out = outs[-1]
        choice = torch.multinomial(out,1,True)
        test_item = MusicItem(np.append(test_item.data,choice.item()),MusicVocab.create())
    item = MusicItem(test_item.data,MusicVocab.create())
    try:
        item.to_stream(bpm = 120).write("midi",path + ".mid")
    except:
        print("error")

device = "cuda"
print("loading model")
model = InterpolatorMusicModel(dim_per_head=64,num_heads=8,num_layers=8,vocab_sz=3242+1,device=device)
model.load_state_dict(torch.load("8_layerd/musicformer.pt"))
print("model_loaded")
model = model.to(device)
model.device = device
print("retriving music")
euph = MusicItem.from_file("datasets/Euphonium_is_the_best.mid", MusicVocab.create())
print(euph.data)
# print(euph.to_tensor(device))

# print(euph.data)
with torch.no_grad():
    # for i in range(18):
    #     print(f"generating music {i}")
    #     sample_d(model,euph,f"test_{i}",350,i)
    # for x in np.linspace(.5,100,100):
        # print(x)
    sample_d_int(model,euph,f"tests/jazz_classical",1024,7,.5,14,.5)