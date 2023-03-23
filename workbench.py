import os
# os.chdir("musicautobot/")
from musicautobot.numpy_encode import *
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

# this sometimes throws an error, if it does just try and catch it\
# os.chdir("datasets/")
# os.chdir("adl-piano-midi/")
# os.chdir("Blues/")
# items = []
# for item in os.listdir():
#     new_thing =MusicItem.from_file(item, MusicVocab.create())
#     print(new_thing)
#     items.append(new_thing)
# new_thing =MusicItem.from_file(os.listdir()[0], MusicVocab.create())
# os.chdir("..")
# os.chdir("Rock/")
# another = MusicItem.from_file(os.listdir()[1], MusicVocab.create())
# print(len(new_thing.data),len(another.data))
# new_thing.data = new_thing.data[:len(another.data)]
# another.data = another.data[:len(new_thing.data)]

# new_data = np.empty((new_thing.data.size + another.data.size,), dtype=new_thing.data.dtype)
# new_data[0::4] = new_thing.data[0::2]
# new_data[1::4] = new_thing.data[1::2]
# new_data[2::4] = another.data[0::2]
# new_data[3::4] = another.data[1::2]
# last = MusicItem(new_data,MusicVocab.create())
# print(new_thing.data)
# os.chdir("..")
# os.chdir("..")
# os.chdir("..")
# item.to_text()
# print(dir(item))
# initial =item.to_tensor(device="cpu")
# print(item.get_pos_tensor(device="cpu").shape)
# print(len(item.vocab.itos))

# emb = torch.nn.Embedding(len(item.vocab.itos),512)
# enc = BeatPositionEncoder(512)
# new_enc = enc(item.get_pos_tensor(device="cpu"))
# print(new_enc, new_enc.shape)
# new_emb = emb(item.to_tensor(device="cpu"))
# retur = torch.argmax(new_emb@emb.weight.T,axis=1)
# print(new_emb,new_emb.shape)
# print((retur == initial).sum()/5836)
# # print(dir(item.vocab))

# print(item.vocab.textify(3))
# print((last.to_stream(bpm = 100).write("midi",os.getcwd() + "/other.mid")))
# (new_thing.to_stream(bpm = 100).write("midi",os.getcwd() + "/first.mid"))
# (another.to_stream(bpm = 100).write("midi",os.getcwd() + "/second.mid"))
# print(len(another.vocab.itos))
# print(len(new_thing.vocab.itos))

# encoder_layer = TEL(d_model = 2,nhead = 2,dim_feedforward=256)
# transformer = TE(encoder_layer,2)
# a = torch.Tensor([[[1,2],[3,4]],[[5,8],[7,8]]])
# mask = Transformer.generate_square_subsequent_mask(2,2)
# src_padding_mask = (a == 8).transpose(1, 2)
# print(src_padding_mask)
# print(mask)

# print(transformer(a,mask=mask,src_key_padding_mask=src_padding_mask))
# # item.play()
# # processors = [Midi2ItemProcessor()]
# # midi_files = get_files("datasets/adl-piano-midi/",recurse = True)
# print(type(items))
# print(items)
import numpy as np
from scipy.signal import convolve

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
data = np.loadtxt("jazz_scaled.txt")
print(data)
ksize = 5
sigma = 1
kernel = np.exp(-(np.arange(-ksize, ksize+1)**2) / (2 * sigma**2))
kernel /= kernel.sum()
data[:,1] = convolve(data[:,1],kernel,mode="same")
data[:,1] = convolve(data[:,1],kernel,mode="same")
sns.lineplot(x=data[:,0],y=data[:,1])
plt.savefig("scaled_jazz.png")
