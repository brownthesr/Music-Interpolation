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
    """
    This is for sampling from the regular transformer
    """
    model.eval()
    test_item.data = test_item.data[:2]
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
    """
    This is for sampling from the conditioned transformer
    """
    model.eval()
    test_item.data = test_item.data[:2]
    for i in range(max_len):
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
    try:
        item.to_stream(bpm = 120).write("midi",path + ".mid")
    except:
        print("error")

def tfm_transpose(x, value, vocab):
    """
    This transposes the music notes to a different key
    """
    x[(x >= vocab.note_range[0]) & (x < vocab.note_range[1])] += value
    return x

def train(model,epochs):
    """
    This is used for training regular decoder only Transformers with no conditioning
    """

    euph = MusicItem.from_file("datasets/Euphonium_is_the_best.mid", MusicVocab.create())
    inputs = torch.load("pt_data/inputs.pt")[1:].to(device)
    outputs = torch.load("pt_data/outputs.pt")[1:].to(device)
    beats = torch.load("pt_data/beats.pt")[1:].to(device)
    genres = torch.load("pt_data/genres.pt")[1:].to(device)
    inputs = torch.where(inputs == 10000.0, torch.Tensor([3242]).to(device),inputs)
    outputs = torch.where(outputs == 10000.0, torch.Tensor([3242]).to(device),outputs)

    perm = torch.randperm(71343).to(device)
    inputs = inputs[perm].long()
    outputs = outputs[perm].long()
    beats = beats[perm].long()
    genres = genres[perm].long()

    split_inputs = torch.split(inputs,64)
    split_outputs = torch.split(outputs,64)
    split_beats = torch.split(beats,64)
    split_genres = torch.split(genres,64)

    inputs = torch.stack(split_inputs[:-1],dim=0).to(device)
    outputs = torch.stack(split_outputs[:-1],dim=0).to(device)
    beats = torch.stack(split_beats[:-1],dim=0).to(device)
    genres = torch.stack(split_genres[:-1],dim=0).to(device)

    val_inputs = split_inputs[-1].to(device)
    val_outputs = split_outputs[-1].to(device)
    val_beats = split_beats[-1].to(device)
    val_genres = split_genres[-1].to(device)

    losses = []
    optimizer = optim.Adam(model.parameters(),lr=.0001, weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    model = model.cuda()

    for epoch in range(epochs):
        for  transposition in range(1):
            for batch_idx,(x,y,beat,genre) in enumerate(zip(inputs,outputs,beats,genres)):
                x = tfm_transpose(x,transposition,MusicVocab.create())
                y = tfm_transpose(y,transposition,MusicVocab.create())
                output = model(x,genre,beat)

                y = y.view(-1)
                output = output.reshape(-1,output.shape[-1])

                loss = F.cross_entropy(output, y)

                with torch.no_grad():
                    val_inputs = tfm_transpose(val_inputs,transposition,MusicVocab.create())
                    val_outputs = tfm_transpose(val_outputs,transposition,MusicVocab.create())

                    val_out = model(val_inputs,val_genres,val_beats)
                    val_out = val_out.reshape(-1,val_out.shape[-1])
                    val_outputs = val_outputs.view(-1)
                    val_loss = F.cross_entropy(val_out,val_outputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print(f'Train Epoch: {epoch} {transposition} [{batch_idx}/{len(inputs)} ({batch_idx/len(inputs):.2f}%)]\tLoss: {loss.item():.6f}')
                    losses.append([loss.item(),val_loss.item()])
                    sys.stdout.flush()
        with torch.no_grad():
            sample(model,euph,str(model.num_layers) +"_layer/" +"epoch"+str(epoch)+ "_"+str(transposition/2),350)
        # scheduler.step()
    torch.save(model.state_dict(), str(model.num_layers) +"_layer/" +"musicformer.pt")
    losses = np.array(losses)
    np.save(str(model.num_layers) +"_layer/"+"losses.npy",losses)

def train_d(model,epochs):
    """
    This is for training conditioned transformers
    """
    euph = MusicItem.from_file("datasets/Euphonium_is_the_best.mid", MusicVocab.create())
    print("loading inputs")
    inputs = torch.load("pt_data/inputs.pt")[1:].to(device)
    print("loading outputs")
    outputs = torch.load("pt_data/outputs.pt")[1:].to(device)
    print("loading beats")
    beats = torch.load("pt_data/beats.pt")[1:].to(device)
    print("loading genres")
    genres = torch.load("pt_data/genres.pt")[1:].to(device)
    inputs = torch.where(inputs == 10000.0, torch.Tensor([3242]).to(device),inputs)
    outputs = torch.where(outputs == 10000.0, torch.Tensor([3242]).to(device),outputs)

    perm = torch.randperm(71343).to(device)
    inputs = inputs[perm].long()
    outputs = outputs[perm].long()
    beats = beats[perm].long()
    genres = genres[perm].long()

    split_inputs = torch.split(inputs,64)
    split_outputs = torch.split(outputs,64)
    split_beats = torch.split(beats,64)
    split_genres = torch.split(genres,64)

    inputs = torch.stack(split_inputs[:-1],dim=0).to(device)
    outputs = torch.stack(split_outputs[:-1],dim=0).to(device)
    beats = torch.stack(split_beats[:-1],dim=0).to(device)
    genres = torch.stack(split_genres[:-1],dim=0).to(device)

    val_inputs = split_inputs[-1].to(device)
    val_outputs = split_outputs[-1].to(device)
    val_beats = split_beats[-1].to(device)
    val_genres = split_genres[-1].to(device)
    val_genres = F.one_hot(val_genres,num_classes=18).float()

    #                                         # we then loop through each batch
    # outputs = torch.randint(0,10,(166,64,512)).long().to(device)
    # beats = torch.randint(0,10,(166,64,512)).long().to(device)
    # genres = torch.randint(0,10,(166,64,1)).long().to(device)

    losses = []
    optimizer = optim.Adam(model.parameters(),lr=.0001, weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    model = model.cuda()

    for epoch in range(epochs):
        for  transposition in range(1):
            for batch_idx,(x,y,beat,genre) in enumerate(zip(inputs,outputs,beats,genres)):
                x = tfm_transpose(x,transposition,MusicVocab.create())
                y = tfm_transpose(y,transposition,MusicVocab.create())
                genre = F.one_hot(genre,num_classes=18).float()
                output = model(x,genre,beat)

                y = y.view(-1)
                output = output.reshape(-1,output.shape[-1])

                loss = F.cross_entropy(output, y)

                with torch.no_grad():
                    val_inputs = tfm_transpose(val_inputs,transposition,MusicVocab.create())
                    val_outputs = tfm_transpose(val_outputs,transposition,MusicVocab.create())
                    val_out = model(val_inputs,val_genres,val_beats)
                    val_out = val_out.reshape(-1,val_out.shape[-1])
                    val_outputs = val_outputs.view(-1)
                    val_loss = F.cross_entropy(val_out,val_outputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print(f'Train Epoch: {epoch} {transposition} [{batch_idx}/{len(inputs)} ({batch_idx/len(inputs):.2f}%)]\tLoss: {loss.item():.6f}')
                    losses.append([loss.item(),val_loss.item()])
                    sys.stdout.flush()
        with torch.no_grad():
            sample_d(model,euph,str(model.num_layers) +"_layerd/" +"epoch"+str(epoch)+ "_"+str(transposition/2),350,14)
        # scheduler.step()
    torch.save(model.state_dict(), str(model.num_layers) +"_layerd/" +"musicformer.pt")
    losses = np.array(losses)
    np.save(str(model.num_layers) +"_layerd/"+"losses.npy",losses)
    for i in range(18):
        sample_d(model,euph,str(model.num_layers) +"_layerd/" +"genre " + str(i),350,i)

        # print("generating batches")
        # inputs = 

device = "cuda"
model = InterpolatorMusicModel(dim_per_head=64,num_heads=12,num_layers=12,vocab_sz=3242+1,device=device)
epochs = 40
train_d(model,epochs)