import numpy as np
import os
from musicautobot.numpy_encode import *
from musicautobot.utils.file_processing import process_all, process_file
from musicautobot.config import *
from musicautobot.music_transformer import *
from musicautobot.music_transformer.dataloader import MusicDataBunch
from pathlib import Path
from scripts.models import BeatPositionEncoder,MusicData
import torch
from tqdm import tqdm


def get_music_items(ctx_size,split_tensors=False):
    dataset = []
    os.chdir("datasets/")
    os.chdir("adl-piano-midi/")
    all_folders = os.listdir()
    bad_files = 1
    max_len = 0
    genre_to_number = {}
    dataset_inputs = torch.zeros((1,ctx_size))
    dataset_outputs = torch.zeros((1,ctx_size))
    dataset_genres = torch.zeros((1,1))
    dataset_beats = torch.zeros((1,ctx_size))
    for i, folder in enumerate(all_folders):
        # print(dataset_inputs.size)
        os.chdir(folder + "/")
        # music_items[folder] = []
        # dataset = []
        tracks = os.listdir()
        genre = folder
        genre_to_number[genre] = i
        print(f"Reading through {folder}")
        for track in tqdm(tracks):
            try:
                item = MusicItem.from_file(track, MusicVocab.create())
            except:
                print(f"There are now {bad_files} bad files")
                bad_files += 1
                continue
            # music_items[folder].append(item)

            inputs = item.to_tensor(device="cpu")[:-1]
            outputs = item.to_tensor(device="cpu")[1:]
            beat_positions = item.get_pos_tensor(device = "cpu")[:-1]# this just makes it the same size as our other guys
            if (len(beat_positions) > max_len):
                max_len = len(beat_positions)
                print(f"the maxlength of an item is now {max_len}")
            if(split_tensors):
                inputs = split_items(inputs, ctx_size)
                outputs = split_items(outputs, ctx_size)
                beat_positions = split_items(beat_positions, ctx_size)
            #312 is padding
            pad = False
            if pad == True:
                padder_i = torch.full((1024*40,),10000)
                padder_o = torch.full((1024*40,),10000)
                padder_b = torch.full((1024*40,),10000)
                padder_i[:len(inputs)] = inputs
                padder_o[:len(outputs)] = outputs
                padder_b[:len(beat_positions)] = beat_positions
                inputs = padder_i
                outputs = padder_o
                beat_positions = padder_b
            dataset_inputs = torch.vstack((dataset_inputs,inputs))
            dataset_outputs = torch.vstack((dataset_outputs,outputs))
            dataset_beats = torch.vstack((dataset_beats,beat_positions))
            dataset_genres = torch.vstack((dataset_genres,torch.full((inputs.shape[0],1),i)))

        


        os.chdir("..")
    os.chdir("..")
    os.chdir("..")
    with open(f"genre_converter.json", "w") as outfile:
        outfile.write(json.dumps(genre_to_number,indent=4))
    torch.save(dataset_inputs,"inputs.pt")
    torch.save(dataset_outputs,"outputs.pt")
    torch.save(dataset_beats,"beats.pt")
    torch.save(dataset_genres,"genres.pt")
        # os.chdir("datasets/")
        # os.chdir("adl-piano-midi/")
    
    # return music_items

def split_items(item,ctx_size):
    splits = torch.split(item,ctx_size)
    single_tens =False
    if len(splits[-1]) != ctx_size:
        if len(splits) != 1:
            ending = torch.cat((splits[-2][len(splits[-1]):], splits[-1]))
        else:
            single_tens = True
            padder_i = torch.full((1,ctx_size),10000)
            padder_i[0,:len(splits[0])] = splits[0]
            ending = padder_i
    else:
        ending = splits[-1]
    if not single_tens:
        ret = torch.vstack(splits[:-1])
        ret = torch.vstack((ret,ending))
    else:
        ret = ending
    return ret

def create_tensors(ctx_size,split_tensors = True):
    music_items = get_music_items()
    dataset = []
    for genre, genre_list in music_items:
        print(f"processing {genre}")
        for item in tqdm(genre_list):
            inputs = item.to_tensor(device="cpu")[:-1]
            outputs = item.to_tensor(device="cpu")[1:]
            beat_positions = item.get_pos_tensor(device = "cpu")[:-1]# this just makes it the same size as our other guys

            if(split_tensors):
                inputs = split_items(inputs, ctx_size)
                outputs = split_items(outputs, ctx_size)
                beat_positions = split_items(beat_positions, ctx_size)
            data = MusicData(inputs,outputs,genre,beat_positions)
            dataset.append(data)
    with open("150_size.json", "w") as outfile:
        outfile.write(json.dumps(dataset,indent=4))



get_music_items(512,True)

