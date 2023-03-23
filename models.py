import torch
import torch.nn as nn
from torch.nn import TransformerEncoder as TE
from torch.nn import TransformerEncoderLayer as TEL
from torch.nn import TransformerEncoderLayer as TEL
from torch.nn import TransformerDecoderLayer as DEL
from torch.nn import TransformerDecoder as DE
from torch.nn import Transformer
import torch.nn.functional as F
PAD_IDX = 3242# There actually is padding for some sequences
class BeatPositionEncoder(nn.Module):
    """
    To initialize this guy you pass in the dimension of your input embeddings, we choose 512
    they learn the embedding here
    """
    def __init__(self, emb_sz:int, beat_len=32, max_bar_len=21126):
        super().__init__()

        self.beat_len, self.max_bar_len = beat_len, max_bar_len
        self.beat_enc = nn.Embedding(beat_len, emb_sz, padding_idx=0)
        self.bar_enc = nn.Embedding(max_bar_len, emb_sz, padding_idx=0)
    
    def forward(self, pos):
        beat_enc = self.beat_enc(pos % self.beat_len)
        bar_pos = pos // self.beat_len % self.max_bar_len
        bar_pos[bar_pos >= self.max_bar_len] = self.max_bar_len - 1
        bar_enc = self.bar_enc((bar_pos))
        return beat_enc + bar_enc
class PositionalEncoding(nn.Module):
    """
    to initialize this guy pass in the dimension of the model, in our case 512
    we can also instead of this guys use nn.Embedding(num_items, dim_model)
    """
    "Encode the position with a sinusoid."
    def __init__(self, d:int): 
        super(PositionalEncoding, self).__init__()
        self.register_buffer('freq', 1 / (10000 ** (torch.arange(0., d, 2.)/d)))

    def forward(self, pos:torch.Tensor):
        inp = torch.ger(pos, self.freq)
        enc = torch.cat([inp.sin(), inp.cos()], dim=-1)
        return enc

def load_data():
    os.chdir("datasets/")
    os.chdir("adl-piano-midi/")
    for genre in os.listdir():
        os.chdir(genre)
        for midi_file in os.listdir():
            new_thing =MusicItem.from_file(item, MusicVocab.create())
            # print(new_thing)
            items.append(new_thing)
        os.chdir("..")

class BaseMusicModel(nn.Module):
    def __init__(self,dim_per_head,num_layers,num_heads,vocab_sz,learned_positions = False, beat_positions = True,device = "cpu"):
        super(BaseMusicModel, self).__init__()
        d_model = dim_per_head*num_heads
        self.device = device
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_sz,d_model)
        self.pos_enc = nn.Embedding(512,d_model) \
        if learned_positions else PositionalEncoding(d_model)
        self.beat_enc = BeatPositionEncoder(d_model)
        self.beat_positions = beat_positions

        self.encoder_layer = TEL(d_model = dim_per_head*num_heads,nhead = num_heads, dim_feedforward=256)
        self.transformer = TE(self.encoder_layer,num_layers)
    
    def forward(self,music_inputs,music_genres,music_beats):
        embedding = self.embed(music_inputs)
        ar = torch.arange(music_inputs.shape[1]).to(self.device)
        position_embedding = self.pos_enc(ar)
        beat_encoding = 0
        if self.beat_positions:
            beat_encoding = self.beat_enc(music_beats)
        entire_embedding = beat_encoding+position_embedding+embedding

        #this is because it is sequence first
        entire_embedding = entire_embedding.transpose(0,1)
        mask = Transformer.generate_square_subsequent_mask(music_inputs.shape[1]).to(self.device)
        src_padding_mask = (music_inputs == PAD_IDX).to(self.device)
        #THere is padding fyi

        transformer_output = self.transformer(entire_embedding,mask=mask,src_key_padding_mask=src_padding_mask)
        output = transformer_output@self.embed.weight.T
        
        output = output.transpose(0,1)
        return output

class InterpolatorMusicModel(nn.Module):
    def __init__(self,dim_per_head,num_layers,num_heads,vocab_sz,learned_positions = False, beat_positions = True,device = "cpu"):
        super(InterpolatorMusicModel, self).__init__()
        d_model = dim_per_head*num_heads
        self.device = device
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_sz,d_model)
        self.pos_enc = nn.Embedding(512,d_model) \
        if learned_positions else PositionalEncoding(d_model)
        self.beat_enc = BeatPositionEncoder(d_model)
        self.beat_positions = beat_positions
        self.genre_embeddings = nn.Embedding(18,d_model)

        self.encoder_layer = DEL(d_model = dim_per_head*num_heads,nhead = num_heads, dim_feedforward=256)
        self.transformer = DE(self.encoder_layer,num_layers)
    
    def forward(self,music_inputs,music_genres,music_beats):
        embedding = self.embed(music_inputs)
        ar = torch.arange(music_inputs.shape[1]).to(self.device)
        music_genres = music_genres@self.genre_embeddings.weight
        music_genres = music_genres.transpose(0,1)
        position_embedding = self.pos_enc(ar)
        beat_encoding = 0
        if self.beat_positions:
            beat_encoding = self.beat_enc(music_beats)
        entire_embedding = beat_encoding+position_embedding+embedding

        #this is because it is sequence first
        entire_embedding = entire_embedding.transpose(0,1)
        mask = Transformer.generate_square_subsequent_mask(music_inputs.shape[1]).to(self.device)
        src_padding_mask = (music_inputs == PAD_IDX).to(self.device)
        #THere is padding fyi
        transformer_output = self.transformer(entire_embedding,music_genres,tgt_mask=mask,tgt_key_padding_mask=src_padding_mask)
        output = transformer_output@self.embed.weight.T
        
        output = output.transpose(0,1)
        return output
        

class MusicTensorDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

class MusicData:
    def __init__(self,inputs : torch.Tensor, labels: torch.Tensor,genre: torch.Tensor, beat_positions: torch.Tensor):
        self.x = inputs
        self.y = labels
        self.genre = genre
        self.beat_positions = beat_positions
        
