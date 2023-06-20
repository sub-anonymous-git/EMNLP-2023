import numpy as np
import random

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig

from utils import to_gpu
from utils import ReverseLayerF


def masked_mean(tensor, mask, dim):
    """Finding the mean along dim"""
    masked = torch.mul(tensor, mask)
    return masked.sum(dim=dim) / mask.sum(dim=dim)

def masked_max(tensor, mask, dim):
    """Finding the max along dim"""
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    neg_inf[~mask] = -math.inf
    return (masked + neg_inf).max(dim=dim)



# let's define a simple model that can deal with multimodal variable length sequence
class fuse(nn.Module):
    def __init__(self, config):
        super(ZEE, self).__init__()

        
        self.text_size = config
        self.visual_size = 512
    


        self.input_sizes = input_sizes = [self.text_size, self.visual_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size)]
        self.output_size = output_size = 7
        self.dropout_rate = dropout_rate = 0.03
        self.activation = nn.ELU()
        self.tanh = nn.Tanh()
        
        
        rnn =  nn.GRU
        # defining modules - two layer bidirectional LSTM with layer norm in between

        
        #self.embed = nn.Embedding(len(config.word2id), input_sizes[0])
        self.trnn1 = rnn(input_sizes[0], hidden_sizes[0], bidirectional=True)
        self.trnn2 = rnn(2*hidden_sizes[0], hidden_sizes[0], bidirectional=True)
        
        self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.vrnn2 = rnn(2*hidden_sizes[1], hidden_sizes[1], bidirectional=True)
        
        self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = rnn(2*hidden_sizes[2], hidden_sizes[2], bidirectional=True)



        ##########################################
        # mapping modalities to same sized space
        ##########################################
       
        self.project_t = nn.Sequential()
        self.project_t.add_module('project_t', nn.Linear(in_features=hidden_sizes[0], out_features=64))
        self.project_t.add_module('project_t_activation', self.activation)
        self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(64))
        

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=hidden_sizes[1], out_features=64))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(64))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=hidden_sizes[2], out_features=64))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(64))


        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=64, out_features=64))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=64, out_features=64))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        
        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3', nn.Linear(in_features=64, out_features=64))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())
        

        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=64, out_features=64))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())


        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=64, out_features=64))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=64, out_features=64))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=64, out_features=64))



        ##########################################
        # shared space adversarial discriminator
        ##########################################
        self.discriminator = nn.Sequential()
        self.discriminator.add_module('discriminator_layer_1', nn.Linear(in_features=64, out_features=64))
        self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
        self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(dropout_rate))
        self.discriminator.add_module('discriminator_layer_2', nn.Linear(in_features=64, out_features=len(hidden_sizes)))

        ##########################################
        # shared-private collaborative discriminator
        ##########################################

        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1', nn.Linear(in_features=64, out_features=4))



        # self.fusion = nn.Sequential()
        # self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=64*6, out_features=64*3))
        # self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        # self.fusion.add_module('fusion_layer_1_activation', self.activation)
        #self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=64*3, out_features= output_size))

        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0]*2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2]*2,))


        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4)
        print("hello there")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        

        
    
    #original ZEE
    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        #packed_sequence = pack_padded_sequence(sequence, lengths)

        
        packed_h1, final_h1 = rnn1(sequence)

        

        #padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(packed_h1)
        #packed_normed_h1 = pack_padded_sequence(normed_h1, lengths)


        _, final_h2 = rnn2(normed_h1)
        #print("final_h1",final_h1.shape)
        

        return final_h1, final_h2

    def alignment(self, sentences, visual, lengths):
        
        batch_size = 32
        
        # extract features from text modality
        #sentences = self.embed(sentences)
       
        # final_h1t, final_h2t = self.extract_features(sentences, lengths, self.trnn1, self.trnn2, self.tlayer_norm)
        # utterance_text = torch.cat((final_h1t, final_h2t), dim=2).permute(0, 1, 2).contiguous()
        # #print(utterance_text.shape)



        # # extract features from visual modality
        # final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        # utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(0, 1, 2).contiguous()
        # #print(utterance_video.shape)
        # # extract features from modality
        # final_h1a, final_h2a = self.extract_features(lengths, self.arnn1, self.arnn2, self.alayer_norm)
        # utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(0, 1, 2).contiguous()
        #print(utterance_audio.shape)

        # Shared-private encoders
        self.shared_private(sentences, visual)

        #ZEE original
        #self.shared_private(utterance_text, utterance_video, utterance_audio)

        
        reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, 1.0)
        reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, 1.0)
        reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, 1.0)

        self.domain_label_t = self.discriminator(reversed_shared_code_t)
        self.domain_label_v = self.discriminator(reversed_shared_code_v)
        self.domain_label_a = self.discriminator(reversed_shared_code_a)
        


        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator( (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a)/3.0 )
        
        # For reconstruction
        self.reconstruct()
        
        # 1-LAYER TRANSFORMER FUSION
        print(self.utt_private_t.shape)
        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v,  self.utt_shared_a), dim=0)
        
        y=list(h.size())
        print("hiiiiiiiii",y)
        
        h=h.contiguous().view(y[0], -1,y[3])
        
        
     
        
        
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]),dim=1)
        print("h shape",h.shape)
        x=list(h.size())
        
        h=h.contiguous().view(-1,y[2], x[1])



        
        

        #x=list(h.size())
        #h = self.transformer_encoder(h.view(x[1],x[2],-1))
        #print("########",h.shape)
        #h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]))
       # print("########",h.shape)
        
        
        
        
        

        
        #print("i am hhhhhhh",h.shape)

        #o = self.fusion(h)
        
        return h
    
    def reconstruct(self,):

        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)
        

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)


    def shared_private(self, utterance_t, utterance_v, utterance_a):
        
        # Projecting to same sized space
       
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)

        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)
        
       


        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)


    def forward(self, sentences, video,  lengths):
        batch_size = 32

        o = self.alignment(sentences, video,  lengths)
        return o

   
# define the autoencoder model
class SparseAutoencoder(nn.Module):
    def __init__(self):
        super(SparseAutoencoder, self).__init__()
 
        # encoder
        self.enc1 = nn.Linear(in_features=784, out_features=256)
        self.enc2 = nn.Linear(in_features=256, out_features=128)
        self.enc3 = nn.Linear(in_features=128, out_features=64)
        self.enc4 = nn.Linear(in_features=64, out_features=32)
        self.enc5 = nn.Linear(in_features=32, out_features=16)
 
        # decoder 
        self.dec1 = nn.Linear(in_features=16, out_features=32)
        self.dec2 = nn.Linear(in_features=32, out_features=64)
        self.dec3 = nn.Linear(in_features=64, out_features=128)
        self.dec4 = nn.Linear(in_features=128, out_features=256)
        self.dec5 = nn.Linear(in_features=256, out_features=784)
 
    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))
 
        # decoding
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
        return x
model = SparseAutoencoder().to(device)