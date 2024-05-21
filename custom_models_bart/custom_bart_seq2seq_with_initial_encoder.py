from transformers import BartConfig, BartModel
from .utils import load_model
import torch.nn as nn
import torch
import math

# Transformer Encoder
class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model: int=768):
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int=100, dropout: float=0.1) -> None:
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)
    
class CustomEncoder(nn.Module):
    def __init__(
        self,
        d_model: int=512,
        num_layers: int=6,
        nhead: int=8,
        dim_feedforward: int=2048,
        dropout: float=0.1,
        activation: str="relu",
        batch_first: bool=True,
    ):
        super(CustomEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=batch_first,
            dim_feedforward=dim_feedforward,
            activation=activation
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
    def forward(self, src, src_key_padding_mask):
        # input_data (batch, seq_len, d_model)
        return self.encoder(
            src=src,
            src_key_padding_mask=src_key_padding_mask,
        )
    
# Custom BartModelSeq2seq
class CustomBartSeq2seq(nn.Module):
    def __init__(
        self,
        config_bart: BartConfig,
        config_encoder: dict,
        checkpoint,
    ):
        super(CustomBartSeq2seq, self).__init__()
        # Input embedding
        self.input_emb = InputEmbeddings(
            vocab_size=config_encoder.vocab_size,
            d_model=config_encoder.d_model,
        )
        # Positional Encoding
        self.pos_emb = PositionalEncoding(
            d_model=config_encoder.d_model,
            dropout=config_encoder.dropout
        )
        # Custom Encoder
        self.encoder = CustomEncoder(
            d_model=config_encoder.d_model,
            num_layers=config_encoder.num_layers,
            nhead=config_encoder.nhead,
            dim_feedforward=config_encoder.encoder_ffn_dim,
            dropout=config_encoder.dropout,
        )
        # Bart Pretrained model
        self.bart_model = BartModel(config_bart)
        self.bart_model = load_model(
            model=self.bart_model,
            checkpoint=checkpoint
        )
        # Linear Classification
        self.out = nn.Linear(config_bart.d_model, config_bart.vocab_size)
        
        # Initialize weights xavier
        self.input_emb.apply(self.initialize_weights)
        self.out.apply(self.initialize_weights)
        self.encoder.apply(self.initialize_weights)
        
    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
    ):
        embed_out = self.input_emb(input_ids)
        embed_out = self.pos_emb(embed_out)
        inputs_embeds = self.encoder(
            src=embed_out,
            src_key_padding_mask=(attention_mask == 0).type(torch.bool)
        )
        outputs = self.bart_model(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
        last_hidden_state = outputs.last_hidden_state
        logits = self.out(last_hidden_state)
        return logits
    
    def initialize_weights(self, layer):
        if isinstance(layer, (nn.Linear, nn.Embedding, nn.MultiheadAttention)):
            nn.init.normal_(layer.weight, mean=0, std=self.config.init_std)
        elif isinstance(layer, nn.LayerNorm):
            layer.weight.data.fill_(1.0)
        else:
            for m in layer.modules():
                for param in m.parameters():
                    if param.dim() > 1:
                        nn.init.normal_(param, mean=0, std=self.config.init_std)
                        
    def get_encoder_out(
        self,
        input_ids,
        attention_mask
    ):
        embed_out = self.input_emb(input_ids)
        embed_out = self.pos_emb(embed_out)
        inputs_embeds = self.encoder(
            src=embed_out,
            src_key_padding_mask=(attention_mask == 0).type(torch.bool)
        )
        return self.bart_model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
    
    def get_decoder_out(
        self,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask
    ):
        outputs = self.bart_model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )
        return outputs