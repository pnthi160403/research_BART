from tokenizers import ByteLevelBPETokenizer
from transformers import  BartModel, BartConfig
import torch
import torch.nn as nn
from .utils import get_weights_file_path, first_train_bart_seq2seq, second_train_bart_seq2seq
import json
import math

BART = "bart"
BART_SEQ2SEQ = "bart_seq2seq"
STEP_TRAIN_BART_SEQ2SEQ = {
    'FIRST': first_train_bart_seq2seq,
    'SECOND': second_train_bart_seq2seq,
}

# get model config
def get_bart_config(config: dict, tokenizer_src, tokenizer_tgt):
    # BART config
    bart_config = BartConfig(
        d_model=config["d_model"],
        encoder_layers=config["encoder_layers"],
        decoder_layers=config["decoder_layers"],
        encoder_attention_heads=config["encoder_attention_heads"],
        decoder_attention_heads=config["decoder_attention_heads"],
        decoder_ffn_dim=config["decoder_ffn_dim"],
        encoder_ffn_dim=config["encoder_ffn_dim"],
        activation_function=config["activation_function"],
        dropout=config["dropout"],
        attention_dropout=config["attention_dropout"],
        activation_dropout=config["activation_dropout"],
        classifier_dropout=config["classifier_dropout"],
        max_position_embeddings=config["max_position_embeddings"],
        init_std=config["init_std"],
        encoder_layerdrop=config["encoder_layerdrop"],
        decoder_layerdrop=config["decoder_layerdrop"],
        scale_embedding=config["scale_embedding"],
        eos_token_id=tokenizer_src.token_to_id("</s>"),
        forced_bos_token_id=tokenizer_src.token_to_id("<s>"),
        forced_eos_token_id=tokenizer_src.token_to_id("</s>"),
        pad_token_id=tokenizer_src.token_to_id("<pad>"),
        num_beams=config["num_beams"],
        vocab_size=tokenizer_tgt.get_vocab_size()
    )

    if not bart_config:
        ValueError("BART config not found")

    return bart_config

# get encoder config
def get_encoder_config(config: dict, tokenizer_src, tokenizer_tgt):
    class ConfigModel():
        pass

    config_encoder = ConfigModel()
    config_encoder.d_model = config["d_model"]
    config_encoder.nhead = config["encoder_attention_heads"]
    config_encoder.vocab_size = tokenizer_src.get_vocab_size()
    config_encoder.dropout = config["dropout"]
    config_encoder.num_layers = config["encoder_layers"]
    config_encoder.encoder_ffn_dim = config["encoder_ffn_dim"]

    return config_encoder

# get bart model
def get_bart_model(config: dict, tokenizer_src, tokenizer_tgt):
    bart_config = get_bart_config(
        config=config,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
    )
    model = CustomBartModel(
        config=bart_config,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt
    )
    if not model:
        ValueError("Model not found")

    print("Check model")
    print(model)
    print("====================================================")

    return model

# get bart model seq2seq
def get_bart_model_seq2seq(config: dict, tokenizer_src, tokenizer_tgt):
    config_bart = get_bart_config(
        config=config,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
    )

    config_encoder = get_encoder_config(
        config=config,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
    )

    checkpoint = config["checkpoint_bart_model"]

    model = CustomBartSeq2seq(
        config_bart=config_bart,
        config_encoder=config_encoder,
        checkpoint=checkpoint,
    )

    if not model:
        ValueError("Model not found")
    
    step_train = config["step_train"]
    if step_train:
        model = STEP_TRAIN_BART_SEQ2SEQ[step_train](
            config=config,
            model=model
        )
    
    print("Check model")
    print(model)
    print("====================================================")
    
    return model
    
GET_MODEL = {
    BART: get_bart_model,
    BART_SEQ2SEQ: get_bart_model_seq2seq,
}

# save model
def save_model(model, epoch, global_step, optimizer, lr_scheduler, config, save_model="model"):
    if save_model == "bart":
        model_filename = get_weights_file_path(config, f"{epoch:02d}", "bart")
    else:
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict()
    }, model_filename)
    
    print(f"Saved model at {model_filename}")

# save config
def save_config(config: dict, epoch: int):
    config_filename = f"{config['model_folder']}/config_{epoch:02d}.json"
    with open(config_filename, "w") as f:
        json.dump(config, f)
    print(f"Saved config at {config_filename}")

# ============================================================================== MODELS ==============================================================================
# custom BartModel
class CustomBartModel(nn.Module):
    def __init__(self, config: BartConfig, tokenizer_src, tokenizer_tgt):
        super().__init__()
        self.config = config
        self.bart_model = BartModel(config)
        self.out = nn.Linear(config.d_model, tokenizer_tgt.get_vocab_size())
        
    def forward(self,**kwargs):
        outputs = self.bart_model(**kwargs)
        last_hidden_state = outputs.last_hidden_state
        logits = self.out(last_hidden_state)
        return logits
    
    def get_encoder_out(
        self,
        input_ids,
        attention_mask
    ):
        return self.bart_model.encoder(
            input_ids,
            attention_mask
        )
    
    def get_decoder_out(
        self,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask
    ):
        return self.bart_model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )
    
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
        batch_first: bool=True,
    ):
        super(CustomEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=batch_first,
            dim_feedforward=dim_feedforward
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
        if torch.cuda.is_available():
            state = torch.load(checkpoint)
        else:
            state = torch.load(checkpoint, map_location=torch.device('cpu'))
        self.bart_model.load_state_dict(state["model_state_dict"])
        # Linear Classification
        self.out = nn.Linear(config_bart.d_model, config_bart.vocab_size)
        
        # Initialize weights xavier
        self._initialize_weights_xavier([
            self.input_emb,
            self.encoder,
            self.out,
        ])
        
    def _initialize_weights_xavier(self, layers):
        for layer in layers:
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        
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