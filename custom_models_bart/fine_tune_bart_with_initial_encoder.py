from transformers import BartConfig, BartModel
from .utils import load_model
import torch.nn as nn
from .custom_bart_with_embedding import CustomBartModelWithEmbedding
    
# Fine-tune BART with initial encoder
class FineTuneBartWithRandomEncoder(nn.Module):
    def __init__(
        self,
        config: BartConfig,
        src_vocab_size,
        tgt_vocab_size,
        vocab_size_encoder_bart=30000,
        checkpoint_custom_bart_with_embedding=None,
        init_type=None,
    ):
        super(FineTuneBartWithRandomEncoder, self).__init__()
        self.config = config

        # vocab size
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.vocab_size_encoder_bart = vocab_size_encoder_bart
        
        # Load checkpoint
        custom_bart_with_embedding = CustomBartModelWithEmbedding(
            config=config,
            src_vocab_size=self.vocab_size_encoder_bart,
            tgt_vocab_size=self.tgt_vocab_size,
            init_type=init_type,
        )
        custom_bart_with_embedding = load_model(
            model=custom_bart_with_embedding,
            checkpoint=checkpoint_custom_bart_with_embedding
        )

        # Src embedding
        self.inputs_embeds = nn.Embedding(
            num_embeddings=self.src_vocab_size,
            embedding_dim=self.config.d_model,
        )

        # Tgt embedding
        self.decoder_inputs_embeds = custom_bart_with_embedding.decoder_inputs_embeds

        # Encoder initialization
        self.random_encoder = BartModel(config).encoder

        # Pretained BART model
        self.bart_model = custom_bart_with_embedding.bart_model

        # Prediction
        self.out = custom_bart_with_embedding.out
        
        # Initialize weights xavier
        modules = [self.inputs_embeds, self.random_encoder]
        self.initialize_weights(
            modules=modules,
            init_type=init_type,
            mean=0,
            std=self.config.init_std
        )
        
    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        label=None,
    ):
        inputs_embeds = self.inputs_embeds(input_ids)
        inputs_embeds = self.random_encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        ).last_hidden_state
        decoder_inputs_embeds = self.decoder_inputs_embeds(decoder_input_ids)
        outputs = self.bart_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_attention_mask=decoder_attention_mask,
        )   
        last_hidden_state = outputs.last_hidden_state
        logits = self.out(last_hidden_state)

        if label is not None:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.01)
            loss = loss_fn(logits.view(-1, self.tgt_vocab_size), label.view(-1))
            return logits, loss
                
        return logits
    
    def initialize_weights(self, modules, init_type="normal", mean=0, std=0.02):
        for module in modules:
            for param in module.parameters():
                if param.dim() > 1:
                    if init_type == "normal":
                        nn.init.normal_(param, mean=mean, std=std)
                    elif init_type == "xavier":
                        nn.init.xavier_normal_(param)
                    else:
                        continue
                    
    def get_encoder_out(
        self,
        input_ids,
        attention_mask
    ):
        inputs_embeds = self.inputs_embeds(input_ids)
        inputs_embeds = self.random_encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        ).last_hidden_state

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