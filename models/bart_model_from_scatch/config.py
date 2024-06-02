class BartConfig:
    def __init__(
        self,
        d_model=768,
        encoder_layers=12,
        decoder_layers=12,
        encoder_attention_heads=12,
        decoder_attention_heads=12,
        decoder_ffn_dim=3072,
        encoder_ffn_dim=3072,
        activation_function="gelu",
        dropout=0.01,
        attention_dropout=0.01,
        activation_dropout=0.01,
        classifier_dropout=0.01,
        max_position_embeddings=1024,
        init_std=0.02,
        scale_embedding=False,
        vocab_size=50265,
    ):
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_ffn_dim = encoder_ffn_dim
        self.activation_function = activation_function
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.classifier_dropout = classifier_dropout
        self.max_position_embeddings = max_position_embeddings
        self.init_std = init_std
        self.scale_embedding = scale_embedding
        self.vocab_size = vocab_size

__all__ = ["BartConfig"]