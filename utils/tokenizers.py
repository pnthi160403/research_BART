from tokenizers import Tokenizer, trainers, models, pre_tokenizers, normalizers, decoders

# const variable
BPE_TOKEN = "bpe"
WORDPIECE_TOKEN = "wordpiece"
WORDLEVEL_TOKEN = "wordlevel"

class ApiTokenizerHuggingFace():
    def __init__(
        self,
        iterator,
        vocab_size,
        min_frequency,
        special_tokens,
        type_token,
    ):
        super().__init__()
        self.iterator = self.get_all_sentences(iterator)
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens
        self.type_token = type_token
    
    def get_all_sentences(dataset):
        for item in dataset:
            yield item

    def train(self) -> Tokenizer:
        if self.type_token == BPE_TOKEN:
            model_tokenizer, decoder_tokenizer, normalizers_tokenizer, pre_tokenizer, trainer = self.__get_prepare_bpe()
        elif self.type_token == WORDPIECE_TOKEN:
            model_tokenizer, decoder_tokenizer, normalizers_tokenizer, pre_tokenizer, trainer = self.__get_prepare_wordpiece()
        elif self.type_token == WORDLEVEL_TOKEN:
            model_tokenizer, decoder_tokenizer, normalizers_tokenizer, pre_tokenizer, trainer = self.__get_prepare_wordlevel()
        tokenizer = Tokenizer(model_tokenizer)
        tokenizer.normalizer = normalizers_tokenizer
        tokenizer.pre_tokenizer = pre_tokenizer
        if decoder_tokenizer:
            tokenizer.decoder = decoder_tokenizer
        tokenizer.train_from_iterator(
            iterator=self.iterator,
            trainer=trainer,
        )
        return tokenizer
    
    def __get_prepare_bpe(self):
        # prepare train tokenizer
        model_tokenizer = models.BPE(unk_token="<unk>") # adjust model
        decoder_tokenizer = None # adjust decoder
        normalizers_tokenizer = normalizers.Sequence([normalizers.Lowercase()])
        pre_tokenizer = pre_tokenizers.Whitespace()
        # define trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
        ) # adjust trainer
        return model_tokenizer, decoder_tokenizer, normalizers_tokenizer, pre_tokenizer, trainer
    
    def __get_prepare_wordpiece(self):
        # prepare train tokenizer
        model_tokenizer = models.WordPiece(unk_token="<unk>") # adjust model
        decoder_tokenizer = decoders.WordPiece() # adjust decoder
        # decoder_tokenizer = None # adjust decoder
        normalizers_tokenizer = normalizers.Sequence([normalizers.Lowercase()])
        pre_tokenizer = pre_tokenizers.Whitespace()
        # define trainer
        trainer = trainers.WordPieceTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
        ) # adjust trainer
        return model_tokenizer, decoder_tokenizer, normalizers_tokenizer, pre_tokenizer, trainer
    
    def __get_prepare_wordlevel(self):
        # prepare train tokenizer
        model_tokenizer = models.WordLevel(unk_token="<unk>") # adjust model
        decoder_tokenizer = None # adjust decoder
        normalizers_tokenizer = normalizers.Sequence([normalizers.Lowercase()])
        pre_tokenizer = pre_tokenizers.Whitespace()
        # define trainer
        trainer = trainers.WordLevelTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
        ) # adjust trainer
        return model_tokenizer, decoder_tokenizer, normalizers_tokenizer, pre_tokenizer, trainer
    
def read_tokenizer(
        tokenizer_src_path: str,
        tokenizer_tgt_path: str
):
    tokenizer_src = Tokenizer.from_file(tokenizer_src_path)
    tokenizer_tgt = Tokenizer.from_file(tokenizer_tgt_path)

    if not tokenizer_src or not tokenizer_tgt:
        ValueError("Tokenizer not found")

    return tokenizer_src, tokenizer_tgt

__all__ = ["ApiTokenizerHuggingFace", "read_tokenizer"]