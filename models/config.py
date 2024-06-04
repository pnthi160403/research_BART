from transformers import BartConfig
class BartSeq2seqConfig(BartConfig):
    def __init__(
        self,
        config: BartConfig,
        src_vocab_size: int,
        tgt_vocab_size: int,
        pad_idx: int,
        share_tgt_emb_and_out: bool,
        init_type: str,
    ):
        super