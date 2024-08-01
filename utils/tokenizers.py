from tokenizers import Tokenizer
    
def read_tokenizer(
        tokenizer_src_path: str,
        tokenizer_tgt_path: str,
        share_vocab: bool=False,
):
    tokenizer_src = Tokenizer.from_file(tokenizer_src_path)
    tokenizer_tgt = Tokenizer.from_file(tokenizer_tgt_path)

    if not tokenizer_src or not tokenizer_tgt:
        ValueError("Tokenizer not found")
    
    if share_vocab:
        tokenizer_tgt = tokenizer_src

    return tokenizer_src, tokenizer_tgt

__all__ = ["read_tokenizer"]