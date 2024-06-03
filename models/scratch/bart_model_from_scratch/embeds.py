import torch
import torch.nn as nn

class BartEmbeds(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        max_position_embeddings: int=1024,
        shared: bool = False,
        embed_scale: float=1.0,
    ):
        super().__init__()

        self.embed_scale = embed_scale
        self.embed_tokens = nn.Embedding(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.embed_positions = nn.Embedding(
            max_position_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.register_buffer(
            "pos_ids",
            torch.arange(0, max_position_embeddings)
        )
        if shared:
            self.embed_positions.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor):
        bsz, seq_len = input_ids.size()
        pos_ids = self.pos_ids[:seq_len]
        embeds = self.embed_tokens(input_ids) * self.embed_scale + self.embed_positions(pos_ids)
        return embeds
    
__all__ = ["BartEmbeds"]