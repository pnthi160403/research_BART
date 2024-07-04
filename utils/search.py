import torch
import torch.nn as nn

class Search(nn.Module):
    def __init__(
        self,
        special_tokens: dict,
        vocab_size: int,
    ):
        super().__init__()
        for k, v in special_tokens.items():
            setattr(self, k, v)
        self.vocab_size = vocab_size

    def step(
        self,
        step: int,
        lprobs: torch.Tensor,
        scores: torch.Tensor,
    ):
        raise NotImplementedError
    
class BeamSearch(Search):
    def __init__(
        self,
        special_tokens: dict,
        vocab_size: int,
        candidate_multiple: int=2,
        **kwargs,
    ):
        super().__init__(
            special_tokens=special_tokens,
            vocab_size=vocab_size,
        )
        self.candidate_multiple = candidate_multiple    

    def step(
        self,
        step: int,
        lprobs: torch.Tensor,
        scores: torch.Tensor,
        mask: torch.Tensor=None,
        **kwargs,
    ):
        '''
        Args:
            step: int
            lprobs: (batch_size, input_beam_size, vocab_size)
            scores: (batch_size, input_beam_size, step)
            mask: (batch_size, input_beam_size), mask[:,i] == 0 means beam i is finished
        Returns:
            scores_pred: (batch_size, candidate_multiple * input_beam_size)
            indices_pred: (batch_size, candidate_multiple * input_beam_size)
            beams_pred: (batch_size, candidate_multiple * input_beam_size)
        '''
        # lprobs: (batch_size, input_beam_size, vocab_size)
        # scores: (batch_size, input_beam_size, step)

        bsz, beam_size, vocab_size = lprobs.size()

        # expand mask
        if mask is not None:
            mask = mask.unsqueeze(-1).repeat(1, 1, vocab_size)
            lprobs = lprobs.masked_fill_(
                mask=mask == 0,
                value=0.0,
            )

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, 0:1, :].contiguous()
        else:
            # make probs contain cumulative scores for each hypothesis
            assert scores is not None
            # print(f"{ scores[:, :, step - 1] = }")
            lprobs = lprobs + scores[:, :, step - 1].unsqueeze(-1)

        # print(f"{ lprobs = }")
        top_prediction = torch.topk(
            lprobs.view(bsz, -1),
            k=self.candidate_multiple * beam_size
        )
        scores_buf = top_prediction[0]
        indices_buf = top_prediction[1]
        # Project back into relative indices and beams
        beams_buf = torch.div(indices_buf, vocab_size, rounding_mode="trunc")
        indices_buf = indices_buf.fmod(vocab_size)

        # At this point, beams_buf and indices_buf are single-dim and contain relative indices
        return scores_buf, indices_buf, beams_buf
    
class DiverseBeamSearch(Search):
    def __init__(
        self,
        special_tokens: dict,
        vocab_size: int,
        num_groups: int,
        diversity_strength: float,
        diversity_discount: float=0.5,
        candidate_multiple: int=1,
    ):
        super().__init__(
            special_tokens=special_tokens,
            vocab_size=vocab_size,
        )
        self.num_groups = num_groups
        self.diversity_strength = -diversity_strength
        self.beam = BeamSearch(
            special_tokens=special_tokens,
            vocab_size=vocab_size,
            candidate_multiple=candidate_multiple,
        )
        self.diversity_discount = diversity_discount
        self.candidate_multiple = candidate_multiple

        # Float tensor to keep track of overlap between groups.
        # Each token shared at the same step between two groups is counted as one.
        # Then token counts are discounted by `diversity_discount` for every next timestep.
        # Once initialized, dimension is batch_size * num_groups * num_groups.
        self.group_overlap = torch.empty(0)

    def step(
        self,
        step: int,
        lprobs,
        scores: torch.Tensor=None,
        original_batch_idxs: torch.Tensor=None,
        mask: torch.Tensor=None,
    ):
        # lprobs: (batch_size, input_beam_size, vocab_size)
        # scores: (batch_size, input_beam_size, step)
        # original_batch_idxs: (batch_size,)
        # mask: (batch_size, input_beam_size), mask[:,i] == 0 means beam i is finished
        bsz, beam_size, vocab_size = lprobs.size()
        if beam_size % self.num_groups != 0:
            raise ValueError(
                "DiverseBeamSearch requires --beam to be divisible by the number of groups"
            )
        
        # diversity_buf (batch_size, vocab_size)
        diversity_buf = torch.zeros(lprobs[:, 0, :].size()).to(lprobs)

        scores_G, beams_G = [], []

        # pre-allocating tensor for indices for all groups
        indices_G_stacked = torch.empty(
            bsz,
            int(beam_size / self.num_groups) * self.candidate_multiple,
            self.num_groups,
            dtype=torch.long,
            device=lprobs.device,
        )

        for g in range(self.num_groups):
            # lpobbs_g: (batch_size, mini_beam, vocab_size)
            lprobs_g = lprobs[:, g :: self.num_groups, :]
            # scores_g: (batch_size, mini_beam, step)
            scores_g = scores[:, g :: self.num_groups, :] if step > 0 else None
            if mask is not None:
                # mask_g: (batch_size, mini_beam)
                mask_g = mask[:, g :: self.num_groups]

            diversity_buf.zero_()
            # apply diversity penalty
            if g > 0:
                indices_ = indices_G_stacked[:, :, :g]
                if step > 0:
                    penalty_val = 1 + self.group_overlap[original_batch_idxs, g, :g]
                    penalty_val = penalty_val.unsqueeze(1)
                else:
                    penalty_val = torch.ones(bsz, 1, 1)
                diversity_buf.scatter_add_(
                    dim=1,
                    index=indices_.reshape(bsz, -1),
                    src=penalty_val.expand(indices_.size())
                    .reshape(bsz, -1)
                    .to(diversity_buf),
                )
                lprobs_g = torch.add(
                    lprobs_g,
                    other=diversity_buf.unsqueeze(1),
                    alpha=self.diversity_strength,
                )
            else:
                lprobs_g = lprobs_g.contiguous()

            scores_buf, indices_buf, beams_buf = self.beam.step(
                step=step,
                lprobs=lprobs_g,
                scores=scores_g,
                mask=mask_g,
            )
            beams_buf.mul_(self.num_groups).add_(g)
            scores_G.append(scores_buf.clone())
            beams_G.append(beams_buf.clone())
            indices_G_stacked[:, :, g] = indices_buf

        # interleave results from different groups
        scores_buf = torch.stack(scores_G, dim=2).view(bsz, -1)
        indices_buf = indices_G_stacked.view(bsz, -1)
        beams_buf = torch.stack(beams_G, dim=2).view(bsz, -1)
        # find num of overlapped tokens for each group pair
        # then discount it for next timestamp
        # overlap_pair_gr (batch_size, mini_beam_size, num_groups, num_groups)
        overlap = (indices_G_stacked.unsqueeze(2) == indices_G_stacked.unsqueeze(3)).int()
        if mask is not None:
            # (batch_size, input_beam_size) -> (batch_size, mini_beam_size, num_groups)
            mask = mask.view(bsz, -1, self.num_groups)
            # overlap_mask (batch_size, mini_beam_size, num_groups, num_groups)
            overlap_mask = mask.unsqueeze(2) & mask.unsqueeze(3)
            # print(f"{ overlap.shape = }")
            # print(f"{ overlap_mask.shape = }")
            # overlap = overlap * overlap_mask
            overlap = overlap.masked_fill_(
                mask=overlap_mask == 0,
                value=0,
            )
        overlap =  torch.sum(overlap, dim=1)
    
        # self.group_overlap: (batch_size, num_groups, num_groups)
        if step == 0:
            self.group_overlap = overlap
        else:
            self.group_overlap[original_batch_idxs] = self.group_overlap[original_batch_idxs] + overlap
        self.group_overlap = self.group_overlap * self.diversity_discount

        return scores_buf, indices_buf, beams_buf

class SearchItem():
    def __init__(
        self,
        eos_token_id: int,
        pad_token_id: int,
        sos_token_id: int,
        tokenizer_tgt,
        tokenizer_src,
        max_len: int,
        encoder_output: torch.Tensor,
        device: str,
        encoder_output_mask: torch.Tensor=None,
    ):
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.sos_token_id = sos_token_id
        self.tokenizer_tgt = tokenizer_tgt
        self.tokenizer_src = tokenizer_src
        self.device = device
        self.max_len = max_len

        self.encoder_output = encoder_output
        self.src_attention_mask = encoder_output_mask
        self.tgt = torch.tensor([self.sos_token_id]).to(device)
        self.tgt_attention_mask = (self.tgt != pad_token_id).type(torch.int64).to(device)
        self.last_token = self.sos_token_id
        self.num_steps = 0
        self.scores = None
        self.past_key_values = None
        self.past_attn_scores = None

    def copy(self):
        new_item = SearchItem(
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            sos_token_id=self.sos_token_id,
            tokenizer_tgt=self.tokenizer_tgt,
            tokenizer_src=self.tokenizer_src,
            max_len=self.max_len,
            encoder_output=self.encoder_output,
            device=self.device,
            encoder_output_mask=self.src_attention_mask,
        )
        new_item.tgt = self.tgt
        new_item.tgt_attention_mask = self.tgt_attention_mask
        new_item.last_token = self.last_token
        new_item.num_steps = self.num_steps
        new_item.scores = self.scores
        new_item.past_key_values = self.past_key_values
        new_item.past_attn_scores = self.past_attn_scores
        return new_item

    def stop_search(self):
        return len(self.tgt) >= self.max_len or self.last_token == self.eos_token_id
    
    def step(
        self,
        score: float,
        indice: int,
        past_key_values: torch.Tensor=None,
        past_attn_scores: torch.Tensor=None,
    ):
        self.num_steps += 1
        if self.scores is None:
            self.scores = torch.tensor([score]).to(self.device)
        else:
            self.scores = torch.cat([
                self.scores,
                torch.tensor([score]).to(self.device)
            ], dim=-1)
        if not self.stop_search():
            self.last_token = indice
            self.tgt = torch.cat([
                self.tgt,
                torch.tensor([indice]).to(self.device)
            ], dim=-1)
        self.tgt_attention_mask = (self.tgt != self.pad_token_id).type(torch.int64).to(self.device)
        self.past_key_values = past_key_values
        self.past_attn_scores = past_attn_scores

BEAM_SEARCH = "beam_search"
DIVERSE_BEAM_SEARCH = "diverse_beam_search"

TYPE_SEARCH = {
    BEAM_SEARCH: BeamSearch,
    DIVERSE_BEAM_SEARCH: DiverseBeamSearch,
}

__all__ = [
    "Search",
    "BeamSearch",
    "DiverseBeamSearch",
    "SearchItem",
    "TYPE_SEARCH",
    "BEAM_SEARCH",
    "DIVERSE_BEAM_SEARCH",
]