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
        scores: torch.Tensor=None,
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
        # mask: (batch_size, input_beam_size, n_gram)

        bsz, beam_size, vocab_size = lprobs.size()

        # expand mask
        if mask is not None:
            _, _, n_gram = mask.size()
            mask = mask.unsqueeze(-2).repeat(1, 1, vocab_size, 1)
            # mask (batch_size, input_beam_size, vocab_size)
            mask = torch.sum(mask, dim=-1)
            # print(f"{ mask.size() = }")
            # print(f"{ mask = }")
            mask = (mask == n_gram).int()
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

HAMMING_CUMULATIVE_TYPE_DIVERSITY = "Hamming_Cumulative"
N_GRAM_TYPE_DIVERSITY = "N_Gram"

class DiverseBeamSearch(Search):
    def __init__(
        self,
        special_tokens: dict,
        vocab_size: int,
        num_groups: int,
        diversity_strength: float,
        diversity_discount: float=0.5,
        candidate_multiple: int=1,
        n_gram: int=1,
        device: str="cpu",
        type_diversity_function: str=HAMMING_CUMULATIVE_TYPE_DIVERSITY,
        **kwargs,
    ):
        super().__init__(
            special_tokens=special_tokens,
            vocab_size=vocab_size,
        )
        TYPE_DIVERSITY_FUNCTION = {
            HAMMING_CUMULATIVE_TYPE_DIVERSITY: self.calc_overlap_type_hamming_cumulative,
            N_GRAM_TYPE_DIVERSITY: self.calc_overlap_type_n_gram,
        }
        self.num_groups = num_groups
        self.diversity_strength = -diversity_strength
        self.beam = BeamSearch(
            special_tokens=special_tokens,
            vocab_size=vocab_size,
            candidate_multiple=candidate_multiple,
        )
        self.diversity_discount = diversity_discount
        self.candidate_multiple = candidate_multiple
        self.n_gram = n_gram
        self.device = device
        self.diversity_fn = TYPE_DIVERSITY_FUNCTION[type_diversity_function]

        # Float tensor to keep track of overlap between groups.
        # Each token shared at the same step between two groups is counted as one.
        # Then token counts are discounted by `diversity_discount` for every next timestep.
        # Once initialized, dimension is batch_size * num_groups * num_groups.
        self.group_overlap = None

    def calc_present_n_gram_indices(
        self,
        last_n_gram_indices: torch.Tensor=None,
        indices: torch.Tensor=None,
        **kwargs,
    ):
        if last_n_gram_indices is None or indices is None:
            return None
        # last_n_gram_indices: (batch_size, input_beam_size, [0, n_gram-1])
        # indices: (batch_size, input_beam_size)
        # print(f"{ last_n_gram_indices.size() = }")
        # print(f"{ indices.size() = }")
        # present_indices (batch_size, input_beam_size, n_gram)
        present_indices = torch.cat([
            last_n_gram_indices,
            indices.unsqueeze(-1),
        ], dim=-1).to(self.device)
        # print(f"{ present_indices.size() = }")
        return present_indices
    
    def calc_overlap_type_n_gram(
        self,
        last_n_gram_indices: torch.Tensor=None,
        indices_n_gram: torch.Tensor=None,
        mask_indices_n_gram: torch.Tensor=None,
        indices: torch.Tensor=None,
        **kwargs,
    ):
        # last_n_gram_indices (batch_size, input_beam_size, n_gram)
        # indices_n_gram (batch_size, input_beam_size, step - n_gram + 1, n_gram)
        # mask_indices_n_gram (batch_size, input_beam_size, step - n_gram + 1, n_gram)

        # present_indices (batch_size, input_beam_size, n_gram)
        present_n_gram_indices = self.calc_present_n_gram_indices(
            last_n_gram_indices=last_n_gram_indices,
            indices=indices,
        )
        if present_n_gram_indices is None or indices_n_gram is None:
            return None
        bsz, input_beam_size, n_gram = present_n_gram_indices.size()

        # (batch_size, input_beam_size, n_gram) -> (batch_size, input_beam_size, step - n_gram + 1, n_gram)
        # print(f"{ present_n_gram_indices.shape = }")
        present_n_gram_indices = present_n_gram_indices.unsqueeze(-2).repeat(1, 1, indices_n_gram.size(-2), 1)
        # print(f"{ present_n_gram_indices.shape = }")
        # (batch_size, mini_beam_size, num_groups, n_gram) -> (batch_size, mini_beam_size, num_groups, step - n_gram + 1, n_gram)
        present_n_gram_indices = present_n_gram_indices.view(bsz, -1, self.num_groups, indices_n_gram.size(-2), n_gram).contiguous()
        # print(f"{ present_n_gram_indices.shape = }")
        # print(f"{ indices_n_gram.shape = }")
        # (batch_size, input_beam_size, step - n_gram + 1, n_gram) -> (batch_size, mini_beam_size, num_groups, step - n_gram + 1, n_gram)
        indices_n_gram = indices_n_gram.view(bsz, -1, self.num_groups, indices_n_gram.size(-2), n_gram).contiguous()
        # print(f"{ indices_n_gram.shape = }")
        # print(f"{ present_n_gram_indices.unsqueeze(2).size() = }")
        # print(f"{ indices_n_gram.unsqueeze(3).size() = }")

        # overlap (batch_size, mini_beam_size, num_groups, num_groups, step - n_gram + 1, n_gram)
        overlap = (present_n_gram_indices.unsqueeze(2) == indices_n_gram.unsqueeze(3)).int()
        # print(f"{ overlap.size() = }")

        if mask_indices_n_gram is not None:
            # mask_indices_n_gram (batch_size, mini_beam_size, num_groups, step - n_gram + 1, n_gram)
            # print(f"{ mask_indices_n_gram.shape = }")
            # print(f"{ mask_indices_n_gram.size(-1) = }")
            # print(f"{ mask_indices_n_gram.view(bsz, -1, self.num_groups, mask_indices_n_gram.size(-1), n_gram).shape = }")
            mask_indices_n_gram = mask_indices_n_gram.view(bsz, -1, self.num_groups, mask_indices_n_gram.size(-2), n_gram).contiguous()
            # overlap_mask (batch_size, mini_beam_size, num_groups, num_groups, step - n_gram + 1, n_gram)
            overlap_mask = mask_indices_n_gram.unsqueeze(2) & mask_indices_n_gram.unsqueeze(3)
            # print(f"{ overlap_mask.size() = }")
            # overlap = overlap * overlap_mask
            overlap = overlap.masked_fill_(
                mask=overlap_mask == 0,
                value=0,
            )
            # print(f"{ overlap.size() = }")
        # overlap (batch_size, mini_beam_size, num_groups, num_groups, step - n_gram + 1)
        overlap = torch.sum(overlap, dim=-1)
        # print(f"{ overlap.size() = }")
        # overlap (batch_size, mini_beam_size, num_groups, num_groups, step - n_gram + 1)
        overlap = (overlap == n_gram).int()
        # print(f"{ overlap.size() = }")
        # overlap (batch_size, mini_beam_size, num_groups, num_groups)
        overlap = torch.sum(overlap, dim=-1)
        # print(f"{ overlap.size() = }")
        # overlap (batch_size, num_groups, num_groups)
        overlap = torch.sum(overlap, dim=1)
        # print(f"{ overlap.size() = }")
        return overlap

    def calc_overlap_type_hamming_cumulative(
        self,
        last_n_gram_indices: torch.Tensor=None,
        indices: torch.Tensor=None,
        mask: torch.Tensor=None,
        **kwargs,
    ):
        # last_n_gram_indices (batch_size, input_beam_size, n_gram)
        # indices (batch_size, input_beam_size)
        # mask (batch_size, input_beam_size, n_gram)
        # if last_n_gram_indices is not None:
        #     print(f"{ last_n_gram_indices.shape = }")

        # present_indices (batch_size, input_beam_size, n_gram)
        present_n_gram_indices = self.calc_present_n_gram_indices(
            last_n_gram_indices=last_n_gram_indices,
            indices=indices,
        )
        if present_n_gram_indices is None:
            return None
        # mask (batch_size, input_beam_size, n_gram)
        bsz, input_beam_size, n_gram = present_n_gram_indices.size()

        # (batch_size, input_beam_size, n_gram) -> (batch_size, mini_beam_size, num_groups, n_gram)
        present_n_gram_indices = present_n_gram_indices.view(bsz, -1, self.num_groups, n_gram).contiguous()
        # print(f"{ present_n_gram_indices.size() = }")
        # print(f"{ present_n_gram_indices.unsqueeze(2).size() = }")
        # print(f"{ present_n_gram_indices.unsqueeze(3).size() = }")

        # overlap (batch_size, mini_beam_size, num_groups, num_groups)
        overlap = (present_n_gram_indices.unsqueeze(2) == present_n_gram_indices.unsqueeze(3)).int()
        # print(f"{ overlap.size() = }")
        if mask is not None:
            # (batch_size, input_beam_size, n_gram) -> (batch_size, mini_beam_size, num_groups, n_gram)
            # print(f"{ mask.size() = }")
            mask = mask.view(bsz, -1, self.num_groups, n_gram).contiguous()
            # overlap_mask (batch_size, mini_beam_size, num_groups, num_groups, n_gram)
            overlap_mask = mask.unsqueeze(2) & mask.unsqueeze(3)
            # print(f"{ overlap_mask.size() = }")
            # overlap = overlap * overlap_mask
            overlap = overlap.masked_fill_(
                mask=overlap_mask == 0,
                value=0,
            )
        # overlap (batch_size, mini_beam_size, num_groups, num_groups)
        overlap = torch.sum(overlap, dim=-1)
        # print(f"{ overlap.size() = }")
        overlap = (overlap == n_gram).int()
        # overlap (batch_size, num_groups, num_groups)
        overlap =  torch.sum(overlap, dim=1)
        return overlap

    def step(
        self,
        step: int,
        lprobs: torch.Tensor,
        scores: torch.Tensor=None,
        original_batch_idxs: torch.Tensor=None,
        last_n_gram_indices: torch.Tensor=None,
        mask_last_n_gram_indices: torch.Tensor=None,
        indices_n_gram: torch.Tensor=None,
        mask_indices_n_gram: torch.Tensor=None,
        **kwargs,
    ):
        # lprobs: (batch_size, input_beam_size, vocab_size)
        # scores: (batch_size, input_beam_size, step)
        # original_batch_idxs: (batch_size,)
        # mask_last_n_gram_indices: (batch_size, input_beam_size, n_gram)
        # last_n_gram_indices (batch_size, input_beam_size, [0, n_gram-1])
        # indices_n_gram (batch_size, input_beam_size, step - n_gram + 1, n_gram)
        # mask_indices_n_gram (batch_size, input_beam_size, step - n_gram + 1, n_gram)
        bsz, beam_size, vocab_size = lprobs.size()
        if beam_size % self.num_groups != 0:
            raise ValueError(
                "DiverseBeamSearch requires --beam to be divisible by the number of groups"
            )
        
        # diversity_buf (batch_size, input_beam_size, vocab_size)
        diversity_buf = torch.zeros(lprobs[:, 0, :].size()).to(lprobs)

        scores_G, beams_G = [], []

        # pre-allocating tensor for indices for all groups
        # (batch_size, mini_beam_size, num_groups)
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
            # mask_g: (batch_size, mini_beam)
            mask_g = mask_last_n_gram_indices[:, g :: self.num_groups, :] if mask_last_n_gram_indices is not None else None
            # print(f"{ lprobs_g = }")
            # print(f"{ scores_g = }")
            # print(f"{ mask_g = }")

            diversity_buf.zero_()
            # apply diversity penalty
            if g > 0:
                indices_ = indices_G_stacked[:, :, :g]
                if self.group_overlap is not None:
                    penalty_val = 1 + self.group_overlap[original_batch_idxs, g, :g]
                    penalty_val = penalty_val.unsqueeze(1)
                else:
                    penalty_val = torch.ones(bsz, 1, 1)
                # print(f"{ penalty_val.shape = }")
                diversity_buf.scatter_add_(
                    index=indices_.reshape(bsz, -1),
                    src=penalty_val.expand(indices_.size())
                    .reshape(bsz, -1)
                    .to(diversity_buf),
                    dim=1,
                )
                lprobs_g = torch.add(
                    lprobs_g,
                    other=diversity_buf.unsqueeze(1),
                    alpha=self.diversity_strength,
                )
            else:
                lprobs_g = lprobs_g.contiguous()
            # print(f"{ lprobs_g.shape = }")

            scores_buf, indices_buf, beams_buf = self.beam.step(
                step=step,
                lprobs=lprobs_g,
                scores=scores_g,
                mask=mask_g,
            )
            # print(f"{ scores_buf = }")
            # print(f"{ indices_buf = }")
            # print(f"{ beams_buf = }")
            beams_buf = beams_buf * self.num_groups + g
            # beams_buf.mul_(self.num_groups).add_(g)
            scores_G.append(scores_buf.clone())
            beams_G.append(beams_buf.clone())
            indices_G_stacked[:, :, g] = indices_buf
            # print("hi")

        # interleave results from different groups
        scores_buf = torch.stack(scores_G, dim=2).view(bsz, -1)
        indices_buf = indices_G_stacked.view(bsz, -1)
        beams_buf = torch.stack(beams_G, dim=2).view(bsz, -1)
        # print(f"{ scores_buf.shape = }")
        # print(f"{ indices_buf.shape = }")
        # print(f"{ beams_buf.shape = }")

        # example
        # [a, b, c] <=> last_n_gram_indices
        # [d] <=> indices_buf
        # [a, b, c, d] <=> present_n_gram_indices
        # convert [a, b, c, d] -> [abcd] and assign [abcd] to present_n_gram_indices
        # find num of overlapped tokens for each group pair
        overlap = self.diversity_fn(
            last_n_gram_indices=last_n_gram_indices,
            indices_n_gram=indices_n_gram,
            indices=indices_buf,
            mask_indices_n_gram=mask_indices_n_gram,
            mask=mask_last_n_gram_indices,
        )

        # then discount it for next timestamp
        # self.group_overlap: (batch_size, num_groups, num_groups)
        if overlap is not None:
            if self.group_overlap is None:
                self.group_overlap = overlap
            else:
                self.group_overlap[original_batch_idxs] = self.group_overlap[original_batch_idxs] + overlap
            self.group_overlap = self.group_overlap * self.diversity_discount

        return scores_buf, indices_buf, beams_buf

class SearchNode():
    def __init__(
        self,
        eos_token_id: int,
        pad_token_id: int,
        sos_token_id: int,
        tokenizer_tgt,
        tokenizer_src,
        device: str,
        encoder_output: torch.Tensor,
        encoder_output_mask: torch.Tensor=None,
        max_len: int=200,
        n_gram: int=1,
    ):
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.sos_token_id = sos_token_id
        self.tokenizer_tgt = tokenizer_tgt
        self.tokenizer_src = tokenizer_src
        self.device = device
        self.max_len = max_len
        self.n_gram = n_gram

        self.encoder_output = encoder_output
        self.src_attention_mask = encoder_output_mask
        self.tgt = torch.tensor([self.sos_token_id]).to(device)
        self.tgt_attention_mask = (self.tgt != pad_token_id).type(torch.int64).to(device)
        self.last_token = self.sos_token_id
        self.num_steps = 0
        self.scores = None
        self.past_key_values = None
        self.past_attn_scores = None
        self.last_n_gram_indices = None
        self.mask_indices_n_gram = None
        self.indices_n_gram = None
        self.mask_indices_n_gram = None
        if self.n_gram > 1:
            self.last_n_gram_indices = torch.tensor([self.sos_token_id]).to(self.device)
            self.mask_last_n_gram_indices = (self.last_n_gram_indices != self.pad_token_id).type(torch.int64).to(self.device)
        if self.n_gram == 1:
            self.indices_n_gram = torch.tensor([[self.sos_token_id]]).to(self.device)
            self.mask_indices_n_gram = (self.indices_n_gram != self.pad_token_id).type(torch.int64).to(self.device)


    def copy(self):
        new_item = SearchNode(
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            sos_token_id=self.sos_token_id,
            tokenizer_tgt=self.tokenizer_tgt,
            tokenizer_src=self.tokenizer_src,
            device=self.device,
            max_len=self.max_len,
            encoder_output=self.encoder_output,
            encoder_output_mask=self.src_attention_mask,
            n_gram=self.n_gram,
        )
        new_item.tgt = self.tgt
        new_item.tgt_attention_mask = self.tgt_attention_mask
        new_item.last_token = self.last_token
        new_item.num_steps = self.num_steps
        new_item.scores = self.scores
        new_item.past_key_values = self.past_key_values
        new_item.past_attn_scores = self.past_attn_scores
        new_item.last_n_gram_indices = self.last_n_gram_indices
        new_item.mask_indices_n_gram = self.mask_indices_n_gram
        new_item.indices_n_gram = self.indices_n_gram
        new_item.mask_indices_n_gram = self.mask_indices_n_gram
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
        if self.stop_search():
            if self.last_n_gram_indices is None:
                self.last_n_gram_indices = torch.tensor([self.pad_token_id]).to(self.device)
            else:
                self.last_n_gram_indices = torch.cat([
                    self.last_n_gram_indices,
                    torch.tensor([self.pad_token_id]).to(self.device)
                ], dim=-1)
        else:
            if self.last_n_gram_indices is None:
                self.last_n_gram_indices = torch.tensor([indice]).to(self.device)
            else:
                self.last_n_gram_indices = torch.cat([
                    self.last_n_gram_indices,
                    torch.tensor([indice]).to(self.device)
                ], dim=-1)
        if self.last_n_gram_indices is not None and self.last_n_gram_indices.size(0) == self.n_gram:
            last_n_gram_indices_cpu = self.last_n_gram_indices.cpu()
            last_n_gram_indices_numpy = last_n_gram_indices_cpu.unsqueeze(0).numpy()
            if self.indices_n_gram is None:
                self.indices_n_gram = torch.tensor(last_n_gram_indices_numpy).to(self.device)
            else:
                self.indices_n_gram = torch.cat([
                    self.indices_n_gram,
                    self.last_n_gram_indices.unsqueeze(0)
                ], dim=0)
            # print(f"{ self.indices_n_gram.size() = }")
            self.mask_indices_n_gram = (self.indices_n_gram != self.pad_token_id).type(torch.int64).to(self.device)
            self.last_n_gram_indices = self.last_n_gram_indices[1:]
        if self.last_n_gram_indices is not None:
            self.mask_last_n_gram_indices = (self.last_n_gram_indices != self.pad_token_id).type(torch.int64).to(self.device)            

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
    "SearchNode",
    "TYPE_SEARCH",
    "BEAM_SEARCH",
    "DIVERSE_BEAM_SEARCH",
]