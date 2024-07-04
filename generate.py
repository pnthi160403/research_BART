import torch

from .utils.search import (
    TYPE_SEARCH,
    SearchItem,
)

# length penalty
def sequence_length_penalty(length: int, alpha: float=0.6) -> float:
    return ((5 + length) / (5 + 1)) ** alpha

# beam search
def generate(model, config, beam_size, tokenizer_src, tokenizer_tgt, src):
    model.eval()
    # print(f"{ beam_size = }")
    # Search Module
    # special token id
    sos_token_id = tokenizer_src.token_to_id("<s>")
    eos_token_id = tokenizer_src.token_to_id("</s>")
    pad_token_id = tokenizer_src.token_to_id("<pad>")
    special_tokens = {
        "<s>": sos_token_id,
        "</s>": eos_token_id,
        "<pad>": pad_token_id,
    }
    type_search = config["type_search"]
    vocab_size = tokenizer_tgt.get_vocab_size()
    search_module = TYPE_SEARCH[type_search](
        special_tokens=special_tokens,
        vocab_size=vocab_size,
        num_groups=config["num_groups_search"],
        diversity_strength=config["diversity_strength_search"],
        diversity_discount=config["diversity_discount_search"],
        candidate_multiple=config["candidate_multiple_search"],
    )

    sos_token = torch.tensor([tokenizer_tgt.token_to_id("<s>")], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer_tgt.token_to_id("</s>")], dtype=torch.int64)
    
    device = config["device"]
    max_len = config["max_len"]

    enc_input_tokens = tokenizer_src.encode(src).ids
    src = torch.cat(
        [
            sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            eos_token,
        ],
        dim=0,
    ).to(device)

    encoder_output = model.get_encoder_out(
        input_ids=torch.tensor(src, dtype=torch.int64).unsqueeze(0).to(device),
    ).last_hidden_state
    encoder_output_mask = (src != pad_token_id).type(torch.int64).to(device)

    candidates = [SearchItem(
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        sos_token_id=sos_token_id,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        encoder_output=encoder_output,
        encoder_output_mask=encoder_output_mask,
        device=device,
        max_len=max_len,
    )] * beam_size

    for step in range(max_len):
        if all([candidate.stop_search() for candidate in candidates]):
            break
        new_candidates = []
        lprobs = []
        scores = None
        candidates_past_key_values = []
        candidates_past_attn_scores = []
        # mask (batch_size, beam_size)
        mask = torch.ones((1, beam_size)).type(torch.int64).to(device)
        for input_beam in range(beam_size):
            # print(f"{ input_beam = }")
            candidate = candidates[input_beam]
            if candidate.stop_search():
                mask[0][input_beam] = 0
                # lprob (1, vocab_size)
                lprob = torch.zeros((1, vocab_size), dtype=torch.float32).to(device)
            else:
                if config["use_cache"]:
                    decoder_out_obj = model.get_decoder_out(
                        input_ids=candidate.tgt[-1:].unsqueeze(0),
                        encoder_hidden_states=candidate.encoder_output,
                        past_key_values=candidate.past_key_values,
                        past_attn_scores=candidate.past_attn_scores,
                        use_cache=True,
                        pos_idx=len(candidate.tgt) - 1,
                    )
                else:
                    decoder_out_obj = model.get_decoder_out(
                        input_ids=candidate.tgt.unsqueeze(0),
                        attention_mask=candidate.tgt_attention_mask.unsqueeze(0),
                        encoder_hidden_states=candidate.encoder_output,
                        encoder_attention_mask=candidate.src_attention_mask.unsqueeze(0),
                    )
                decoder_out = decoder_out_obj.last_hidden_state
                past_key_values = decoder_out_obj.past_key_values
                past_attn_scores = decoder_out_obj.past_attn_scores
                
                out = model.out(decoder_out)
                # lprob (1, vocab_size)
                lprob = torch.nn.functional.log_softmax(out[:, -1], dim=1)
                lprob = lprob / sequence_length_penalty(len(candidate.tgt), alpha=0.6)
                # print(f"{ lprob.shape = }")
            lprobs.append(lprob)
            if step > 0:
                # print(f"{ input_beam = }")
                # print(f"{ candidate.scores = }")
                # print()
                if scores is None:
                    scores = [candidate.scores.unsqueeze(0)]
                else:
                    scores.append(candidate.scores.unsqueeze(0))
            candidates_past_key_values.append(past_key_values)
            candidates_past_attn_scores.append(past_attn_scores)
        
        # lprobs (batch_size, beam_size, vocab_size)
        lprobs = torch.cat(lprobs, dim=0).unsqueeze(0)
        # scores (batch_size, beam_size, step)
        if step > 0 and scores is not None:
            scores = torch.cat(scores, dim=0).unsqueeze(0)
        # print(f"{ step = }")
        # print(f"{ lprobs.shape = }")
        # if scores is not None:
            # print(f"{ scores.shape = }")
        # print(f"{ mask.shape = }")

        scores, indices, beams = search_module.step(
            step=step,
            lprobs=lprobs,
            scores=scores,
            original_batch_idxs=torch.tensor([0]).to(device),
            mask=mask,
        )
        # print(f"{ scores.shape = }")
        # print(f"{ indices.shape = }")
        # print(f"{ beams.shape = }")
        # print(f"{ scores = }")
        # print(f"{ indices = }")
        # print(f"{ beams = }")
        # print()

        for output_beam in range(config["candidate_multiple_search"] * beam_size):
            # print(f"{ output_beam = }")
            input_beam = beams[0][output_beam]
            # copy candidate
            candidate = candidates[input_beam].copy()
            # print(f"{ scores[0][output_beam] = }")
            candidate.step(
                score=scores[0][output_beam],
                indice=indices[0][output_beam],
                past_key_values=candidates_past_key_values[input_beam],
                past_attn_scores=candidates_past_attn_scores[input_beam],
            )
            # print(f"{ candidate.scores = }")
            # print(f"{ input_beam = }")
            # print(f"{ scores[0, output_beam].item() = }")
            # print(f"{ indices[0, output_beam].item() = }")
            # print(f"{ candidates_past_attn_scores[input_beam] = }")
            # print(f"{ candidates_past_key_values[input_beam] = }")
            new_candidates.append(candidate)

        # del all elements in candidates from memory
        del candidates

        # sort by score
        candidates = sorted(new_candidates, key=lambda x: x.scores[-1], reverse=True)
        candidates = candidates[:beam_size]
    pred_ids = candidates[0].tgt.squeeze()
    return pred_ids