from .bart.seq2seq import get_model as get_bart_seq2seq_from_scratch
from .bart.classification import get_model as get_bart_classification_from_scratch
from .bart.fine_tune_seq2seq import get_model as get_fine_tune_seq2seq_from_scratch
from .bart.fine_tune_seq2seq_with_random_encoder import get_model as get_fine_tune_seq2seq_with_random_encoder_from_scratch
from .bart.seq2seq_transformers import get_model as get_seq2seq_transformers

BART_SEQ2SEQ_FROM_SCRATCH = "bart_seq2seq_from_scratch"
BART_CLASSIFICATION_FROM_SCRATCH = "bart_classification_from_scratch"
FINE_TUNE_BART_SEQ2SEQ_FROM_SCRATCH = "fine_tune_bart_seq2seq_from_scratch"
FINE_TUNE_SEQ2SEQ_WITH_RANDOM_ENCODER_FROM_SCRATCH = "fine_tune_seq2seq_with_random_encoder_from_scratch"
BART_SEQ2SEQ_TRANSFORMERS = "bart_seq2seq_transformers"

GET_MODEL = {
    BART_SEQ2SEQ_FROM_SCRATCH: get_bart_seq2seq_from_scratch,
    BART_CLASSIFICATION_FROM_SCRATCH: get_bart_classification_from_scratch,
    FINE_TUNE_BART_SEQ2SEQ_FROM_SCRATCH: get_fine_tune_seq2seq_from_scratch,
    FINE_TUNE_SEQ2SEQ_WITH_RANDOM_ENCODER_FROM_SCRATCH: get_fine_tune_seq2seq_with_random_encoder_from_scratch,
    BART_SEQ2SEQ_TRANSFORMERS: get_seq2seq_transformers,
}

def get_model(config, model_train):
    get_model_fn = GET_MODEL[model_train]
    return get_model_fn(**config)

__all__ = ["get_model"]