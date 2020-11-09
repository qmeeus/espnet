from pathlib import Path
from tokenizers.models import Unigram, BPE, WordPiece
from tokenizers.normalizers import NFD, StripAccents, Lowercase
from tokenizers.pre_tokenizers import Whitespace, Digits, Punctuation
from tokenizers.trainers import UnigramTrainer, BpeTrainer, WordPieceTrainer
from tokenizers import (
    Tokenizer,
    normalizers,
    pre_tokenizers,
)

VOCAB_SIZE = 5000
TEXTFILE = Path("~/spchdisk/repos/espnet/egs/cgn/asr1/data/lang_unigram/input.txt").expanduser()
MODEL_CLASS = BPE
TRAINER_CLASS = BpeTrainer
OUTDIR = "data/tokenizer"


def build_tokenizer(model_class):
    tokenizer = Tokenizer(model_class())
    tokenizer.normalizer = normalizers.Sequence([Lowercase(), NFD(), StripAccents()])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])
    return tokenizer

def build_trainer(trainer_class, vocab_size=None):
    return trainer_class(
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BLANK]"],
        vocab_size=vocab_size,
        unk_token="[UNK]"
    )


if __name__ == "__main__":
    tokenizer = build_tokenizer(MODEL_CLASS)
    trainer = build_trainer(TRAINER_CLASS, VOCAB_SIZE)
    tokenizer.train(trainer, [str(TEXTFILE)])
    tokenizer.save(f"{OUTDIR}/tokenizer.json")

