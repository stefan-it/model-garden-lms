from pathlib import Path
from tokenizers import BertWordPieceTokenizer

pretraining_corpus = Path("./pretraining-corpus")

pretraining_files = [str(file) for file in pretraining_corpus.rglob("part-*")]

tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=False,
)

trainer = tokenizer.train(
    chosen_files,
    vocab_size=vocab_size,
    min_frequency=2,
    show_progress=True,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    limit_alphabet=1000,
    wordpieces_prefix="##",
)

tokenizer.save_model("./fineweb-lms-vocab-64000")
