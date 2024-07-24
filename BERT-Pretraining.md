# BERT Pretraining

The original [BERT](https://github.com/google-research/bert) implementation can no longer be used for pretraining on TPUs because it was written for TensorFlow 1.15, which is very old and deprecated. This version can no longer be installed on TPUs (both legacy and TPU VMs).

However, the TensorFlow Model Garden library offers a great alternative. In this section we will pretrain a BERT model on the recently released [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) and [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) corpora by Hugging Face.

More precisely, we used the 10BT sample data packages (so 20BT in total) resulting in 86GB of plaintext data.

## Preprocessing FineWeb corpora

As the original BERT implementation uses the next sentence prediction task, we also create a sentence-split pretraining corpus. We use an own `datatrove` plugin that performs sentence splitting with NLTK and plaintext extraction.

The implementation, saved under `plaintext_writer.py`, looks like this:

```python
from typing import IO, Callable

from datatrove.io import DataFolderLike
from datatrove.pipeline.writers.disk_base import DiskWriter

from nltk.tokenize import sent_tokenize


class PlaintextWriter(DiskWriter):
    """Write plain text data to datafolder (local or remote)

    Args:
        output_folder: a str, tuple or DataFolder where data should be saved
        output_filename: the filename to use when saving data, including extension. Can contain placeholders such as `${rank}` or metadata tags `${tag}`
        perform_sentence_splitting: whether to perform sentence splitting or not. Enabled by default
        sentence_splitting_language: defines the language used for sentence splitting. Set to english by default
        compression: if any compression scheme should be used. By default, no compression is used
        adapter: a custom function to "adapt" the Document format to the desired output format
    """

    default_output_filename: str = "${rank}.txt"
    name = "📑 Plaintext writer"

    def __init__(
        self,
        output_folder: DataFolderLike,
        output_filename: str = None,
        perform_sentence_splitting: bool = True,
        sentence_splitting_language: str = "english",
        compression: str | None = None,
        adapter: Callable = None,
    ):
        self.perform_sentence_splitting = perform_sentence_splitting
        self.sentence_splitting_language = sentence_splitting_language
        super().__init__(
            output_folder,
            output_filename=output_filename,
            compression=compression,
            adapter=adapter,
            mode="wt",
            max_file_size=-1,
        )

    def _write(self, document: dict, file_handler: IO, _filename: str):
        import re

        content = document["text"]

        if self.perform_sentence_splitting:
            content = content.replace("\n", "")
            sentences = sent_tokenize(content, self.sentence_splitting_language)
            file_handler.write("\n".join(sentences) + "\n\n")
        else:
            content = re.sub(r"\n+", "\n", content).strip()
            file_handler.write(content + "\n\n")
```

Then the preprocessing pipeline can be started with:

```python
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader

from plaintext_writer import PlaintextWriter


pipeline_exec = LocalPipelineExecutor(
    pipeline=[
        ParquetReader("hf://datasets/HuggingFaceFW/fineweb/sample/10BT"),
        ParquetReader("hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT"),
        PlaintextWriter(output_folder="./extracted-plaintext-corpus", perform_sentence_splitting=True, sentence_splitting_language="english", compression=None),
    ],
    tasks=15,
    workers=15,
)
pipeline_exec.run()
```

This results in 86GB plaintext:

```bash
$ du -csh extracted-plaintext-corpus/
86G     extracted-plaintext-corpus/
86G     total
```

## Pretraining Data Generation

In the next step, we first split our plaintext corpus into 400M chunks. These chunks are later used for generating the pretraining data (TFRecords). The following script, `split_plaintext_corpus.sh`, can be used:

```bash
# The final, splitted pretraining corpus output folder
PRETRAINING_CORPUS_PATH=./pretraining-corpus

# Path, where the extracted 000{00..14}.txt files are located at
EXTRACTED_CORPUS_PATH=./extracted-plaintext-corpus

# We use 400M chunks (it is maybe a hyper-parameter or not...)
SPLIT_SIZE=400M

mkdir -p ${PRETRAINING_CORPUS_PATH}

for index in $(seq -w 0 14)
do
    echo Splitting 000${index}.txt ...

    mkdir -p ${PRETRAINING_CORPUS_PATH}/part-${index}

    split -C ${SPLIT_SIZE} -d ${EXTRACTED_CORPUS_PATH}/000${index}.txt ${PRETRAINING_CORPUS_PATH}/part-${index}/part-${index}-
done
```

## Vocab Generation

Next, we will train a WordPiece vocabulary using the Tokenizers library. We use the complete FineWeb and FineWeb-Edu corpora for training our 64k vocab:

```python
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
```

The `vocab.txt` together with a `config.json` and `tokenizer_config.json` is then uploaded to the Hugging Face Model Hub:

* [`stefan-it/fineweb-lms-vocab-64000`](https://huggingface.co/stefan-it/fineweb-lms-vocab-64000)

## TFRecord Generation - BERT Original Packing

After the vocab generation and uploading it to the Model Hub, our `create_pretraining_data.py` script can be used to create TFRecords that are later used for pretraining the model:

```bash
find ./pretraining-corpus/part-{00..14}/ -type f -iname "part-*" | sort | 
xargs -I% -P 6 \
python3 create_pretraining_data.py \
--max_seq_length=512 \
--max_predictions_per_seq=76 \
--masked_lm_prob=0.15 \
--random_seed=12345 \
--dupe_factor=1 \
--use_v2_feature_names \
--input_file % \
--output_file %_original.tfrecord \
--packing_strategy original \
--tokenizer_model_id stefan-it/fineweb-lms-vocab-64000
```

All TFRecords need to be uploaded to a GCP Bucket (the `service` TPU user must have `Storage Administrator` permissions). More hints can be found in this [section](https://github.com/GermanT5/pre-training?tab=readme-ov-file#preparing-gcp-bucket) about preparing a GCP bucket.

## Start Pretraining

The pretraining needs to be started within the `models/official/nlp/` folder. The following configuration file, `fineweb_pretrain.yaml`, is used and includes all necessary model and training information:

```yaml
task:
  init_checkpoint: ''
  model:
    cls_heads: [{activation: tanh, cls_token_idx: 0, dropout_rate: 0.1, inner_dim: 768, name: next_sentence, num_classes: 2}]
    encoder:
      type: bert_v2
      bert_v2:
        vocab_size: 64000
  train_data:
    drop_remainder: true
    global_batch_size: 512
    input_path: 'gs://fineweb-lms/tfrecords-op/*.tfrecord'
    is_training: true
    max_predictions_per_seq: 76
    seq_length: 512
    use_next_sentence_label: true
    use_position_id: false
    use_v2_feature_names: true
trainer:
  checkpoint_interval: 50000
  max_to_keep: 300
  optimizer_config:
    learning_rate:
      polynomial:
        cycle: false
        decay_steps: 1000000
        end_learning_rate: 0.0
        initial_learning_rate: 0.0001
        power: 1.0
      type: polynomial
    optimizer:
      type: adamw
    warmup:
      polynomial:
        power: 1
        warmup_steps: 10000
      type: polynomial
  steps_per_loop: 1000
  summary_interval: 1000
  train_steps: 1000000
  validation_interval: 1000
  validation_steps: 64
```

Then the pretraining can be started by running:

```bash
$ python3 train.py --experiment=bert/pretraining \
--config_file=fineweb_pretrain.yaml \
--params_override=runtime.distribution_strategy=tpu \
--tpu=fineweb \
--model_dir=gs://fineweb-lms/models/fineweb-10BT-bert \
--mode=train
```
