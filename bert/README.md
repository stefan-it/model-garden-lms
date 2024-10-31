# BERT Pretraining

The original [BERT](https://github.com/google-research/bert) implementation can no longer be used for pretraining on TPUs because it was written for TensorFlow 1.15, which is very old and deprecated. This version can no longer be installed on TPUs (both legacy and TPU VMs).

However, the TensorFlow Model Garden library offers a great alternative. In this section we will pretrain a BERT model on the recently released [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) and [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) corpora by Hugging Face.

More precisely, we used the 10BT sample data packages (so 20BT in total) resulting in 86GB of plaintext data.

## Preprocessing FineWeb corpora

As the original BERT implementation uses the next sentence prediction task, we also create a sentence-split pretraining corpus. We use an own `datatrove` plugin that performs sentence splitting with NLTK and plaintext extraction.

The implementation, saved under `plaintext_writer.py`, looks like this:

https://github.com/stefan-it/model-garden-lms/blob/d49919e25574c0753fd49cb4f09eeb5bbb4aa58a/bert/plaintext_writer.py#L9-L19

Then the preprocessing pipeline can be started with:

https://github.com/stefan-it/model-garden-lms/blob/d49919e25574c0753fd49cb4f09eeb5bbb4aa58a/bert/pipeline.py#L1-L16

This results in 86GB plaintext:

```bash
$ du -csh extracted-plaintext-corpus/
86G     extracted-plaintext-corpus/
86G     total
```

## Pretraining Data Generation

In the next step, we first split our plaintext corpus into 400M chunks. These chunks are later used for generating the pretraining data (TFRecords). The following script, `split_plaintext_corpus.sh`, can be used:

https://github.com/stefan-it/model-garden-lms/blob/d49919e25574c0753fd49cb4f09eeb5bbb4aa58a/bert/split_plaintext_corpus.sh#L1-L19

## Vocab Generation

Next, we will train a WordPiece vocabulary using the Tokenizers library. We use the complete FineWeb and FineWeb-Edu corpora for training our 64k vocab:

https://github.com/stefan-it/model-garden-lms/blob/d49919e25574c0753fd49cb4f09eeb5bbb4aa58a/bert/train_vocab.py#L1-L25

The `vocab.txt` together with a `config.json` and `tokenizer_config.json` is then uploaded to the Hugging Face Model Hub:

* [`stefan-it/fineweb-lms-vocab-64000`](https://huggingface.co/stefan-it/fineweb-lms-vocab-64000)

## TFRecord Generation - BERT Original Packing

After the vocab generation and uploading it to the Model Hub, our `create_pretraining_data.py` script can be used to create TFRecords that are later used for pretraining the model:

```bash
find ./pretraining-corpus/part-{00..14}/ -type f -iname "part-*" | sort |
xargs -I% -P 6 \
python3 ../utils/create_pretraining_data.py \
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

More details about pretraining data generation can be found in [this section](../utils/README.md).

All TFRecords need to be uploaded to a GCP Bucket (the `service` TPU user must have `Storage Administrator` permissions). More hints can be found in this [section](https://github.com/GermanT5/pre-training?tab=readme-ov-file#preparing-gcp-bucket) about preparing a GCP bucket.

## Start Pretraining

The pretraining needs to be started within the `models/official/nlp/` folder. The following configuration file, `fineweb_pretrain.yaml`, is used and includes all necessary model and training information:

https://github.com/stefan-it/model-garden-lms/blob/d49919e25574c0753fd49cb4f09eeb5bbb4aa58a/bert/fineweb_pretrain.yaml#L1-L42

Then the pretraining can be started by running:

```bash
$ python3 train.py --experiment=bert/pretraining \
--config_file=fineweb_pretrain.yaml \
--params_override=runtime.distribution_strategy=tpu \
--tpu=fineweb \
--model_dir=gs://fineweb-lms/models/fineweb-10BT-bert \
--mode=train
```
