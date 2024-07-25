# Token Dropping BERT Pretraining

To pretrain a token dropping BERT model, we use the same pretraining corpus as used for pretraining the BERT model. However, the pretraining data generation is slightly different because we use the RoBERTa-style packing strategy, as was used in the original  [Token Dropping BERT](https://aclanthology.org/2022.acl-long.262/) paper.

Short summary of the RoBERTa-style packing: The pretraining corpus is split into chunks of a fixed size (512 in our case) - it can also cross document boundaries!

## TFRecord Generation - BERT Original Packing

Our `create_pretraining_data.py` script is then used to create TFRecords. Please note the `roberta` option as packing strategy. Another difference, compared to the TFRecord generation using the original BERT packing, is that we use a dupe factor of 2 to approximately match the number of pretraining instances of the BERT packing output.

```bash
find ./pretraining-corpus/part-{00..14}/ -type f -iname "part-*" | sort |
xargs -I% -P 6 \
python3 create_pretraining_data.py \
--max_seq_length=512 \
--max_predictions_per_seq=76 \
--masked_lm_prob=0.15 \
--random_seed=12345 \
--dupe_factor=2 \
--use_v2_feature_names \
--input_file % \
--output_file %_original.tfrecord \
--packing_strategy roberta \
--tokenizer_model_id stefan-it/fineweb-lms-vocab-64000
```

All TFRecords need to be uploaded to a GCP Bucket.

## Start Pretraining

The pretraining needs to be started within the `models/official/projects/token_dropping` folder. The following configuration file, `fineweb_pretrain.yaml`, is used and includes all necessary model and training information:

https://github.com/stefan-it/model-garden-lms/blob/a2b25404cb8ca206e76df0949c589561e6a5a23a/token-dropping-bert/fineweb_pretrain.yaml#L1-L26

Please note that this configuration is slightly different compared to the configuration found in the TF Model Garden repository for the (English) BERT model. You must ensure that the `token_allow_list` mapping matches the order in your vocab.

In our vocabulary, the special tokens (`[UNK]`, `[CLS]`, `[SEP]`, `[MASK]` and `[PAD]`) are in different index positions than in the vocabulary of the original BERT model. If you have a custom vocabulary, please adjust the index positions according to your needs.

Training-specific configurations are stored in the `fineweb_pretrain_sequence_pack.yaml` file:

https://github.com/stefan-it/model-garden-lms/blob/a2b25404cb8ca206e76df0949c589561e6a5a23a/token-dropping-bert/fineweb_pretrain_sequence_pack.yaml#L1-L38

Please also make sure that you are using the correct folder on GCP (e.g., TFRecords with BERT packing are stored in the `tfrecords-op` folder, whereas the RoBERTa packed TFRecords are stored in the `tfrecords-rp` folder).

Then the pretraining can be started by running:

```bash
$ python3 train.py --experiment=token_drop_bert/pretraining \
--config_file=fineweb_pretrain.yaml \
--config_file=fineweb_pretrain_sequence_pack.yaml \
--params_override=runtime.distribution_strategy=tpu \
--tpu=fineweb \
--model_dir=gs://fineweb-lms/models/fineweb-10BT-token-dropping-bert \
--mode=train
```
