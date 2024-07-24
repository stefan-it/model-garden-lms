# Token Dropping BERT Pretraining

In order to pretrain a token dropping BERT model, we use the same pretraining corpus as used for pretraining the BERT model. However, the pretraining data generation is slightly different, because we use the RoBERTa-style packing strategy, as this was also used in the original [Token Dropping BERT](https://aclanthology.org/2022.acl-long.262/) paper.

Short summary of the RoBERTa-style packing: the pretraining corpus is splitted into chunks of a fixed size (512 in our case) - it can also cross document boundaries!

## TFRecord Generation - BERT Original Packing

Our `create_pretraining_data.py` script is then used to create TFRecords - please note the `roberta` option as packing strategy. Another difference - compared to the TFRecord generation using the original BERT packing - is that we use a dupe factor of 2 to approx. match the number of pretraining instances of the BERT packing output!

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

The pretraining needs to be started within the `models/official/projects/token_dropping` folder. The following configuration file `fineweb_pretrain.yaml` is used that includes all necessary model and training information:

```yaml
task:
  model:
    encoder:
      type: any
      any:
        token_allow_list: !!python/tuple
        - 1  # [UNK]
        - 2  # [CLS]
        - 3  # [SEP]
        - 4  # [MASK]
        token_deny_list: !!python/tuple
        - 0  # [PAD]
        attention_dropout_rate: 0.1
        dropout_rate: 0.1
        hidden_activation: gelu
        hidden_size: 768
        initializer_range: 0.02
        intermediate_size: 3072
        max_position_embeddings: 512
        num_attention_heads: 12
        num_layers: 12
        type_vocab_size: 2
        vocab_size: 32000
        token_loss_init_value: 10.0
        token_loss_beta: 0.995
        token_keep_k: 256
```

Please not that this configuration is sligtly different compared to the configuration that you can find in the TF Model Garden repository for the (English) BERT model. You have to make sure, that the `token_allow_list` mapping matches the order in your vocab.

In our vocabulary, the special tokens (`[UNK]`, `[CLS]`, `[SEP]`, `[MASK]` and `[PAD]`) are on different index positions than in the vocabulary of the original BERT model. If you have a fancy vocabulary, please adjust the index positions according to your needs.

Training-specific configurations are stored in the `fineweb_pretrain_sequence_pack.yaml`:

```yaml
task:
  init_checkpoint: ''
  model:
    cls_heads: []
  train_data:
    drop_remainder: true
    global_batch_size: 512
    input_path: "gs://fineweb-lms/tfrecords-rp/*.tfrecord" 
    is_training: true
    max_predictions_per_seq: 76
    seq_length: 512
    use_next_sentence_label: false
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

Please also make sure that you are using the correct folder on GCP (e.g. TFRecords with BERT packing are stored in the `tfrecords-op` folder, whereas the RoBERTa packed TFRecords are stored in the `tfrecords-rp` folder).

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
