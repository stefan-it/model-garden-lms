# Weights conversion from TF Model Garden to HF Transformers

Some historical notes:

The Token Dropping BERT and TEAMS papers come with a great and stable code release into TF Model Garden library. Unfortunately, no models were released.

For the [hmBERT](https://arxiv.org/abs/2205.15575) paper we open sourced the first ever Token Dropping BERT model incl. a conversion scripts that created a HF Transformers-compatible BERT model. The script was merged into official code base of HF Transformers, see this [PR](https://github.com/huggingface/transformers/pull/17142).

We later found out, that the same conversion script can also be used to convert "normal" BERT models, that were pretrained with TF Model Garden library. Thus, we also include this conversion script in this repository here.

One year later the TEAMS approach came out and we pretrained a [hmTEAMS](https://github.com/stefan-it/hmTEAMS) model and also a model that was trained on [German Wikipedia only](https://huggingface.co/gwlms/teams-base-dewiki-v1-discriminator). For TEAMS we also wrote a conversion that creates a HF Transformers-compatible ELECTRA model.

# BERT and Token Dropping BERT

The [conversion script](convert_bert_token_dropping_original_tf2_checkpoint_to_pytorch.py) `convert_bert_token_dropping_original_tf2_checkpoint_to_pytorch.py` can be used to convert models - trained with TF Model Garden - to HF Transformers-compatible models:

```bash
$ python3 convert_bert_token_dropping_original_tf2_checkpoint_to_pytorch.py --help
usage: convert_bert_token_dropping_original_tf2_checkpoint_to_pytorch.py [-h] --tf_checkpoint_path TF_CHECKPOINT_PATH
                                                                         --bert_config_file BERT_CONFIG_FILE --pytorch_dump_path
                                                                         PYTORCH_DUMP_PATH

options:
  -h, --help            show this help message and exit
  --tf_checkpoint_path TF_CHECKPOINT_PATH
                        Path to the TensorFlow Token Dropping checkpoint path.
  --bert_config_file BERT_CONFIG_FILE
                        The config json file corresponding to the BERT model. This specifies the model architecture.
  --pytorch_dump_path PYTORCH_DUMP_PATH
                        Path to the output PyTorch model.
```

A demo config file (`config.json`) can be found [here](https://huggingface.co/gwlms/bert-base-token-dropping-dewiki-v1/blob/main/config.json).

# TEAMS

The [conversion script](convert_teams_original_tf2_checkpoint_to_pytorch.py) `convert_teams_original_tf2_checkpoint_to_pytorch.py` can be used to convert models - trained with TF Model Garden - to HF Transformers-compatible ELECTRA models:

```bash
$ usage: convert_teams_original_tf2_checkpoint_to_pytorch.py [-h] --tf_checkpoint_path TF_CHECKPOINT_PATH --config_file CONFIG_FILE
                                                           --pytorch_dump_path PYTORCH_DUMP_PATH --discriminator_or_generator
                                                           DISCRIMINATOR_OR_GENERATOR

options:
  -h, --help            show this help message and exit
  --tf_checkpoint_path TF_CHECKPOINT_PATH
                        Path to the TensorFlow TEAMS checkpoint path.
  --config_file CONFIG_FILE
                        The config json file corresponding to the TEAMS model. This specifies the model architecture.
  --pytorch_dump_path PYTORCH_DUMP_PATH
                        Path to the output PyTorch model.
  --discriminator_or_generator DISCRIMINATOR_OR_GENERATOR
                        Whether to export the generator or the discriminator. Should be a string, either 'discriminator' or
                        'generator'.
```

Please note that you need to specify whether a discriminator or generator model should be exported. Here are example configurations for a [discriminator](https://huggingface.co/gwlms/teams-base-dewiki-v1-discriminator/blob/main/config.json) and [generator](https://huggingface.co/gwlms/teams-base-dewiki-v1-generator/blob/main/config.json) model.
