# TensorFlow Model Garden LMs

<p align="center">
  <picture>
    <img alt="BERT with TensorFlow Model Garden" src="https://github.com/stefan-it/model-garden-lms/raw/main/bert_tf_model_garden.png" style="max-width: 100%;">
  </picture>
  <br/>
</p>

This repository show-cases language model pretraining with the awesome [TensorFlow Model Garden](https://github.com/tensorflow/models) library.

The following LMs are current supported:

* [BERT Pretraining](https://aclanthology.org/N19-1423/) - see [pretraining instructions](BERT-Pretraining.md)
* [Token Dropping for efficient BERT Pretraining](https://aclanthology.org/2022.acl-long.262/)
* [Training ELECTRA Augmented with Multi-word Selection](https://aclanthology.org/2021.findings-acl.219/) (TEAMS) - see [pretraining instructions](TEAMS-Pretraining.md)

Additionally, the following features are provided:

* Cheatsheet for TPU VM creation (incl. all necessary dependencies to pretrain models with TF Model Garden library), which can be found [here](TPU-VM-Cheatsheet.md).
* Extended pretraining data generation script that allows e.g. to use tokenizer from Hugging Face Model Hub or different data packing strategies (Original BERT packing or RoBERTa-like packing), which can be found [here](Pretraining-Data-Generation.md).
* Conversion scripts that convert TF Model Garden weights to Hugging Face Transformers-compatible models, which can be found [here](Model-Conversion.md).

❤️ This repository is the outcome of the last two years working with TPUs from the awesome [TRC program](https://sites.research.google/trc/about/) and the [TensorFlow Model Garden](https://github.com/tensorflow/models) library.

