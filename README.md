# 🏡 TensorFlow Model Garden LMs

<p align="center">
  <picture>
    <img alt="BERT with TensorFlow Model Garden" src="https://github.com/stefan-it/model-garden-lms/raw/main/bert_tf_model_garden.png" style="max-width: 100%;">
  </picture>
  <br/>
</p>

## 🔎 Overview

This repository showcases language model pretraining with the awesome [TensorFlow Model Garden](https://github.com/tensorflow/models) library.

The following LMs are currently supported:

* [BERT Pretraining](https://aclanthology.org/N19-1423/) - see [pretraining instructions](bert/README.md)
* [Token Dropping for efficient BERT Pretraining](https://aclanthology.org/2022.acl-long.262/) - see [pretraining instructions](token-dropping-bert/README.md)
* [Training ELECTRA Augmented with Multi-word Selection](https://aclanthology.org/2021.findings-acl.219/) (TEAMS) - see [pretraining instructions](teams/README.md)


## 💡 Features

Additionally, the following features are provided:

* A cheatsheet for TPU VM creation (including all necessary dependencies to pretrain models with TF Model Garden library), which can be found [here](cheatsheet/README.md).
* An extended pretraining data generation script that allows, for example, the use of tokenizers from the Hugging Face Model Hub or different data packing strategies (Original BERT packing or RoBERTa-like packing), which can be found [here](utils/README.md).
* Conversion scripts that convert TF Model Garden weights to Hugging Face Transformers-compatible models, which can be found [here](conversion/README.md).

## 🏡 Model Zoo

### FineWeb-LMs

Following LMs were pretrained on the (10BT subset) of the famous [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) and [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset:

* BERT-based - find the [best model checkpoint here](https://huggingface.co/model-garden-lms/bert-base-finewebs-951k)

## ❤️ Acknowledgements

This repository is the outcome of the last two years of working with TPUs from the awesome [TRC program](https://sites.research.google/trc/about/) and the [TensorFlow Model Garden](https://github.com/tensorflow/models) library.

Made from Bavarian Oberland with ❤️ and 🥨.
