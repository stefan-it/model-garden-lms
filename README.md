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
* Token Dropping BERT-based - find the [best model checkpoint here](https://huggingface.co/model-garden-lms/bert-base-token-dropping-finewebs-901k)
* TEAMS-based - fine the [best model checkpoint here](https://huggingface.co/model-garden-lms/teams-base-finewebs-1m)

All models can be found in the [TensorFlow Model Garden LMs](https://huggingface.co/model-garden-lms) organization on the Model Hub and in [this collection](https://huggingface.co/collections/stefan-it/fineweb-lms-67561ed9d83c390221aaa2d4).

Detailed evaluation results with the [ScandEval](https://github.com/ScandEval/ScandEval) library are available in [this repository](https://huggingface.co/datasets/model-garden-lms/finewebs-scandeval-results).

## ❤️ Acknowledgements

This repository is the outcome of the last two years of working with TPUs from the awesome [TRC program](https://sites.research.google/trc/about/) and the [TensorFlow Model Garden](https://github.com/tensorflow/models) library.

Made from Bavarian Oberland with ❤️ and 🥨.
