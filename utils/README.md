# Pretraining Data Generation

This section covers the modification of the original script used for generating the pretraining data (the outcome being TFRecords).

We use the `create_pretraining_data.py` [implementation](https://github.com/tensorflow/models/blob/master/official/nlp/data/create_pretraining_data.py) from the TF Model Garden repository, which traces back to the original implementation taken from the [BERT repository](https://github.com/google-research/bert/blob/master/create_pretraining_data.py).

The following modifications were performed:

* Code style: Mainly four spaces instead of two; Black was used to format it.
* Tokenizer: The Hugging Face Tokenizer is used instead of the TF Model Garden tokenization logic.
* Vocabulary file: There's no need to pass a `vocab.txt` file because a Hugging Face Model Hub identifier can be used.
* Packing methods: In addition to the original BERT packing, two new packing methods can be used: RoBERTa-style (`FULL-SENTENCES`) packing and Best-Fit packing.

Our modified version can be found [here](create_pretraining_data.py).

# Showcase

The RoBERTa-style packing strategy uses the whole input corpus and chunks it into a fixed sequence length (typically 512). The `[CLS]` and `[SEP]` special tokens are automatically prepended/appended. This strategy allows for pretraining BERT models with the Token Dropping approach.

To use this packing strategy, the `packing_strategy` parameter needs to be set to `roberta`.

Additionally, the Best-Fit packing is an alternative approach presented in the [Fewer Truncations Improve Language Modeling](https://arxiv.org/abs/2404.10830) paper. We used the code example from the great [Occiglot Models](https://huggingface.co/DiscoResearch/Llama3-German-8B-32k#document-packing) and optimized it a bit.

The Best-Fit packing strategy can be used by setting `packing_strategy` to `best-fit`. We did not experiment much with this packing strategy because the creation of TFRecords takes a long time!
