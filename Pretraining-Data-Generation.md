# Pretraining Data Generation

This section covers the modification of the original script that is used for generating the pretraining data (outcome are TFRecords).

We use the `create_pretraining_data.py` [implementation](https://github.com/tensorflow/models/blob/master/official/nlp/data/create_pretraining_data.py) from TF Model Garden repo, which goes back to the very original implementation taken from the [BERT repo](https://github.com/google-research/bert/blob/master/create_pretraining_data.py).

The following modifications are performed:

* Code style: mainly four spaces instead of two, `black` was used for format it.
* HF Tokenizer is used instead of using an own TF Model Garden tokenization logic.
* There's no need to pass a `vocab.txt` file, because a HF Model Hub identifier can be used
* In addition to the original BERT packing two new packing methods can be used: RoBERTa-style (`FULL-SENTENCES`) packing and Best-Fit packing

Our modified version can be found [here](create_pretraining_data.py).

# Show-Case

The RoBERTa-style packing strategy uses the whole input corpus can chunks it into a fixed sequence length (typical 512). The `[CLS]` and `[SEP]` special tokens are automatically prepended/appended. This strategy allows to pretraing BERT models with Token Dropping approach.

To use this packing strategy, the `packing_strategy` parameter needs to be set to `roberta`.

Additionally, the Best-Fit packing is an alternative approach and was presented in the [Fewer Truncations Improve Language Modeling](https://arxiv.org/abs/2404.10830) paper. We used the code example from the great [Occiglot Models](https://huggingface.co/DiscoResearch/Llama3-German-8B-32k#document-packing) and optimized them a bit.

The Best-Fit packing strategy can be used by setting `packing_strategy` to `best-fit`. We did not experiment much which this packing strategy, because the creation of TFRecords will take a long time!
