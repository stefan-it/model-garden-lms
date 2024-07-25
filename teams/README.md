# TEAMS Pretraining

We use the same pretraining data as used for [BERT Pretraining](../bert/README.md). This is great because we can reuse the created TFRecords, which also allows us to compare BERT and TEAMS directly, eliminating potential differences in pretraining data generation.

# Start Pretraining

We need to navigate to `models/official/projects/teams` and create a YAML-based configuration file under `experiments/base/fineweb-10BT-teams.yaml`:

https://github.com/stefan-it/model-garden-lms/blob/913e8025d7ab438fffc17aeed006ca34af23a448/teams/experiments/base/fineweb-10BT-teams.yaml#L1-L67

Then the pretraining can be started with:

```bash
$ python3 train.py --experiment=teams/pretraining \
--config_file=experiments/base/fineweb-10BT-teams.yaml \
--params_override="runtime.distribution_strategy=tpu" \
--tpu=fineweb \
--model_dir=gs://fineweb-lms/models/fineweb-10BT-teams \
--mode=train
```
