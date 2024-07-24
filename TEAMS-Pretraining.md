# TEAMS Pretraining

We use the same pretraining data as used for [BERT Pretraining](BERT-Pretraining.md). This is great because we can reuse the created TFRecords, which also allows us to compare BERT and TEAMS directly, eliminating potential differences in pretraining data generation.

# Start Pretraining

We need to navigate to `models/official/projects/teams` and create a YAML-based configuration file under `experiments/base/fineweb-10BT-teams.yaml`:

```yaml
task:
  model:
    candidate_size: 5
    num_shared_generator_hidden_layers: 3
    num_discriminator_task_agnostic_layers: 11
    tie_embeddings: true
    generator:
      attention_dropout_rate: 0.1
      dropout_rate: 0.1
      embedding_size: 768
      hidden_activation: gelu
      hidden_size: 768
      initializer_range: 0.02
      intermediate_size: 3072
      max_position_embeddings: 512
      num_attention_heads: 12
      num_layers: 6
      type_vocab_size: 2
      vocab_size: 64000
    discriminator:
      attention_dropout_rate: 0.1
      dropout_rate: 0.1
      embedding_size: 768
      hidden_activation: gelu
      hidden_size: 768
      initializer_range: 0.02
      intermediate_size: 3072
      max_position_embeddings: 512
      num_attention_heads: 12
      num_layers: 12
      type_vocab_size: 2
      vocab_size: 64000
  train_data:
    drop_remainder: true
    global_batch_size: 256
    input_path: 'gs://fineweb-lms/tfrecords-op/*.tfrecord'
    is_training: true
    max_predictions_per_seq: 76
    seq_length: 512
    use_next_sentence_label: false
    use_position_id: false
    cycle_length: 8
    use_v2_feature_names: true
trainer:
  checkpoint_interval: 50000
  max_to_keep: 50
  optimizer_config:
    learning_rate:
      polynomial:
        cycle: false
        decay_steps: 1000000
        end_learning_rate: 0.0
        initial_learning_rate: 0.0002
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
  validation_interval: 100
  validation_steps: 64
```

Then the pretraining can be started with:

```bash
$ python3 train.py --experiment=teams/pretraining \
--config_file=experiments/base/fineweb-10BT-teams.yaml \
--params_override="runtime.distribution_strategy=tpu" \
--tpu=fineweb \
--model_dir=gs://fineweb-lms/models/fineweb-10BT-teams \
--mode=train
```
