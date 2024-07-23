# TPU VM Cheatsheet

This section shows how to setup TPU VMs to start LM pretraining with the TensorFlow Model Garden library.

# TPU VM Creation

We use the [queued resource](https://cloud.google.com/tpu/docs/queued-resources) mananger to create a TPU VM Pod (technically a single TPU VM such as v3-8 or v4-8 could also be used):

```bash
$ gcloud alpha compute tpus queued-resources create your-queued-resource-id \
  --node-id fineweb \
  --project $PROJECT_ID \
  --zone $ZONE \
  --accelerator-type v4-32 \
  --runtime-version tpu-vm-tf-2.16.1-pod-pjrt
```

Please adjust project id and tpu zone. We use TensorFlow 2.16.1 in this tutoral. However, TensorFlow versions 2.9.2 and 2.13.2 were also extensively tested.

You can check the creation status with:

```bash
$ gcloud alpha compute tpus queued-resources list --project $PROJECT_ID --zone $ZONE
```

# Installing TensorFlow Model Garden library with all dependencies

Once the TPU VM reached the status `ACTIVE` you can install all necessary dependencies on each of the TPU VM worker. Here's an example for installing it on first worker (with id `0`):

```bash
$ gcloud alpha compute tpus tpu-vm ssh fineweb --zone $ZONE --worker 0
```

Please start an interactive `tmux` session and install the following dependencies:

```bash
$ tmux # Do not forget this!!!
$ sudo apt update

$ git clone https://github.com/tensorflow/models.git
$ cd models/
$ git checkout v2.16.0
$ export PYTHONPATH=$(pwd)
$ export PATH=$PATH:/home/$USER/.local/bin

$ sudo apt-get install -y libgl1

$ pip3 install gin-config tensorflow_datasets scipy sentencepiece tensorflow_hub scikit-learn seqeval sacrebleu immutabledict pycocotools opencv-python

$ cd official/projects/teams/

$ python3 train.py --help
```

Some notes on that:

* We use TF Model Garden version `v2.16.0` in this example. We also tested `v2.9.2` and `v2.13.2`.
* The used TensorFlow version (specified in the TPU VM creation step) should match the TF Model Garden version! Otherwise there could be problems and errors.
