# tmux # Do not forget this!!!
sudo apt update

git clone https://github.com/tensorflow/models.git
cd models/
git checkout v2.16.0
export PYTHONPATH=$(pwd)
export PATH=$PATH:/home/$USER/.local/bin

sudo apt-get install -y libgl1

pip3 install gin-config tensorflow_datasets scipy sentencepiece tensorflow_hub scikit-learn seqeval sacrebleu immutabledict pycocotools opencv-python

cd official/projects/teams/

python3 train.py --help
