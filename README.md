# Learn to Make First Contact
Modelling chit-chat between two newly met persons.

## Install Dependencies

### Python Environment

First of all, you should setup a python environment. This code base has been tested under python 3.x, and we officially support python 3.7.

After installing python 3.7, we strongly recommend you to use `virtualenv` (a tool to create isolated Python environments) to manage the python environment. You could use following commands to create a environment.

```bash
python -m pip install virtualenv
virtualenv venv
```

### Activate Virtual Environment
Then you should activate the environment to install the dependencies. You could achieve it via using the command as below. (Please change $ENV_FOLDER to your own virtualenv folder path, e.g. venv)

```bash
$ENV_FOLDER\Scripts\activate.bat (Windows)
source $ENV_FOLDER/bin/activate (Linux)
```

### Install PyTorch

The only requirement of our code base is as following:

- pytorch >= 1.0.1


### Install Custom Dependencies

Besides pytorch, our code is mainly based on [ParlAI](https://github.com/facebookresearch/ParlAI) and [Huggingface's transformers](https://github.com/huggingface/transformers) (pytorch-pretrained-bert v0.6.2) library. As they are under active development, for the purpose to reproduce our results, we provide two custom repos to install them. It is worth noting that we also modify a little on Huggingface's code to achieve the auxiliary task `Next Utterance Prediction` (See Section 3.1 in our paper), and more details on changes could be seen [here](https://github.com/SivilTaram/transformers/commit/e1e718496c32c0d99291c0b890fd4ae6365191ba). 

```bash
git clone https://github.com/SivilTaram/transformers.git
cd transformers
python setup.py install
cd ..
git clone https://github.com/SivilTaram/ParlAI.git
cd ParlAI
python setup.py install
```

## Training

We provide three files to train `Transmitter`, `Receiver` and `PSquare` (details can be found in our paper). And the corresponding training scripts and commands are as below.

### Training Transmitter

The transmitter is based OpenAI's GPT model. We provide the default hyper-parameter for our paper experiments. Therefore, you could use the following command to train a transmitter. The script will automatically download the ConvAI2 dataset into the `./data/` folder. 

```python
python train_transmitter.py
```

### Training Receiver

If you have downloaded the ConvAI2 dataset, you could use `./tasks/convai2receiver/build_data.py` to build the dataset for receiver:

```python
cd tasks/convai2receiver
python build_data.py
```

Then you could train the Receiver model as:

```python
python train_receiver.py
```

### Training PSquare

Before training PSquare, you should have a trained transmitter and receiver. Specifying the model names in line 33-42 in `train_psquare.py`, you can run the following script to execute the self-play procedure.

```python
python train_psquare.py
```

## Trained Model Weights

We also provide trained PSquare weights for reproducing our experimental results in the paper.

- Trained model weights under the Original setting: https://www.dropbox.com/s/ozw9xmfv4f0tud9/psqaure_original.zip?dl=0
- Trained model weights under the Revised setting: https://www.dropbox.com/s/bbvamaj9r019wsw/psqaure_revised.zip?dl=0

## Evaluation

You could run `eval_f1.py`, `eval_hits.py` to obtain the `F1`, `Hits@1` for either Transmitter or PSquare. The evaluation logs on our provided model weights can be found in the folder `./logs/`.

As for the `ppl` metric, you could run the training script on a trained model file to fake the continuation of training. The restoring will first validate on the valid dataset.

## Acknowledgement

The `parlai` module is modified from [ParlAI](https://github.com/facebookresearch/ParlAI). Thanks them for their huge contributions on developing such a great conversational platform!

## Citation

Please consider citing our paper if it is helpful to you :)

```bib
@inproceedings{liu-etal-2020-personachat,
    title = "You Impress Me: Dialogue Generation via Mutual Persona Perception",
    author = "Liu, Qian  and
      Chen, Yihong  and
      Chen, Bei  and
      Lou, Jian-Guang  and
      Chen, Zixuan  and
      Zhou, Bin  and
      Zhang, Dongmei",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = july,
    year = "2020",
    publisher = "Association for Computational Linguistics"
}
```

