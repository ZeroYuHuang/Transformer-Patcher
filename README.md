# Transformer-Patcher

This repo contains the code and data of ICLR 2023 accepted paper:

Transformer-Patcher: One mistake worth one neuron. Zeyu Huang, Yikang Shen, Xiaofeng Zhang, Jie Zhou, Wenge Rong, Zhang Xiong.

## Steps for running Transformer-Patcher

## Directory Structure

```
data/ --> data folder including data of FEVER and zsRE
scripts/ --> folder containing main.py to run groups of experiments
src/ --> folder containing classes used for loading data and constructing model
requirements.txt --> contains required packages
```

### Setup

#### Environment

Our codes are based on Python 3.8.10. Other versions may work as well.

Create a virtual environment and install the dependencies ([conda](https://www.anaconda.com/) can help you do this) :

```
$ conda create --name T-Patcher python=3.8.10
$ conda activate T-Patcher
(T-Patcher)$ pip install -r requirements.txt
```

### Data

The data used is already in **data.zip** file, please unzip this file and implement the following pre-processing steps:

```
(T-Patcher)$ python src/dataset/fever_dataloader.py
(T-Patcher)$ python src/dataset/zsre_dataloader.py
```

Afterwards the original data is randomly split and we obtain *a training set*, *a validation set*, *a edit set* and  *a test set*.

## Running the code

### Training initial models

Before conducting Sequential Model Editing, we first need an initial model.

For fine-tuning a BERT base model on FEVER dataset, you can run:

```
(T-Patcher)$ python scripts/train_bert_fever.py
```

For fine-tuning a BART base model on zsRE dataset, you can run:

```
(T-Patcher)$ python scripts/train_bart_seq2seq.py
```

Then the initial model is trained and its checkpoint is saved in `log/models/bert_binary/version_0/checkpoints` or `log/models/bart_seq2seq/version_0/checkpoints` 

### Running Transformer-Patcher

Running Transformer-Patcher requires several arguments:

- `--task` : fever or zsre, depending on which task you would like to run
- `--edit_folder_num`: means how many folders the edit set $D_{edit}$ is randomly split into, ($n$ in our paper)
- `--process_folders` : two choices, (1) 'all_folders' means you utilize all edit folder in parallel (2) a list containing the folder number you want to process, such as [1,3,5,7]. We recognize you pass [0] to run folder 0 tentatively.   
- `--model_path`: the path of the initial model checkpoint
- `--task_id` all relevant files will be saved in `log/T-patch/$TASK/$TASK_ID` , the metrics are saved in `res.pkl`

```
(T-Patcher)$ python scripts/main.py --task=$TASK --edit_folder_num=$EDIT_FOLDER_NUM --process_folders=$PROCESS_FOLDERS --model_path=$MODEL_PATH
```