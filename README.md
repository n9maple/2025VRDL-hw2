# 2025VRDL-hw2
2025 Visual Recognition using Deep Learning hw2
Student id: 313551056
Name: 王力得

## Introduction
This repository implement a 100-class classification task with Resnet-152. It may achieve 92% accuracy on validation data and 95% accuracy on public test data.
## Installation
This project runs on python version 3.10.16 with cuda 12.4. It is recommended to use Conda to create a virtual environment.

1. clone the repository and enter the directory

2. Create a virtual environment with Conda:

```bash
conda create -n hw1 python=3.10.16
```

3. Activate the environment and download the packages:

```bash
conda activate hw1
pip install requirements.txt # Make sure you are using cuda 12.4
```

## Run the code
Please make sure you have download the dataset and the dataset are orginized. The directory should be like:
```bash
data
├── test
├── train
│   ├── 0
│   ├── 1
│   ├── 10
    ...
└── val
    ├── 0
    ├── 1
    ├── 10
    ...
```
1. Activate the environment

```bash
conda activate hw1
```

2. Run the training code (the arguments option may be seen by adding ```--help```)

```bash
python train.py
```

3. Run the inference code and give the path of the model weight (in default, it will be in *'save_model'* directory). If the path is not given, it will find *'save_model/model_epoch19.pth'*. You may download the weight from the link: https://drive.google.com/file/d/1jN02eioHV0dEsW-YuYEvqSWs3rjDGBEy/view?usp=sharing

```bash
python inference.py --model_path [your_model_path]
```

4. (optional) If you want to inference on validation data, run the following code:

```bash
python val_to_test.py
python inference.py --test_data_dir ./data/val_all --model_path [your_model_path] --validation
python val_accuracy.py
```

## Performance Snapshot 
<figure>
  <img src="images/1461.png">
</figure>

<figure>
  <img src="images/4239.png">
</figure>

<figure>
  <img src="images/4345.png">
</figure>

<figure>
  <img src="images/4450.png">
</figure>