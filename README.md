# InsightFace-tensorflow

This is a tensorflow implementation of paper "[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)". This implementation aims at making both usage of pretrained model and training of your own model easier. Whether you just want to use pretrained model to do face recognition/verification or you want train/finetune your own model, this project can give you a favor. An introduction on face recognition losses can be found [here]()(in Chinese).

The implementation referred to [the official implementation in mxnet](https://github.com/deepinsight/insightface) and [the previous third-party implementation in tensorflow](https://github.com/auroua/InsightFace_TF).

- [InsightFace-tensorflow](#insightface-tensorflow)
  - [TODO List](#todo-list)
  - [Running Environment](#running-environment)
  - [Usage of Pretrained Model](#usage-of-pretrained-model)
    - [Pretrained Model](#pretrained-model)
    - [Model Evaluation](#model-evaluation)
    - [Extract Embedding with Pretrained Model](#extract-embedding-with-pretrained-model)
  - [Train Your Own Model](#train-your-own-model)
    - [Data Prepare](#data-prepare)
    - [Train with Softmax](#train-with-softmax)
    - [Finetune with Softmax](#finetune-with-softmax)

## TODO List

1. *Train with softmax [done!]*
2. *Model evaluation [done!]*
3. *Finetune with softmax [done!]*
4. *Get embedding with pretrained model [done!]*
5. **Train with triplet loss [todo]**
6. **Finetune with triplet loss [todo]**
7. Backbones    
   7.1 *ResNet [done!]*    
   7.2 **ResNeXt [todo]**    
   7.3 **DenseNet [todo]**    
8. Losses    
   8.1 *Arcface loss [done!]*    
   8.2 **Cosface loss [todo]**    
   8.3 **Sphereface loss [todo]**    
   8.4 **Triplet loss [todo]**
9.  **Face detection and alignment [todo]**

## Running Environment

- python 3.6 
- scipy, numpy (Anaconda 3 recommended)
- tensorflow 1.7.0
- mxnet 1.3.1 (only needed when reading mxrec file)

## Usage of Pretrained Model

Here we open our pretrained models for easier application of face recognition or verification. Codes on model evaluation and extracting embedding from face images are supplied.

### Pretrained Model

Pretrained models and their accuracies on validation datasets are shown as following:

|config|training steps|lfw|calfw|cplfw|agedb_30|cfp_ff|cfp_fp|vgg2_fp|download|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|[config_ms1m_100]()|k|%|%|%|%|%|%|%|[ms1m_100_k]()|
|[config_ms1m_200]()|k|%|%|%|%|%|%|%|[ms1m_200_k]()|

### Model Evaluation

You can evaluate a pretrained model with [evaluate.py]() by specifying the config path and model path, for example:

```
python evaluate.py 
--config_path=./configs/config_ms1m_100.yaml 
--model_path=$DIRECTORY_TO_PRETRAINED_MODEL$/best-m-150000
```

This will evaluate the pretrained model on validation datasets specified in the config file. If you want to evaluate the model on other validation dataset, you can specify it by --val_data as following:

```
python evaluate.py 
--config_path=./configs/config_ms1m_100.yaml 
--model_path=$DIRECTORY_TO_PRETRAINED_MODEL$/best-m-150000 
--val_data=$DIRECTORY_TO_VAL_DATA$/xxx.bin
```

### Extract Embedding with Pretrained Model

You can extract embedding from face images with [get_embd.py]() by the following script:

```
python get_embd.py 
--config_path=./configs/config_ms1m_100.yaml 
--model_path=$DIRECTORY_TO_PRETRAINED_MODEL$/best-m-150000 
--read_path=$PATH_TO_FACE_IMAGES$
--save_path=$SAVING_DIRECTORY$/embd.pkl
```

where config_path and model_path specify the config file and pretrained model respectively. read_path is path to face images, that can be a path to one image or a directory with only images in it. save_path specifies where to save the embedding. The saved file is a dict with image file name as key, the corresponding embedding as value, and can be loaded with pickle in python. Note that face images should be well cropped here.

## Train Your Own Model

If you want train your own model from scratch, or finetune pretrained model with your own data, here is what you should do.

### Data Prepare

The official InsightFace project open their training data in the [DataZoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo). This data is in mxrec format, you can transform it to tfrecord format with [./data/generateTFRecord.py]() by the following script:

```
python generateTFRecord.py 
--mode=mxrec
--image_size=112
--read_dir=$DIRECTORY_TO_THE_TRAINING_DATA$
--save_path=$DIRECTORY_TO_SAVE_TFRECORD_FILE$/xxx.tfrecord
```

Or, if you want to train the model with your own data, you can prepare the tfrecord file by the following script:

```
python generateTFRecord.py 
--mode=folders
--image_size=112
--read_dir=$DIRECTORY_TO_THE_TRAINING_DATA$
--save_path=$DIRECTORY_TO_SAVE_TFRECORD_FILE$/xxx.tfrecord
```

Here, the read_dir should be the directory to your own face images, where images to one person are saved in one folder. The directory should have a structure like this:

```
read_dir/
  - id1/
    -- id1_1.jpg
    ...
  - id2/
    -- id2_1.jpg
    ...
  - id3/
    -- id3_1.jpg
    -- id3_2.jpg
    ...
  ...
```

### Train with Softmax

To train your own model with softmax, firstly you should prepare a config file like those in [./configs](). It is recommended to modify one example config file to your own config. Secondly, the following script starts training:

```
python train_softmax.py --config_path=./configs/config_ms1m_100.yaml
```

### Finetune with Softmax

To finetune a pretrained model with your own data, you should prepare a finetune config file like [./configs/config_finetune.yaml](), and start training by the following script:

```
python finetune_softmax.py --config_path=./configs/config_finetune.yaml
```