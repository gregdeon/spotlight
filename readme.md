# The Spotlight: A General Method for Discovering Systematic Errors in Deep Learning Models

This repository is the official implementation of "The Spotlight: A General Method for Discovering Systematic Errors in Deep Learning Models" (under submission at NeurIPS 2021).
It includes:

- Training code for FairFace and X-ray classifiers
- Code for running the spotlight (inference passes and spotlight optimizer)
- Analysis notebooks used to visualize results in paper

## Requirements
### Packages

To install requirements for training and running spotlights:

```setup
pip install -r requirements.txt
```

For analysis notebooks, we used Singularity to run the [scipy-notebook](https://github.com/jupyter/docker-stacks/tree/master/scipy-notebook) Jupyter Docker stack.

### Datasets

Our experiments use the following datasets.
Set the environment variable `DATA_DIR` appropriately:

- `$DATA_DIR/fairface`: [FairFace](https://github.com/joojs/fairface), using the `padding=0.25` version of the dataset
- `$DATA_DIR/imagenet`: [ImageNet](https://pytorch.org/vision/stable/datasets.html#imagenet)
- `$DATA_DIR/amazon`: [Amazon Polarity](https://huggingface.co/datasets/amazon_polarity)
- `$DATA_DIR/squad`: [SQuAD](https://huggingface.co/datasets/squad)
- `$DATA_DIR/movielens`: [MovieLens 100k](https://github.com/jhartford/AutoEncSets), from Graham, Hartford et al.'s implementation of DeepSet
- `$DATA_DIR/xray`: [X-ray](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

## Training Scripts

For two of the domains in the paper, we train classifiers using standard architectures and training methods.
These scripts assume that `DATA_DIR` and `MODEL_DIR` have been set appropriately:

FairFace: 
```
python train_fairface.py --checkpoint_dir $MODEL_DIR/fairface
```

X-ray:
```
python train_xray.py
```

## Inference

We include inference scripts for each model, saving final-layer embeddings along with model outputs and losses:

- `inference_fairface.py` (FairFace)
- `inference_imagenet.py` (ImageNet)
- `inference_amazon.py` (Amazon Polarity)
- `inference_squad.py` (SQuAD)
- `inference_movielens.py` (MovieLens)
- `inference_xray.py` (X-ray)

## Spotlights

The spotlight is implemented as a command-line utility in `spotlight/run_spotlight.py`.
The specific commands that we ran in our experiments are listed in:

- `spotlights_fairface.sh` (FairFace)
- `spotlights_imagenet.sh` (ImageNet)
- `spotlights_amazon.sh` (Amazon Polarity)
- `spotlights_squad.sh` (SQuAD)
- `spotlights_movielens.sh` (MovieLens)
- `spotlights_xray.sh` (X-ray)

## Analysis

The results shown in our paper are produced by analyzing examples in each dataset that are given high weights by the spotlights. 
We include our spotlight weights in `spotlight_outputs/`, and Jupyter notebooks to visualize these results in `analysis.ipynb` and `analysis_nlp.ipynb` (for image/recommender systems and NLP models, respectively).
