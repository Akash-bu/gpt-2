# Build GPT-2

I cloned the repo from Stanford CS 224N and implemented a custom gpt-2 model and used it for classifcation as a downstream task. 

This project comprises two parts. In the first part, you will implement some important components of the GPT-2 model to
better understand its architecture.
In the second part, you will use the token embeddings produced by your GPT-2 model on two downstream tasks: paraphrase
detection and sonnet generation. You will implement extensions to improve your model's performance on these tasks.

In broad strokes, Part 1 of this project targets:

* modules/attention.py: Missing code blocks.
* modules/gpt2_layer.py: Missing code blocks.
* models/gpt2.py: Missing code blocks.
* classifier.py: Missing code blocks.
* optimizer.py: Missing code blocks.

To test Part 1, you will run:

* `optimizer_test.py`: To test your implementation of `optimizer.py`.
* `sanity_check.py`: To test your implementation of GPT models.
* `classifier.py` : To perform sentiment classification using your models.

In Part 2 of this project, you will use GPT2 (via cloze-style classification) detect if one sentence is a paraphrase of 
another as well as generate sonnets via autoregressive language modeling.  

To test Part 2, you will run:

* `paraphrase_detection.py`: To perform paraphrase detection. 
* `sonnet_generation.py`: To perform sonnet generation.

Important: Adjust training hyperparameters, particularly batch size, according to your GPU's specifications to optimize performance and prevent out-of-memory errors.

## Pre-testing instructions

While there are missing code blocks that you need to implement in both of these files, the main focus of this second 
part are the extensions: how you modify your GPT2 model to improve its ability to determine if one sentence is a 
paraphrase of another as well as its ability to generate sonnets. 

## Setup instructions

Follow `setup.sh` to properly setup a conda environment and install dependencies.

## Acknowledgement

This project is adapted from a prior year's CS 224N
project [Implement BERT](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/project/default-final-project-handout-minbert-spr2024-updated.pdf)
.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers)
library ([Apache License 2.0](./LICENSE)).

## Sentiment-classification results

Default configuration: `last-linear-layer`, batch size `8`, learning rate `1e-3`, hidden dropout `0.3`, and `10` epochs.

| Dataset | Best epoch | Train loss | Train accuracy | Dev accuracy | Final checkpoint dev accuracy |
|---|---:|---:|---:|---:|---:|
| SST | 7 | 1.416 | 0.494 | 0.466 | 0.466 |
| CFIMDB | 9 | 0.456 | 0.872 | 0.890 | 0.890 |

Epoch numbers are reported exactly as logged by the training script (zero-based). The final evaluation loaded the best saved checkpoint for each dataset.

## Full-model fine-tuning

Configuration: full-model fine-tuning, batch size `8`, learning rate `1e-5`, hidden dropout `0.3`, and `10` epochs.

| Dataset | Best epoch | Train loss | Train accuracy | Dev accuracy | Final checkpoint dev accuracy |
|---|---:|---:|---:|---:|---:|
| SST | 6 | 1.084 | 0.587 | 0.502 | 0.502 |
| CFIMDB | 7 | 0.023 | 0.998 | 0.971 | 0.971 |

## Full-model fine-tuning with split learning rates

Configuration: full-model fine-tuning, SST batch size `16`, CFIMDB batch size `8`, GPT learning rate `1e-5`, classifier learning rate `1e-3`, hidden dropout `0.1`, and `10` epochs.

| Dataset | Best dev accuracy | Final checkpoint dev accuracy |
|---|---:|---:|
| SST | 0.510 | 0.510 |
| CFIMDB | 0.984 | 0.984 |
