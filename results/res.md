# Sentiment-classification results

Command: `python3 classifier.py --use_gpu`

Default configuration: `last-linear-layer`, batch size `8`, learning rate `1e-3`, hidden dropout `0.3`, and `10` epochs.

| Dataset | Best epoch | Train loss | Train accuracy | Dev accuracy | Final checkpoint dev accuracy |
|---|---:|---:|---:|---:|---:|
| SST | 7 | 1.416 | 0.494 | 0.466 | 0.466 |
| CFIMDB | 9 | 0.456 | 0.872 | 0.890 | 0.890 |

Epoch numbers are reported exactly as logged by the training script (zero-based). The final evaluation loaded the best saved checkpoint for each dataset.

## Full-model fine-tuning

Command: `python3 classifier.py --use_gpu --fine-tune-mode full-model --lr 1e-5`

Configuration: full-model fine-tuning, batch size `8`, learning rate `1e-5`, hidden dropout `0.3`, and `10` epochs.

| Dataset | Best epoch | Train loss | Train accuracy | Dev accuracy | Final checkpoint dev accuracy |
|---|---:|---:|---:|---:|---:|
| SST | 6 | 1.084 | 0.587 | 0.502 | 0.502 |
| CFIMDB | 7 | 0.023 | 0.998 | 0.971 | 0.971 |

## Full-model fine-tuning with split learning rates

Command: `python3 classifier.py --use_gpu --fine-tune-mode full-model --batch_size 16 --hidden_dropout_prob 0.1 --epochs 10`

Configuration: full-model fine-tuning, SST batch size `16`, CFIMDB batch size `8`, GPT learning rate `1e-5`, classifier learning rate `1e-3`, hidden dropout `0.1`, and `10` epochs.

| Dataset | Best dev accuracy | Final checkpoint dev accuracy |
|---|---:|---:|
| SST | 0.510 | 0.510 |
| CFIMDB | 0.984 | 0.984 |

## Paraphrase detection

Command: `python3 paraphrase_detection.py --use_gpu`

Configuration: full-model fine-tuning of GPT-2 small, batch size `8`, learning rate `1e-5`, and `10` epochs on the Quora paraphrase dataset.

| Dataset | Final checkpoint dev accuracy |
|---|---:|
| Quora | 0.897 |

The final evaluation loaded the best saved checkpoint, `10-1e-05-paraphrase.pt`.

## Sonnet generation

Command: `python -c "from evaluation import test_sonnet; print(test_sonnet())"`

The generated sonnets in `predictions/generated_sonnets.txt` were evaluated against `data/TRUE_sonnets_held_out.txt`.

| Metric | Score |
|---|---:|
| CHRF | 56.6372 |
