# Sentiment-classification results

Command: `python3 classifier.py --use_gpu`

Default configuration: `last-linear-layer`, batch size `8`, learning rate `1e-3`, hidden dropout `0.3`, and `10` epochs.

| Dataset | Best epoch | Train loss | Train accuracy | Dev accuracy | Final checkpoint dev accuracy |
|---|---:|---:|---:|---:|---:|
| SST | 7 | 1.416 | 0.494 | 0.466 | 0.466 |
| CFIMDB | 9 | 0.456 | 0.872 | 0.890 | 0.890 |

Epoch numbers are reported exactly as logged by the training script (zero-based). The final evaluation loaded the best saved checkpoint for each dataset.
