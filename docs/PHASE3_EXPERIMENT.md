# Phase 3 bounded S1/S2 fusion experiment

This is a local research experiment, not an official BigEarthNet benchmark.
The split contains 428 training, 211 validation, and 161 test records grouped
by connected Sentinel-1 and Sentinel-2 acquisition scenes.

The final experiment uses seed `20261013`, batch size 8, AdamW learning rate
`3e-4`, weight decay `1e-4`, class-wise `pos_weight` computed from the training
split, and `BCEWithLogitsLoss`. Three full epochs completed on CPU (162 train
batches and 81 validation batches). The checkpoint selection criterion is
validation macro average precision at threshold-independent sigmoid scores;
the test split was not used for selection.

The selected checkpoint is `results/fusion-phase3-final.pt`; its last-epoch
counterpart is `results/fusion-phase3-final-last.pt`. The saved JSON report
contains all batch counts, support, positive rates, gradient norms, losses,
architecture, environment, and history. The fusion result remains below the
pretrained ResNet-50 and is displayed as `trained limited` in the dashboard.
