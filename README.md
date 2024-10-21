# SELIN (**S**elf-**E**xplaining deep models with **LIN**ear weight)

This repository is the official implementation of "Toward Faithful and Human-Aligned Self-Explanation of Deep Models".

## Training
Train the base model with the following command.
```
python3 base.py --dataset <DATASET> --base <BASE_MODEL> --gpu <GPU>
```

Extract the embedding of base model for train instances with the following command.
```
python3 extract_base_embedding.py --dataset <DATASET> --base <BASE_MODEL> --gpu <GPU>
```

Build the atom pool with the following command.
```
python3 build_atom_pool.py --dataset <DATASET>
```

Train the SELIN model with the following command.
```
python3 selin.py --dataset <DATASET> --base <BASE_MODEL> --gpu <GPU>
```

## Evaluation
Evaluation results are automatically provided after training.
If you want to produce only evaluation result of a trained model, please use the following command.
```
python3 selin.py --dataset <DATASET> --base <BASE_MODEL> --gpu <GPU> --only_eval
```
