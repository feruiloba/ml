# ml

To install python it's recommended to use [PyEnv](https://github.com/pyenv/pyenv)
Then install version `3.9.12`
```
pyenv install 3.9.12
```

To create new homework folder run
```
python -m venv hw1
```

You might need to run if you get the "pyenv: python: command not found" error
```
pyenv global 3.9.12
```

Before running any python file or installing any package, activate the virtual environment:
```
source hw1/bin/activate
```

Then you can add a package
```
pip install numpy==1.21.2
pip install matplotlib
```

## HW1

```
python majority_vote.py heart_train.tsv heart_test.tsv heart_train_labels.txt heart_test_labels.txt heart_metrics.txt
```

## HW2

```
python decision_tree.py heart_train.tsv heart_test.tsv 2 heart_2_train.txt heart_2_test.txt heart_2_metrics.txt heart_2_print.txt
```

## HW4

```
python feature.py smalldata/train_small.tsv smalldata/val_small.tsv smalldata/test_small.tsv glove_embeddings.txt formatted_train_small.tsv formatted_val_small.tsv formatted_test_small.tsv
python lr.py formatted_train_small.tsv formatted_val_small.tsv formatted_test_small.tsv train_labels_out_small.txt test_labels_out_small.txt metrics_small.txt 500 0.1
```

## HW5

```
python neuralnet.py small_train.csv small_validation.csv  small_train_out.labels small_validation_out.labels  small_metrics_out.txt 2 4 2 0.1
```

Also added .vscode settings to be able to debug and set breakpoints as well as test with VS Code extensions