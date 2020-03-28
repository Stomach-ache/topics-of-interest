## Table of Contents<a name="top"></a>
[Precision@k](#precision)<br>
[Data Shuffling](#shuffle)<br>
[Conda Enviroment](#condaenv)<br>
[Fix Random Seed](#randseed)<br>
[LaTeX Tabular to DataFrame](#table2df)<br>


--------------


## precision@k  (*Pytorch*)<a name="precision"></a>

```python
from __future__ import print_function, absolute_import

__all__ = ['accuracy']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k in Pytorch"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
```

## Shuffle (sample, label) paris<a name="shuffle"></a>
```python
from sklearn.utils import shuffle
train_data, train_labels = shuffle(train_data, train_labels)
```


## Create mini-batch
```python
class Batch(object):
  def __init__(self, X, y, batch_size):
    self.batch_size = batch_size
    self.X = X
    self.y = y
    self.size = X.shape[0]
  def getBatch(self):
    indices = np.random.choice(range(self.size), self.batch_size)
    return self.X[indices], self.y[indices]

x_train = x_train.reshape([-1, 28, 28, 1])

batch_size = 512
batch = Batch(x_train, y_train, batch_size)
```

## Anaconda create/remove environment<a name="condaenv"></a>
[Tutorial](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/)

## Fix random seeds (for reproducibility)<a name="randseed"></a>
[Pytorch Notes](https://pytorch.org/docs/stable/notes/randomness.html)

## Conversion between LaTeX Tabular and Pandas DataFrame<a name="table2df"></a>
```python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re

def tabular2df(filename):
    with open(filename, 'r') as fp:
        table = fp.read().splitlines()

    df = []
    for line in table:
        if line.startswith('\\'):
            continue
        if line[-1] == '\\':
            line = line[:-2]
        line = line.split('&')
        for i, cell in enumerate(line):
            if i == 0:
                continue
            regex = re.search(r"\d", cell)
            if regex == None:
                continue
            else:
                line[i] = cell[regex.start():]
        df.append(line)

    df = np.array(df)
    columns = df[0, 1:]
    index = df[1:, 0]
    data = df[1:, 1:]
    df = pd.DataFrame(data=data, columns=columns, index=index)
    return df

def calc_average_and_rank(df, column_wise=True):
    if column_wise == False:
        df = df.T

    avg = []
    for col in df.columns:
        data = df[col]
        if isinstance(data[0], str):
            data = list(map(float, data))
        avg.append(np.mean(data))
    avg = np.array(avg)
    tmp = np.argsort(avg)[::-1]
    rnk = np.arange(len(avg))
    for i, v in enumerate(tmp):
        rnk[v] = i + 1

    return avg, rnk

def highlight_max(df):
    # row-wise max
    for i in range(df.shape[0]):
        row = df.iloc[i]
        if isinstance(row[0], str):
            row = list(map(float, row))
        pos = np.argmax(row)
        df.iloc[i, pos] = '\\bf' + df.iloc[i, pos]
    return df

def df2tabular(df):
    return df.to_latex(float_format="%.4f", escape=False)
```
[back to top](#top)
