
## precision@k 

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

## Shuffle (sample, label) paris
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
