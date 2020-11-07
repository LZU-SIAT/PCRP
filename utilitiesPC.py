
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import time
import math
#plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import os


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    lr_decay_epochs = opt[0]
    learning_rate = opt[1]
    lr_decay_rate = opt[2]
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    if steps > 0:
        new_lr = learning_rate * (lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

def reverse_sequence(data, len_list):
    new_seq = torch.zeros_like(data)
    for i, lens in enumerate(len_list):
        time_range_order = [j for j in range(lens)]
        time_range_reverse = list(reversed(time_range_order))
        new_seq[i, time_range_order, :] = data[i, time_range_reverse, :]
    return new_seq


def exist_dir(path):
    
    if not os.path.exists(path):
        os.makedirs(path)


class LinearClassifier(nn.Module):
    def __init__(self,  last_layer_dim = None, n_label=None,  ):
        super(LinearClassifier, self).__init__()

        self.classifier = nn.Linear(last_layer_dim, n_label)
        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.classifier(x)



def knn(data_train, data_test, label_train, label_test, nn=1):
    label_train = np.asarray(label_train)
    label_test = np.asarray(label_test)

    Xtr_Norm = preprocessing.normalize(data_train)
    Xte_Norm = preprocessing.normalize(data_test)

    knn = KNeighborsClassifier(n_neighbors=nn,
                               metric='cosine')  # , metric='cosine'#'mahalanobis', metric_params={'V': np.cov(data_train)})
    knn.fit(Xtr_Norm, label_train)
    pred = knn.predict(Xte_Norm)
    acc = accuracy_score(pred, label_test)
    return acc

def save_checkpoint(model, epoch, optimizer, loss, PATH):
    torch.save({
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'loss': loss,
    }, PATH)


def save_checkpoint_auto(model,  PATH):
    torch.save({
         'model_state_dict': model.state_dict(),
       }, PATH)
    

def load_checkpoint(model, optimizer, PATH):
    data = torch.load(PATH)
    model.load_state_dict(data['model_state_dict'])
    optimizer.load_state_dict(data['optimizer_state_dict'])
    return data['epoch'], data['loss']

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))