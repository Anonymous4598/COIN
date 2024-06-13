import torch
import torch.nn as nn
import numpy as np
import numpy as np

from abc import ABC, abstractmethod
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score, normalized_mutual_info_score, adjusted_rand_score, precision_score, recall_score
from sklearn.model_selection import PredefinedSplit, GridSearchCV, StratifiedKFold
from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

import functools
import warnings
warnings.filterwarnings("ignore")

def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()

def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator

def get_split(num_samples: int, train_ratio: float = 0.16, test_ratio: float = 0.8):
    assert train_ratio + test_ratio <= 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'val': indices[train_size: test_size + train_size],
        'test': indices[test_size + train_size:]
    }


def from_predefined_split(data):
    assert all([mask is not None for mask in [data.train_mask, data.test_mask, data.val_mask]])
    num_samples = data.num_nodes
    indices = torch.arange(num_samples)
    return {
        'train': indices[data.train_mask],
        'val': indices[data.val_mask],
        'test': indices[data.test_mask]
    }

def split_to_numpy(x, y, split):
    keys = ['train', 'test', 'val']
    objs = [x, y]
    return [obj[split[key]].detach().cpu().numpy() for obj in objs for key in keys]


def get_predefined_split(x_train, x_val, y_train, y_val, return_array=True):
    test_fold = np.concatenate([-np.ones_like(y_train), np.zeros_like(y_val)])
    ps = PredefinedSplit(test_fold)
    if return_array:
        x = np.concatenate([x_train, x_val], axis=0)
        y = np.concatenate([y_train, y_val], axis=0)
        return ps, [x, y]
    return ps


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, x=None, y=None, split=None, train_z=None, train_y=None, val_z=None, val_y=None, test_z=None, test_y=None) -> dict:
        pass

    def __call__(self, x=None, y=None, split=None, train_z=None, train_y=None, val_z=None, val_y=None, test_z=None, test_y=None) -> dict:
        if split is not None:
            for key in ['train', 'test', 'val']:
                assert key in split
        if train_z is not None:
            result = self.evaluate(x, y, split, train_z, train_y, val_z, val_y, test_z, test_y)
        else:
            result = self.evaluate(x, y, split)
        return result

class WiKiBaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, x=None, y=None, train_mask=None, val_mask=None, test_mask=None) -> dict:
        pass

    def __call__(self, x=None, y=None, train_mask=None, val_mask=None, test_mask=None) -> dict:
        result = self.evaluate(x, y, train_mask, val_mask, test_mask)
        return result



class SVMBaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, x, y, split) -> dict:
        pass

    def __call__(self, x, y, split) -> dict:
        for key in ['train', 'test', 'val']:
            assert key in split
        result = self.evaluate(x, y, split)
        return result

class BaseSVMSKLearnEvaluator(SVMBaseEvaluator):
    def __init__(self, evaluator, params):
        self.evaluator = evaluator
        self.params = params

    def evaluate(self, x, y, split):
        x_train, x_test, x_val, y_train, y_test, y_val = split_to_numpy(x, y, split)
        ps, [x_train, y_train] = get_predefined_split(x_train, x_val, y_train, y_val)
        classifier = GridSearchCV(self.evaluator, self.params, cv=ps, scoring='accuracy', verbose=0)
        classifier.fit(x_train, y_train)
        acc = accuracy_score(y_test, classifier.predict(x_test))
        test_macro = f1_score(y_test, classifier.predict(x_test), average='macro')
        test_micro = f1_score(y_test, classifier.predict(x_test), average='micro')

        return {
            'acc': acc,
            'micro_f1': test_micro,
            'macro_f1': test_macro,
        }

class BaseSKLearnEvaluator(BaseEvaluator):
    def __init__(self, evaluator, params):
        self.evaluator = evaluator
        self.params = params

    @repeat(3)
    def evaluate(self, x=None, y=None, split=None, train_z=None, train_y=None, val_z=None, val_y=None, test_z=None, test_y=None):
        x_train, x_test, x_val, y_train, y_test, y_val = split_to_numpy(x, y, split)
        ps, [x_train, y_train] = get_predefined_split(x_train, x_val, y_train, y_val)
        classifier = GridSearchCV(self.evaluator, self.params, cv=ps, scoring='accuracy', verbose=0)
        classifier.fit(x_train, y_train)
        acc = accuracy_score(y_test, classifier.predict(x_test))
        test_macro = f1_score(y_test, classifier.predict(x_test), average='macro')
        test_micro = f1_score(y_test, classifier.predict(x_test), average='micro')

        return {
            'acc': acc,
            'micro_f1': test_micro,
            'macro_f1': test_macro,
        }


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

def logistic_classify(x, y, device):

    nb_classes = np.unique(y).shape[0]
    xent = nn.CrossEntropyLoss()
    hid_units = x.shape[1]

    accs = []
    accs_val = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None) 
    for train_index, test_index in kf.split(x, y):

        train_embs, test_embs = x[train_index], x[test_index]
        train_lbls, test_lbls= y[train_index], y[test_index]

        train_embs, train_lbls = torch.from_numpy(train_embs).to(device), torch.from_numpy(train_lbls).to(device)
        test_embs, test_lbls= torch.from_numpy(test_embs).to(device), torch.from_numpy(test_lbls).to(device)


        log = LogReg(hid_units, nb_classes)
        log.to(device)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        best_val = 0
        test_acc = None
        for it in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc.item())

        val_size = len(test_index)
        test_index = np.random.choice(test_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        train_embs, test_embs = x[train_index], x[test_index]
        train_lbls, test_lbls= y[train_index], y[test_index]

        train_embs, train_lbls = torch.from_numpy(train_embs).to(device), torch.from_numpy(train_lbls).to(device)
        test_embs, test_lbls= torch.from_numpy(test_embs).to(device), torch.from_numpy(test_lbls).to(device)


        log = LogReg(hid_units, nb_classes)
        log.to(device)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        best_val = 0
        test_acc = None
        for it in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs_val.append(acc.item())

    return np.mean(accs_val), np.mean(accs)

def svc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    accuracies_std = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
        accuracies_std.append(100*accuracy_score(y_test, classifier.predict(x_test)))

        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))

    return np.mean(accuracies_val), np.mean(accuracies), np.std(accuracies_std)

def randomforest_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'n_estimators': [100, 200, 500, 1000]}
            classifier = GridSearchCV(RandomForestClassifier(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = RandomForestClassifier()
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

        val_size = len(test_index)
        test_index = np.random.choice(test_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'n_estimators': [100, 200, 500, 1000]}
            classifier = GridSearchCV(RandomForestClassifier(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = RandomForestClassifier()
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))

    ret = np.mean(accuracies)
    return np.mean(accuracies_val), ret

def linearsvc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = LinearSVC(C=10)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = LinearSVC(C=10)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))

    return np.mean(accuracies_val), np.mean(accuracies)

def link_prediction_evaluation(z, pos_edge_index, neg_edge_index, sigmoid=True):
    pos_y = z.new_ones(pos_edge_index.size(1))
    neg_y = z.new_zeros(neg_edge_index.size(1))

    edge_labels = torch.cat([pos_y, neg_y], dim=0)

    pos_pred = (z[pos_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
    pos_pred = torch.sigmoid(pos_pred) if sigmoid else pos_pred

    neg_pred = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
    neg_pred = torch.sigmoid(neg_pred) if sigmoid else neg_pred

    pred = torch.cat([pos_pred, neg_pred])
    edge_labels, pred = edge_labels.detach().cpu().numpy(), pred.detach().cpu().numpy()

    return roc_auc_score(edge_labels, pred), average_precision_score(edge_labels, pred)


def kf_evaluate_embedding(embeddings, labels, search=True):
    device = embeddings.device
    embeddings = embeddings.detach().cpu()
    labels = labels.detach().cpu()
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)

    acc = 0
    acc_val = 0

    
    _acc_val, _acc = logistic_classify(x, y, device)
    if _acc_val > acc_val:
        acc_val = _acc_val
        acc = _acc
    '''

    # _acc_val, _acc, _std = svc_classify(x,y, search)
    # if _acc_val > acc_val:
    #     acc_val = _acc_val
    #     acc = _acc
    '''
    """
    _acc_val, _acc = linearsvc_classify(x, y, search)
    if _acc_val > acc_val:
        acc_val = _acc_val
        acc = _acc
    """
    '''
    _acc_val, _acc = randomforest_classify(x, y, search)
    if _acc_val > acc_val:
        acc_val = _acc_val
        acc = _acc
    '''


    return acc_val, acc