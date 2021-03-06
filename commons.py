from glob import glob
import numpy as np
import torch
import cv2
from torchvision import transforms
from IPython.display import clear_output
from matplotlib import pyplot as plt
import copy
from sklearn.metrics import roc_auc_score
from tqdm.notebook import tqdm


BASE_PATH = '/data/avgarkavyj/idao_dataset'
TRAIN_PATH = BASE_PATH + '/train'
PUBLIC_TEST_PATH = BASE_PATH + '/public_test'
PRIVATE_TEST_PATH = BASE_PATH + '/private_test'


def read_dataset(path, target, X, y_1, y_2):
    fnames = glob(path + '/*')
    for fname in fnames:
        assert fname[-4:] == '.png', 'folder contains not a png file'

        energy = fname[:fname.index('_keV')][-2:]
        if energy[0] == '_':
            energy = energy[1]
        energy = float(energy)

        X.append(fname)
        y_1.append(target)
        y_2.append(energy)
    return X, y_1, y_2


def load_images(fnames):
    assert cv2.imread(fnames[0]).shape == (576, 576, 3), 'shape is not equal for all images?'
    X = torch.Tensor([cv2.imread(fname).T for fname in fnames]) / 255
    X = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(X)
    return X


def get_batch(X_batch, y_batch, device, classification=False):
    inp = load_images(X_batch)
    if classification:
        out = torch.LongTensor(y_batch)
    else:
        out = torch.Tensor(y_batch).reshape(-1, 1)
    return inp.to(device), out.to(device)


def sample_batch(X, y, batch_size, device, classification=False):
    batch_ix = np.random.randint(len(X), size=batch_size)
    return get_batch(X[batch_ix], y[batch_ix], device, classification=classification)


def calc_mae(model, X_test, y_test, device, batch_size):
    mae = 0
    for i in range(0, len(X_test), batch_size):
        inp, out = get_batch(X_test[i:i+batch_size], y_test[i:i+batch_size], device)
        mae += np.sum(np.abs(model(inp).detach().cpu().numpy() - out.detach().cpu().numpy()))
    return mae / len(X_test)


def calc_auc(model, X_test, y_test, device, batch_size):
    pred_labels = []
    for i in range(0, len(X_test), batch_size):
        inp, _ = get_batch(X_test[i:i+batch_size], y_test[i:i+batch_size], device, True)
        pred = model(inp)
        pred_labels += list(pred.argmax(dim=-1).detach().cpu())
    return roc_auc_score(y_test, pred_labels)


def is_better(x, y, metric):
    if metric == 'MAE':
        return x < y
    assert metric == 'AUC', 'only MAE and AUC metric are supported'
    return x > y

def train_model(model, optimizer, criterion, X_train, X_test, y_train, y_test, T,
                batch_size, n_steps, metric, device, best_metric_init, checkpoint_filename):
    test_metric = 'test_' + metric
    metrics = {'train_loss': [], test_metric: [] }

    best_value = best_metric_init
    best_model_wts = None
    for step in range(n_steps):
        inp, out = sample_batch(X_train, y_train, batch_size, device, metric == 'AUC')

        model.train()
        optimizer.zero_grad()

        pred = model(inp)
        loss = criterion(pred, out)

        loss.backward()
        optimizer.step()

        metrics['train_loss'].append((step, loss.item()))
        if step % T == 0:
            # calc metrics using test data
            if step == 0:
                metrics[test_metric].append((step, best_value))
            else:
                with torch.no_grad():
                    model.eval()

                    if metric == 'MAE':
                        value = calc_mae(model, X_test, y_test, device, batch_size)   
                    else:
                        assert metric == 'AUC', 'only MAE and AUC metric are supported'
                        value = calc_auc(model, X_test, y_test, device, batch_size)
                    metrics[test_metric].append((step, value))
                    if is_better(value, best_value, metric):
                        best_value = value
                        best_model_wts = copy.deepcopy(model.state_dict())
                        torch.save(best_model_wts, checkpoint_filename + '_' + metric + '_' + str(value))

        clear_output(wait=True)
        plt.figure(figsize=(12, 4))
        for i, (name, history) in enumerate(sorted(metrics.items())):
            plt.subplot(1, len(metrics), i + 1)
            plt.title(name)
            plt.plot(*zip(*history))
            plt.grid()
        plt.show()

        print("Mean loss=%.3f" % np.mean(metrics['train_loss'][-10:], axis=0)[1], flush=True)
        print("Max %s*1000=%.3f" % (metric, np.max(metrics[test_metric], axis=0)[1] * 1000), flush=True) 
    model.load_state_dict(best_model_wts)


def get_predictions(model, path, batch_size, device, classification=False):
    predict_files = glob(path + '/*')
    for fname in predict_files:
        id_ = fname.split('/')[-1][:-4]

    preds = []
    for i in tqdm(range(0, len(predict_files), batch_size)):
        with torch.no_grad():
            model.eval()
            inp, _ = get_batch(predict_files[i:i+batch_size], [0], device)
            pred = model(inp)
            if classification:
                preds += list(pred.argmax(dim=-1).detach().cpu())
            else:
                preds += list(pred.detach().cpu().flatten())
    return [x.item() for x in preds]
