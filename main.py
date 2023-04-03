import os
import pickle
from typing import List

import scipy
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold

import numpy as np
import pandas as pd

from kmeans_19 import set_plot_params, learn_hist_by_kmeans
from utils.spcup_utils import conjection, jitter, mix_up, ret_triangle_matrix, fnc_vec_to_sym_matrix
import timecorr.timecorr as tc
# from sktime.sktime.classification.hybrid import HIVECOTEV2


# hive = HIVECOTEV2()


def test_models(model_list: List[object], X_train, y_train, X_test, subjects_test,
                save_test_predictions: bool = False):
    chosen_model = None
    best_score = 0
    for model in model_list:
        scores = []
        k_fold = KFold(n_splits=5, random_state=30, shuffle=True)
        for train_idx, val_idx in k_fold.split(X_train):
            X_train_part, X_val, y_train_part, y_val = X_train[train_idx], X_train[val_idx], \
                                             y_train[train_idx], y_train[val_idx]

            model.fit(X_train_part, y_train_part)
            scores.append(np.count_nonzero(model.predict(X_val) == y_val) / len(y_val))
        score = np.array(scores).mean()
        print(f'model {model} average accuracy is = {score * 100}%')
        if best_score < score:
            chosen_model = model
            best_score = score
        if save_test_predictions:
            model.fit(X_train, y_train)
            test_prediction = chosen_model.predict_proba(X_test)[:, 1]
            results = pd.DataFrame()
            results['ID'] = subjects_test
            results['Predicted'] = test_prediction
            results = results.groupby(['ID']).mean()

            # Save results:
            file_name = 'results.csv'
            results_file = f'results/{model}_{file_name}'
            results.to_csv(results_file)
    print(f'Best model is: {chosen_model}')


def add_mix_up(X_train_part, y_train_part):
    a, b = [], []
    for i in range(len(y_train_part) - 1)[::30]:
        new_X, new_y = mix_up((X_train_part[i], y_train_part[i]), (X_train_part[i + 1], y_train_part[i + 1]))
        a.append(new_X)
        b.append(np.argmax(new_y))
    X_train_part = np.concatenate([a, X_train_part], axis=0)
    y_train_part = np.concatenate([np.stack(b).squeeze(), y_train_part], axis=0)
    return X_train_part, y_train_part


def create_dynamic_fncs(dir_path: str = 'data/train', file_name: str = 'icn_tc.npy', label_name: str = 'BP',
                        step_size: int = 4, width: int = 10):
    gaussian = {'name': 'Gaussian', 'weights': tc.gaussian_weights, 'params': {'var': width}}
    riemann_dataset = []
    new_train_labels = []
    if len(label_name) > 0:
        walk = os.walk(f'{dir_path}/{label_name}')
    else:
        walk = os.walk(f'{dir_path}')
    for root, dirs, files in walk:
        for file in files:
            if file.endswith(file_name):
                data = np.load(os.path.join(root, file))
                label = 0 if label_name == 'SZ' else 1
                vec_corrs = tc.timecorr(data, weights_function=gaussian['weights'],
                                        weights_params=gaussian['params'])[::step_size, :]
                mat_corrs = tc.vec2mat(vec_corrs)
                riemann_mat = ret_triangle_matrix(conjection(mat_corrs))
                riemann_dataset.append(riemann_mat)
                if label_name != '':
                    new_train_labels.append(np.repeat(label, mat_corrs.shape[2]))
                np.save(f'{root}/dFNC.npy', riemann_mat)
                print(f'finished subject num: {root[-6:]}')

    new_train_labels = np.concatenate(new_train_labels, axis=0)
    riemann_dataset = np.concatenate(riemann_dataset, axis=0)
    return ret_triangle_matrix(riemann_dataset), new_train_labels


def get_data(fnc_train_path: str = 'data/fnc_full_train_dataset.npy',
             fnc_test_path: str = 'data/fnc_test.npy',
             test_subjects_path: str = 'data/test_subjects.npy',
             icn_train_path: str = "data/ICN_train_dataset.pkl",
             icn_test_path: str = "data/ICN_test_dataset.pkl",
             mix_up_aug: bool = False, dynamic_fnc_aug: bool = False, gaussian_noise_aug: bool = False):

    dataset = np.load(fnc_train_path)
    X_train, y_train = dataset[:, :-1], dataset[:, -1]
    X_test = np.load(fnc_test_path).T
    subjects_test = np.load(test_subjects_path)

    open_file = open(icn_train_path, "rb")
    icn_train_x = pickle.load(open_file)
    open_file.close()
    icn_y = y_train

    open_file = open(icn_test_path, "rb")
    icn_test_x = pickle.load(open_file)
    open_file.close()

    # Augmentations:
    if dynamic_fnc_aug:
        X_train_bp, y_train_bp = create_dynamic_fncs()
        X_train_sz, y_train_sz = create_dynamic_fncs(label_name='SZ')
        X_test, subjects_test = create_dynamic_fncs(dir_path='data/test', label_name='')
        print('finish dFNC creation')
        X_train = np.concatenate([X_train_bp, X_train_sz])
        y_train = np.concatenate([y_train_bp, y_train_sz])
    if mix_up_aug:
        X_train, y_train = add_mix_up(X_train, y_train)
    if gaussian_noise_aug:
        X_train = jitter(X_train)

    return X_train, y_train, X_test, subjects_test


if __name__ == "__main__":
    X_train, y_train, X_test, subjects_test = get_data(dynamic_fnc_aug=True)
    # np.save('data/dfnc_train_data.npy', X_train)
    # np.save('data/dfnc_train_labels.npy', y_train)

    k = 50
    window_width = 20
    step_size = 5
    # possible values for algo: {“lloyd”, “elkan”, “auto”, “full”}
    algo = 'full'
    set_plot_params()
    X_train = learn_hist_by_kmeans(k, window_width, step_size, algo)
    X_test = learn_hist_by_kmeans(k, window_width, step_size, algo, dir_path=r'data/test')

    svm_model_rbf = svm.SVC(kernel='rbf', probability=True, C=1.0, class_weight='balanced')
    svm_model_poly = svm.SVC(kernel='poly', probability=True, C=1.0, class_weight='balanced')
    svm_model_lin = svm.SVC(kernel='linear', probability=True, C=1.0, class_weight='balanced')
    svm_model_sig = svm.SVC(kernel='sigmoid', probability=True, C=1.0, class_weight='balanced')
    xgboost = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0)
    mlp_model = MLPClassifier(random_state=0, max_iter=800)

    models = [svm_model_rbf, mlp_model]
    print('Start models')
    test_models(models, X_train, y_train, X_test, subjects_test, save_test_predictions=False)

