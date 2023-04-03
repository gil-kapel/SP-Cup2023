import os
import numpy as np


def np_load_files(dir_path, file_name, aggr_func=lambda x: x):
    files = [np.load(os.path.join(root, file)).T
             for root, dirs, files in os.walk(dir_path)
             for file in files if(file.endswith(file_name))]
    try:
        files = np.stack(files)
    except ValueError:
        pass
    return files


def create_crop_data(disease_data):
    max_series = max([data.shape for data in disease_data])[1]
    min_series = min([data.shape for data in disease_data])[1]
    crop_new_data = []
    for data in disease_data:
        crop_new_data.append(data[:, :min_series])
        crop_new_data.append(data[:, -min_series:])
        for j in range(2):
            idx = np.random.randint(0, max_series-min_series)
            if min_series+idx > data.shape[1]:
                continue
            crop_new_data.append(data[:, idx:min_series+idx])
    return np.stack(crop_new_data)


def add_noise(disease_data):
    for d in disease_data:
        if np.random.rand() > 2:
            d += np.random.normal(0, 1e-3)


icn_bp = np_load_files("../data/train/bp", 'icn_tc.npy')
icn_sz = np_load_files("../data/train/sz", 'icn_tc.npy')

sz_crop_data = create_crop_data(icn_sz)
bp_crop_data = create_crop_data(icn_bp)

m = sz_crop_data.shape[0] + bp_crop_data.shape[0]
icn_dataset = np.concatenate([sz_crop_data, bp_crop_data])
icn_labels = np.zeros(m)
icn_labels[sz_crop_data.shape[0]:] = 1
# np.save('icn_crop_train_dataset.npy', icn_dataset, allow_pickle=True, fix_imports=True)
# np.save('icn_crop_train_dataset_labels.npy', icn_labels, allow_pickle=True, fix_imports=True)