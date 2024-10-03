import numpy as np
from sklearn.model_selection import train_test_split
import random

class DataPreparer:
    def __init__(self, config, dataset_list, labels, verbose=0):
        self.config = config
        self.dataset_list = dataset_list
        self.labels = labels
        self.verbose = verbose
        self.validate_input()
        self.subjects_all = list(self.labels.keys())

    def validate_input(self):
        dataset_keys = [set(d.keys()) for d in self.dataset_list]
        label_keys = set(self.labels.keys())

        if not all(keys == dataset_keys[0] for keys in dataset_keys):
            raise ValueError("Inconsistent keys found in dataset_list.")
        if dataset_keys[0] != label_keys:
            raise ValueError("Keys in dataset_list do not match keys in labels.")
        if not all(len(d) == len(self.labels) for d in self.dataset_list):
            raise ValueError("Inconsistent dataset lengths found.")
    

    def prepare_dataset_dict(self):
        if 'method' not in self.config:
            raise ValueError("Method is required in config.")
        method = self.config['method'] # required
        val_st_per = self.config['val_st_per'] if 'val_st_per' in self.config else 0 # temporal
        val_ratio = self.config['val_ratio'] if 'val_ratio' in self.config else 0.2 # temporal, random
        win_len = self.config['win_len'] if 'win_len' in self.config else 1.5 # temporal
        extra_thickness = self.config['extra_thickness'] if 'extra_thickness' in self.config else 6 # random, holdout
        depth_avai = self.config['depth_available'] if 'depth_available' in self.config else False # random, holdout
        val_subject_id = self.config['val_subject_id'] if 'val_subject_id' in self.config else self.subjects_all[len(self.subjects_all)//2] # holdout
        num_samples = self.config['num_samples'] if 'num_samples' in self.config else 1000 # pair

        X_train_dict, X_test_dict, y_train_dict, y_test_dict = {}, {}, {}, {}
        sample_cnt = []
        random.seed(42)
        seeds = random.sample(range(1000), 31) # random seeds for pair method
        for idx, subject in enumerate(self.subjects_all):
            if self.verbose >0:
                print(f'========================={subject}=============================')
            if method == 'temporal':
                X_train_dict[subject], X_test_dict[subject], y_train_dict[subject], y_test_dict[subject] = self.prepare_temporal(subject, idx, val_st_per, val_ratio, win_len)
                val_st_per = (int(val_st_per*10) + 2)%10/10
            elif method == 'random':
                X_train_dict[subject], X_test_dict[subject], y_train_dict[subject], y_test_dict[subject] = self.prepare_random(subject, idx, val_ratio, extra_thickness, depth_avai)
            elif method == 'holdout':
                X_train_dict[subject], X_test_dict[subject], y_train_dict[subject], y_test_dict[subject] = self.prepare_holdout(subject, idx, depth_avai, extra_thickness, val_subject_id)
            elif method == 'pair':
                X_train_dict[subject], X_test_dict[subject], y_train_dict[subject], y_test_dict[subject] = self.prepare_pair(subject, idx, num_samples, seeds[idx])
            elif method == 'holdout_pair':
                if idx == 12:
                    X_test_dict[subject], y_test_dict[subject] = self.prepare_pair(subject, idx, num_samples, seeds[idx], split=False)
                else:
                    X_train_dict[subject], X_test_dict[subject], y_train_dict[subject], y_test_dict[subject] = self.prepare_pair(subject, idx, num_samples, seeds[idx])
            else:
                raise ValueError(f'Unknown method {method}.')
        
        X_train_dict = {k: v for k, v in X_train_dict.items() if v is not None}
        X_test_dict = {k: v for k, v in X_test_dict.items() if v is not None}
        y_train_dict = {k: v for k, v in y_train_dict.items() if v is not None}
        y_test_dict = {k: v for k, v in y_test_dict.items() if v is not None}
        sample_cnt = [len(v) for v in X_train_dict.values()]
        return X_train_dict, X_test_dict, y_train_dict, y_test_dict, sample_cnt
    

    def prepare_dataset(self):
        dataset_dict = self.prepare_dataset_dict()
        X_train = np.concatenate([dataset_dict[0][k] for k in dataset_dict[0].keys()], axis=0)
        X_test = np.concatenate([dataset_dict[1][k] for k in dataset_dict[1].keys()], axis=0)
        y_train = np.concatenate([dataset_dict[2][k] for k in dataset_dict[2].keys()], axis=0)
        y_test = np.concatenate([dataset_dict[3][k] for k in dataset_dict[3].keys()], axis=0)
        return X_train, X_test, y_train, y_test
        

    def prepare_temporal(self, subject, subject_id, val_st_per, val_ratio, win_len):
        labels = self.labels[subject].reshape(-1, 1)
        subject_data = [self.dataset_list[i][subject] for i in range(len(self.dataset_list))]

        n = len(labels)
        val_start_idx = int(n * val_st_per)
        val_len = int(n * val_ratio)
        win_len_half = int(win_len*60 // 2)
        
        val_start_idx = max(val_start_idx, win_len_half)
        val_end_idx = val_start_idx + val_len
        
        if val_end_idx > n - win_len_half:
            val_end_idx = n - win_len_half
            val_start_idx = val_end_idx - val_len
        
        train_indices = list(range(0, val_start_idx - win_len_half)) + list(range(val_end_idx + win_len_half, n))
        val_indices = list(range(val_start_idx, val_end_idx))

        SaO2_train = labels[train_indices].reshape(-1, 1)
        SaO2_val = labels[val_indices].reshape(-1, 1)

        X_train = np.concatenate([subject_data[i][train_indices] for i in range(len(subject_data))], axis=1)
        X_test = np.concatenate([subject_data[i][val_indices] for i in range(len(subject_data))], axis=1)
        y_train = np.concatenate((SaO2_train, np.ones_like(SaO2_train)*subject_id), axis=1)
        y_test = np.concatenate((SaO2_val, np.ones_like(SaO2_val)*subject_id), axis=1)

        if self.verbose > 0:
            print(f"Training samples: {len(train_indices)}, Validation samples: {len(val_indices)}")
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
    
    def prepare_random(self, subject, subject_id, val_ratio, extra_thickness, depth_avai):
        depth = subject + extra_thickness
        labels = self.labels[subject].reshape(-1, 1)
        subject_data = [self.dataset_list[i][subject] for i in range(len(self.dataset_list))]
        X = np.concatenate(subject_data, axis=1)
        if depth_avai:
            X = np.concatenate((X, np.ones_like(labels)*depth), axis=1)
        y = np.concatenate((labels, np.ones_like(labels)*subject_id), axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_ratio, random_state=42)
        if self.verbose > 0:
            print(f"Training samples: {len(X_train)}, Validation samples: {len(X_test)}")
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)



        
    


        
            
            
        
            

    