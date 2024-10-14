import torch
from torch.utils.data import Dataset, Sampler, DataLoader
import h5py
import numpy as np

from utils import load_file, load_ref_file, compute_iv
from augmentation import addGaussianNoise
from preprocess import standardization

    
### handle the dataset
class TorchDataset(Dataset):
    def __init__(self, traces, labels):
        self.traces = traces
        self.labels = labels
        print(f"traces.shape:{traces.shape}, labels.shape{labels.shape}")

    def __getitem__(self, index):
        trace, label = self.traces[index], self.labels[index]
        trace, label = torch.tensor(trace).float(), torch.tensor(label).long()
        return trace, label

    def __len__(self):
        return len(self.labels)
    

class PairDataset(Dataset):
    def __init__(self, trace_target, label_target, trace_ref, label_ref, aug_level:float=0):
        self.trace_target = trace_target
        self.label_target = label_target
        self.trace_ref = trace_ref
        self.label_ref = label_ref
        self.aug_level = aug_level
        print(f"trace_target.shape:{trace_target.shape}, label_target.shape{label_target.shape}")
        print(f"trace_ref.shape:{trace_ref.shape}, label_ref.shape{label_ref.shape}")

    def __getitem__(self, index):
        X_target, Y_target = self.trace_target[index[0]], self.label_target[index[0]] # Sample target tarce and label
        X_target = addGaussianNoise(X_target.reshape(1,-1), self.aug_level).reshape(-1)  # Data Augmentation
        F_ref, Y_ref = self.trace_ref[index[1]], self.label_ref[index[1]] # Sample reference trace and label

        X_target, Y_target = torch.tensor(X_target).float(), torch.tensor(Y_target).long()
        F_ref, Y_ref = torch.tensor(F_ref).float(), torch.tensor(Y_ref).long()
        return (X_target, Y_target), (F_ref, Y_ref)

    def __len__(self):
        return len(self.label_target)
    

class PairBatchSampler(Sampler):
    def __init__(self, target_labels, ref_labels, batch_size, shuffle:bool=True):
        self.target_labels = target_labels
        self.ref_labels = ref_labels
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        target_indices = np.arange(len(self.target_labels))
        if self.shuffle:
            np.random.shuffle(target_indices)

        ref_indices_dict = {}
        for i, label in enumerate(self.ref_labels):
            if label not in ref_indices_dict:
                ref_indices_dict[label] = []
            ref_indices_dict[label].append(i)

        for target_index in range(0, len(self.target_labels), self.batch_size):
            target_batch_indices = target_indices[target_index : target_index+self.batch_size]
            ref_batch_indices = []

            for target_idx in target_batch_indices:
                label = self.target_labels[target_idx]
                ref_idx = np.random.choice(ref_indices_dict[label])
                ref_batch_indices.append(ref_idx)

            yield list(zip(target_batch_indices, ref_batch_indices))  # Use zip to pair train and ref indices

    def __len__(self):
        return max(len(self.target_labels)//self.batch_size, len(self.ref_labels)//self.batch_size)
    

def load_dataset(ref_dataset, tar_dataset, num_ref:int, num_tar:int, num_tar_valid:int=5000, target_byte:int=0, 
                 labeling_method:str='ID', batch_size:int=400, refidx:int=0, aug_level:float=0):
    '''Creating training and validation datasets for LD-PA.'''
    
    # Reference set for training
    ref_traces, ref_labels = load_ref_file(ref_dataset, num_ref, refidx=refidx)
    ref_traces = standardization(ref_traces, desc='Preprocessing reference training features')

    # Target set for training
    tar_traces_train, tar_plaintext_train, tar_key_train, _ = load_file(tar_dataset, 0, num_tar, profile=True)
    tar_traces_train = standardization(tar_traces_train, desc='Preprocessing target training traces')
    tar_plaintext_train, tar_key_train = tar_plaintext_train[:, target_byte], tar_key_train[:, target_byte]
    tar_labels_train = compute_iv(tar_plaintext_train, tar_key_train, None, labeling_method).squeeze()

    # Target set for validation
    tar_traces_valid, tar_plaintext_valid, tar_key_valid, _ = load_file(tar_dataset, 0, num_tar_valid, profile=False)
    tar_traces_valid = standardization(tar_traces_valid, desc='Preprocessing target validation traces')
    tar_plaintext_valid, tar_key_valid = tar_plaintext_valid[:, target_byte], tar_key_valid[:, target_byte]
    tar_labels_valid = compute_iv(tar_plaintext_valid, tar_key_valid, None, labeling_method).squeeze()

    pair_dataset = PairDataset(tar_traces_train, tar_labels_train, ref_traces, ref_labels, aug_level)     # Pairsampler
    valid_dataset = TorchDataset(tar_traces_valid, tar_labels_valid)

    train_loader = DataLoader(pair_dataset, batch_sampler=PairBatchSampler(tar_labels_train, ref_labels, batch_size), num_workers=6)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=6, pin_memory=True)

    return train_loader, valid_loader
