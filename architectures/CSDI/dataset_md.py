import numpy as np
from torch.utils.data import DataLoader, Dataset


class MDTrajectoryDataset(Dataset):
    def __init__(self, positions, context_length, prediction_length,normalization=None):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.seq_len = context_length + prediction_length
        self.normalization = normalization
        N = len(positions)

        pos = positions.astype(np.float32)
        self.positions = pos

        self.observed_mask = np.ones((N, self.seq_len, 1), dtype=np.float32)
        self.gt_mask = np.ones((N, self.seq_len, 1), dtype=np.float32)
        self.gt_mask[:, -self.prediction_length:, :] = 0.0

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        seq = self.positions[index, :self.seq_len][:, None]  # (L, 1)

        local_scaler = np.array([1.0], dtype=np.float32)
        return {
            "observed_data" : seq,
            "observed_mask" : self.observed_mask[index],
            "gt_mask" : self.gt_mask[index],
            "timepoints" : np.arange(self.seq_len, dtype=np.float32),
            "feature_id" : np.array([0], dtype=np.float32),
            "local_scaler" : local_scaler,  
        }


def _make_train_windows(positions, context_length, prediction_length, stride):
    seq_len = context_length + prediction_length
    T       = positions.shape[1]
    windows = []
    for start in range(0, T - seq_len + 1, stride):
        windows.append(positions[:, start:start + seq_len])
    return np.concatenate(windows, axis=0)


'''def get_dataloader_md(npz_path, context_length, prediction_length, batch_size, val_size, test_size,
                    stride,flag="train",train_test_split=None):
    data = np.load(npz_path)
    positions = data["positions"]

    print(f"Number of trajectories: {positions.shape[0]}")
    print(f"Number of time steps: {positions.shape[1]}")

    if flag == "train":
        train_val_split = int(data["train_val_split"])
        val_start = train_val_split - context_length

        print(positions[:, :val_start].shape[0])
        print(positions[:, :val_start].shape[1])
        train_dataset = MDTrajectoryDataset(
            _make_train_windows(positions[:, :train_val_split],context_length, prediction_length, stride),
            context_length, prediction_length
        )
        
        print(positions[:val_size, val_start:].shape[0])
        print(positions[:val_size, val_start:].shape[1])
        val_dataset = MDTrajectoryDataset(positions[:val_size, val_start:],context_length, prediction_length,
        )
        
        #train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,  num_workers=1)
        #val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False, num_workers=1)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,  num_workers=0)
        val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False, num_workers=0)
    
        return train_loader, val_loader
    
    elif flag == "test":
        print(positions[:test_size,-(prediction_length+context_length):].shape)
        #train_test_split = int(data["train_test_split"])
        test_dataset = MDTrajectoryDataset(
            #positions[:test_size,-(prediction_length+context_length):],
            positions[:test_size, train_test_split - context_length : train_test_split + prediction_length],
            context_length, prediction_length,
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False, num_workers=1)
        return test_loader'''
        
def get_dataloader_md(npz_path, context_length, prediction_length, batch_size, val_size, test_size,
                    stride, flag="train", train_test_split=None):
    data = np.load(npz_path, allow_pickle=True)
    positions = data["positions"]

    print(f"Number of trajectories: {positions.shape[0]}")
    print(f"Number of time steps: {positions.shape[1]}")

    if flag == "train":
        # ---- Handle missing train_val_split ----
        try:
            train_val_split = int(data["train_val_split"])
        except (KeyError, ValueError):
            T = positions.shape[1]
            train_val_split = int(0.8 * T)  # use 80% of time steps as training
            print(f"train_val_split not found; using default {train_val_split}")

        val_start = train_val_split - context_length
        if val_start < 0:
            val_start = 0
            print(f"Adjusted val_start to {val_start}")

        train_data = positions[:, :train_val_split]
        val_data = positions[:val_size, val_start:]

        train_dataset = MDTrajectoryDataset(
            _make_train_windows(train_data, context_length, prediction_length, stride),
            context_length, prediction_length
        )
        
        val_dataset = MDTrajectoryDataset(
            val_data,
            context_length, prediction_length
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        return train_loader, val_loader
    
    elif flag == "test":
        # ---- Cast all indices to int to avoid TypeError ----
        if train_test_split is None:
            T = positions.shape[1]
            train_test_split = int(T - prediction_length)
            print(f"train_test_split not provided; using default {train_test_split}")
        else:
            train_test_split = int(train_test_split)

        start = int(train_test_split - context_length)
        end = int(train_test_split + prediction_length)
        test_size = int(test_size)

        if start < 0:
            start = 0
            print(f"Adjusted start to {start}")

        # Take up to test_size trajectories, or all if test_size is larger
        n_traj = min(test_size, positions.shape[0])
        test_data = positions[:n_traj, start:end]

        print(f"Test data shape: {test_data.shape}")

        test_dataset = MDTrajectoryDataset(test_data, context_length, prediction_length)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        return test_loader