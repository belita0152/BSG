from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import mne

import warnings
warnings.filterwarnings(action='ignore')

class TorchDataset(Dataset):  # Parsing + ToTensor
    def __init__(self, data_dir, mask_dir, transform=None):
        self.ch_names = ['AIRFLOW', 'THOR RES', 'ABDO RES', 'SaO2']
        self.event_name = ['Obstructive apnea',  'Central apnea', 'Mixed apnea',
                           'Hypopnea', 'Arousal', 'SpO2 artifact']
        self.second = 300  # segmentation size
        self.sfreq = 10  # sampling rate

        self.data_path = data_dir  # len: 2535
        self.mask_path = mask_dir  # len: 2535
        self.split = {'SSL': 0.7, 'Tuning': 0.2, 'Eval': 0.1}


        self.data_x, self.data_y = self.parser(self.data_path, self.mask_path)

    def balance_data(self, x, y):
        idx_cls_0 = np.where(y == 0)[0]
        idx_cls_1 = np.where(y == 1)[0]

        n_samples = min(len(idx_cls_0), len(idx_cls_1))

        idx_cls_0_down = np.random.choice(idx_cls_0, size=n_samples, replace=False)
        idx_cls_1_down = np.random.choice(idx_cls_1, size=n_samples, replace=False)

        balanced_idx = np.concatenate([idx_cls_0_down, idx_cls_1_down])
        np.random.shuffle(balanced_idx)

        return x[balanced_idx], y[balanced_idx]


    def parser(self, data_dir, mask_dir):
        total_x, total_y = [], []
        for x_path, y_path in zip(data_dir, mask_dir):
            x = pd.read_parquet(x_path).T  # ex) (258000, 4) -> (4, 258000)
            y = pd.read_parquet(y_path).T  # ex) (258000, 1) -> (1, 258000)

            x, y = np.array(x), np.array(y)  # pd.DF -> np.array
            x = x.reshape(-1, self.second * self.sfreq, len(self.ch_names))
            y = y.reshape(-1, self.second * self.sfreq, 1)

            x = np.swapaxes(x, 1, 2)  # (86, 4, 3000)
            info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types='resp')
            x = mne.decoding.Scaler(info, scalings='median').fit_transform(x)  # 데이터 scaling하는 이유: outlier의 영향 최소화 (mne의 robust scaler를 사용할 것.)
            total_x.append(x)

            y = torch.tensor(y)  # (158423, 3000, 1)
            for i in range(y.shape[0]):
                segment_has_event = (torch.sum(y[i, :] != 0) > 0).int().item()
                total_y.append(segment_has_event)

        total_x, total_y = np.concatenate(total_x), np.array(total_y)
        total_y = np.transpose(total_y)

        total_x, total_y = self.balance_data(total_x, total_y)

        return total_x, total_y  # 전체 파싱한 데이터셋

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, item):  # 샘플마다 데이터 로드
        x = torch.tensor(self.data_x[item])
        y = torch.tensor(self.data_y[item], dtype=torch.float32)

        return x, y


class TorchSegDataset(Dataset):  # Parsing + ToTensor
    def __init__(self, data_dir, mask_dir, transform=None):
        self.ch_names = ['AIRFLOW', 'THOR RES', 'ABDO RES', 'SaO2']
        self.event_name = ['Obstructive apnea',  'Central apnea', 'Mixed apnea',
                           'Hypopnea', 'Arousal', 'SpO2 artifact']
        self.second = 300  # segmentation size
        self.sfreq = 10  # sampling rate

        self.data_path = data_dir  # len: 2535
        self.mask_path = mask_dir  # len: 2535
        self.split = {'SSL': 0.7, 'Tuning': 0.2, 'Eval': 0.1}

        self.data_x, self.data_y = self.parser(self.data_path, self.mask_path)

    def balance_data(self, x, y, seg_labels):
        # 3. 1:1 downsampling
        idx_0 = torch.where(seg_labels == 0)[0]
        idx_1 = torch.where(seg_labels == 1)[0]
        n_sample = min(len(idx_0), len(idx_1))

        idx_0_down = idx_0[torch.randperm(len(idx_0))[:n_sample]]
        idx_1_down = idx_1[torch.randperm(len(idx_1))[:n_sample]]

        selected_idx = torch.cat([idx_0_down, idx_1_down])
        selected_idx = selected_idx[torch.randperm(len(selected_idx))]

        x_selected = x[selected_idx]
        y_selected = y[selected_idx]

        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types='resp')
        scaled = []
        for x_seg in x_selected:
            x_np = x_seg.numpy()
            x_scaled = mne.decoding.Scaler(info, scalings='median').fit_transform(x_np[np.newaxis])[0]
            scaled.append(torch.tensor(x_scaled, dtype=torch.float32))

        final_x = torch.stack(scaled)
        final_y = torch.tensor(y_selected)
        return final_x, final_y

    def parser(self, data_dir, mask_dir):
        all_x, all_y, seg_labels = [], [], []

        for x_path, y_path in zip(data_dir, mask_dir):
            # Load and reshape
            x = pd.read_parquet(x_path).T.values  # (4, T)
            y = pd.read_parquet(y_path).T.values.squeeze()  # (T,)

            total_len = (x.shape[1] // (self.second * self.sfreq)) * (self.second * self.sfreq)
            x = x[:, :total_len]
            y = y[:total_len]

            x = x.reshape(len(self.ch_names), -1, self.second * self.sfreq).transpose(1, 0, 2)  # (n_seg, 4, 3000)
            y = y.reshape(-1, self.second * self.sfreq)  # (n_seg, 3000)

            # 1. Binary 변환 먼저 (이벤트 있으면 1)
            y_bin = (y != 0).astype(np.int32)  # (n_seg, 3000)

            # 2. Segment-level label 추출
            seg_label = (y_bin.sum(axis=1) > 0).astype(np.int32)  # (n_seg,)

            all_x.append(torch.tensor(x, dtype=torch.float32))
            all_y.append(torch.tensor(y_bin, dtype=torch.float32))
            seg_labels.append(torch.tensor(seg_label, dtype=torch.int32))

        # 전체 모으기
        all_x = torch.cat(all_x)  # (N, 4, 3000)
        all_y = torch.cat(all_y)  # (N, 3000)
        seg_labels = torch.cat(seg_labels)  # (N,)

        total_x, total_y = self.balance_data(all_x, all_y, seg_labels)  # torch.Size([36374, 4, 3000]) torch.Size([36374, 3000])
        return total_x, total_y

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, item):  # 샘플마다 데이터 로드
        x = torch.tensor(self.data_x[item])
        y = torch.tensor(self.data_y[item], dtype=torch.float32)

        return x, y