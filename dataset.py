
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

def zscore_normalize(volume):
    mask = volume > 0
    mean = volume[mask].mean()
    std = volume[mask].std()
    volume[mask] = (volume[mask] - mean) / (std + 1e-8)
    return volume

class BraTSDataset(Dataset):
    def __init__(self, cases):
        self.cases = cases

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        modalities = []
        for mod in ['t1','t1ce','t2','flair']:
            img = nib.load(case[f'{mod}']).get_fdata()
            img = zscore_normalize(img)
            modalities.append(img)

        image = np.stack(modalities, axis=0)
        mask = nib.load(case['seg']).get_fdata()

        return torch.tensor(image, dtype=torch.float32),                torch.tensor(mask, dtype=torch.long)
