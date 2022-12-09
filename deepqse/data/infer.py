from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from deepqse.data import get_dataset


class InferDataset(Dataset):
    def __init__(self, args, data_path):
        ds_cls, _ = get_dataset(args)
        # inference train file
        self.ds = ds_cls(args, data_path=data_path)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        return self.ds[index], index


class InferCollate(DataLoader):
    def __init__(self, args):
        _, dl_cls = get_dataset(args)
        self.dl = dl_cls(args)

    def __call__(self, data):
        ds_data, indexes = zip(*data)
        inputs, langs = self.dl(ds_data)
        return inputs, langs, indexes