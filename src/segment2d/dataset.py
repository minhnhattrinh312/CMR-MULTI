import glob
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import defaultdict
import torch
import numpy as np
import nibabel as nib
import random
class Image_Loader(Dataset):

    def __init__(self, train_path="", list_subject=[]):
        if list_subject:
            self.listName = list_subject
        else:
            self.listName = glob.glob(train_path)

    def __len__(self):
        return len(self.listName)

    def __getitem__(self, idx):
        data = np.load(self.listName[idx])
        image, mask = data["image"], data["mask"]
        # convert image, mask to tensor
        image = torch.tensor(image.transpose(-1, 0, 1), dtype=torch.float32)

        mask = torch.from_numpy(mask[None])
        return image, mask.float()


class MultiViewDataset(Dataset):
    def __init__(self, files_dict):
        """
        files_dict: dict of view_name -> list of .npz paths
        """
        self.samples = []
        for view, files in files_dict.items():
            for f in files:
                self.samples.append({"file": f, "view": view})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = np.load(sample["file"])
        image, mask = data["image"], data["mask"]
        # convert to tensor
        image = torch.tensor(image.transpose(-1, 0, 1), dtype=torch.float32)
        mask = torch.from_numpy(mask[None]).long()
        return image, mask, sample["view"]


class SingleViewBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=4, shuffle=True):
        """
        dataset: MultiViewDataset
        batch_size: number of samples per batch (all from same view)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Build indices per view
        self.view_indices = defaultdict(list)
        for i, s in enumerate(dataset.samples):
            self.view_indices[s["view"]].append(i)

    def __iter__(self):
        # Shuffle indices per view
        for view in self.view_indices:
            if self.shuffle:
                random.shuffle(self.view_indices[view])

        # Round-robin through views
        view_names = list(self.view_indices.keys())
        view_iters = {v: iter(self.view_indices[v]) for v in view_names}
        finished_views = set()
        while len(finished_views) < len(view_names):
            for v in view_names:
                if v in finished_views:
                    continue
                batch = []
                try:
                    for _ in range(self.batch_size):
                        batch.append(next(view_iters[v]))
                    yield batch
                except StopIteration:
                    finished_views.add(v)


# Example usage:

# files_dict = {
#     "SAX": sax_files,
#     "2CH": ch2_files,
#     "4CH": ch4_files,
# }

# dataset = MultiViewDataset(files_dict)
# batch_sampler = SingleViewBatchSampler(dataset, batch_size=4)

# loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)
