import pathlib
import random
import h5py
from torch.utils.data import Dataset


class SliceData(Dataset):
    def __init__(self, root, transform, sample_rate=1, resolution=(320,320)):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        self.transform = transform

        self.examples = []

        files = list(pathlib.Path(root).iterdir())
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        #sample_count = 1
        for fname in sorted(files):
            #if sample_count <= 0:
            #    break
            try:
                with h5py.File(fname, 'r') as data:

                    if data.attrs['acquisition'] == 'AXT2':   # should be 'CORPD_FBK' or 'CORPDFS_FBK'

                        kspace = data['kspace']
                        if kspace.shape[-2] < resolution[0] or kspace.shape[-1] < resolution[1]:
                            continue

                        self.examples += [fname]
            except:
                continue
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace']
            target = data['reconstruction_rss'] if 'reconstruction_rss' in data else None
            return self.transform(kspace, target, data.attrs, fname.name, slice)