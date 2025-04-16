from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import pdb

# Note: num_colors = 1 or 3 depending on the number of channels you want to set for input image 
def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False, num_colors=3, obj_cond=False,
two_cls_cond=False):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised. ---> TDN : this is the topological constraint
    :param obj_cond: if True, include a "y_obj" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised. ----> TDN : object type condition (giraffe etc)
    :param two_cls_cond: if True, include a "y0" and "y1" key in returned dicts for class
                       labels 0dim and 1dim. This flag is to train 0dim and 1dim together. If classes are not available and this is true, an
                       exception will be raised. ----> TDN : train 0dim and 1dim together
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    o_classes = None
    two_classes_0 = None
    two_classes_1 = None
    if class_cond:
        # Assume classes are the first part of the filename, before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: float(x) for _, x in enumerate(sorted(set(class_names)))} 
        classes = [sorted_classes[x] for x in class_names]
    if obj_cond:
        o_class_names = [bf.basename(path).split("_")[1] for path in all_files] # filename form num_animal_xxxx.png
        o_sorted_classes = {x: i for i, x in enumerate(sorted(set(o_class_names)))}
        o_classes = [o_sorted_classes[x] for x in o_class_names] # ctrl+F o_classes to see usage
    if two_cls_cond:
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: float(x) for _, x in enumerate(sorted(set(class_names)))} 
        two_classes_0 = [sorted_classes[x] for x in class_names]

        class_names = [bf.basename(path).split("_")[1] for path in all_files]
        sorted_classes = {x: float(x) for _, x in enumerate(sorted(set(class_names)))}
        two_classes_1 = [sorted_classes[x] for x in class_names]


    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        num_colors=num_colors,
        o_classes=o_classes,
        two_classes_0=two_classes_0,
        two_classes_1=two_classes_1,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1, num_colors=3, o_classes=None, two_classes_0=None, two_classes_1=None):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

        self.o_local_classes = None if o_classes is None else o_classes[shard:][::num_shards]

        self.local_two_classes_0 = None if two_classes_0 is None else two_classes_0[shard:][::num_shards]
        self.local_two_classes_1 = None if two_classes_1 is None else two_classes_1[shard:][::num_shards]

        if classes is not None:
            print("[TDN] Unique self.local_classes (numeric) in ImageDataset: {}".format(set(self.local_classes)))
        if o_classes is not None:
            print("[TDN] Unique self.o_local_classes (obj) in ImageDataset: {}".format(set(self.o_local_classes)))
        if two_classes_0 is not None:
            print("[TDN] Unique self.local_two_classes_0 (0dim from 01dim) in ImageDataset: {}".format(set(self.local_two_classes_0)))
        if two_classes_1 is not None:
            print("[TDN] Unique self.local_two_classes_1 (1dim from 01dim) in ImageDataset: {}".format(set(self.local_two_classes_1)))
        self.num_colors = num_colors
        print("[TDN] num_colors in ImageDataset: {}".format(num_colors))
        #self.cntr = 0

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        if self.num_colors == 3:
            arr = np.array(pil_image.convert("RGB"))
        elif self.num_colors == 1:
            arr = np.array(pil_image)
            arr = arr.reshape((arr.shape[0], arr.shape[1], 1))

        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.float32) # TDN: Setting class label to float for nn.Linear layer
        if self.o_local_classes is not None:
            out_dict["y_obj"] = np.array(self.o_local_classes[idx], dtype=np.int64)
        if self.local_two_classes_0 is not None:
            out_dict["y0"] = np.array(self.local_two_classes_0[idx], dtype=np.float32) # TDN: Setting class label to float for nn.Linear layer
        if self.local_two_classes_1 is not None:
            out_dict["y1"] = np.array(self.local_two_classes_1[idx], dtype=np.float32) # TDN: Setting class label to float for nn.Linear layer
        return np.transpose(arr, [2, 0, 1]), out_dict
