""" MONAI app."""

import math
import os
from collections import Counter, OrderedDict
import collections 

from datasets import Dataset
from flwr_datasets.partitioner import IidPartitioner, PathologicalPartitioner
from monai.networks.nets import resnet10
import monai
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    ToTensor,
)

seed = 42

def set_seed(seed_to_set):
    """Set the seed for reproducibility."""
    global seed
    seed = seed_to_set
    print(f"Seed set to {seed}.")

def load_model():
    """Load a resnet10."""
    print(f"Initializing model with seed {seed}.")
    monai.utils.set_determinism(seed=seed)
    return resnet10(
        spatial_dims=2,
        n_input_channels=1,
        num_classes=6
    )

def get_params(model):
    """Return tensors in the model's state_dict."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model, ndarrays):
    """Apply parameters to a model."""
    params_dict = zip(model.state_dict().keys(), ndarrays)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def train_func(model, train_loader, epoch_num, device):
    """Train a model using the supplied dataloader."""
    model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    running_loss = 0.0
    for _ in range(epoch_num):
        print(f"Epoch {_+1}/{epoch_num}")
        model.train()
        for batch in train_loader:
            images, labels = batch["img"], batch["label"]
            optimizer.zero_grad()
            loss = loss_function(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(train_loader)
    return avg_trainloss


def test_func(model, test_loader, device):
    """Evaluate a model on a held-out dataset."""
    model.to(device)
    model.eval()
    loss = 0.0
    y_true = list()
    y_pred = list()
    loss_function = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch["img"], batch["label"]
            out = model(images.to(device))
            labels = labels.to(device)
            loss += loss_function(out, labels).item()
            pred = out.argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(labels[i].item())
                y_pred.append(pred[i].item())
    accuracy = sum([1 if t == p else 0 for t, p in zip(y_true, y_pred)]) / len(
        test_loader.dataset
    )
    return loss, accuracy


def _get_transforms():
    """Return transforms to be used for training and evaluation."""
    train_transforms = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensity(),
            RandRotate(range_x=math.pi/12, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
            ToTensor(),
        ]
    )

    val_transforms = Compose(
        [LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity(), ToTensor()]
    )

    return train_transforms, val_transforms


def get_apply_transforms_fn(transforms_to_apply):
    """Return a function that applies the transforms passed as input argument."""

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [transforms_to_apply(img) for img in batch["img_file"]]
        return batch

    return apply_transforms


ds = None
partitioner = None
global_label_map = {
    'AbdomenCT': 0,
    'BreastMRI': 1,
    'ChestCT': 2,
    'CXR': 3,
    'Hand': 4,
    'HeadCT': 5 
}


def load_data(num_partitions, partition_id, batch_size):
    """Download dataset, partition it and return data loader of specific partition."""
    global ds, partitioner
    if ds is None:
        image_file_list, image_label_list = _download_data()

        ds = Dataset.from_dict({"img_file": image_file_list, "label": image_label_list})
        ds = ds.shuffle(seed=seed)


        partitioner = PathologicalPartitioner(num_partitions=num_partitions,
                                                partition_by="label",
                                                num_classes_per_partition=3,
                                                class_assignment_mode="first-deterministic", seed=seed
                                            )
        partitioner.dataset = ds

    partition = partitioner.load_partition(partition_id)
    label_counts = Counter(partition["label"])
    class_distribution = {list(global_label_map.keys())[list(global_label_map.values()).index(label)] : count for label, count in label_counts.items()}
    print(f"Class distribution in partition {partition_id}: {class_distribution}")

    print(f"Partition {partition_id} has {len(partition)}, initialized with seed {seed}.")
    partition_train_test = partition.train_test_split(test_size=0.2, seed=seed)

    train_t, test_t = _get_transforms()

    train_partition = partition_train_test["train"]
    test_partition = partition_train_test["test"]

    partition_train = train_partition.with_transform(get_apply_transforms_fn(train_t))
    partition_val = test_partition.with_transform(get_apply_transforms_fn(test_t))

    train_loader = monai.data.DataLoader(
        partition_train, batch_size=batch_size, shuffle=True
    )
    val_loader = monai.data.DataLoader(partition_val, batch_size=batch_size)
    print(f"Partition {partition_id} loaded with {len(train_loader.dataset)} training examples and {len(val_loader.dataset)} validation examples.")

    return train_loader, val_loader


def _download_data():
    """Download and extract dataset."""
    print("Downloading data...")
    data_dir = "/app/MedNIST"

    class_names = sorted(
        [x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))]
    )
    print(f"Class names: {class_names}")
    image_files = [
        [
            os.path.join(data_dir, class_name, x)
            for x in os.listdir(os.path.join(data_dir, class_name))
        ]
        for class_name in class_names
    ]
    image_file_list = []
    image_label_list = []
    for i, class_name in enumerate(class_names):
        image_file_list.extend(image_files[i])
        global_label = global_label_map[class_name]
        image_label_list.extend([global_label] * len(image_files[i]))
        print(f"Found {len(image_files[i])} images for class '{class_name}' mapped to global label {global_label}")
    print(f"Data downloaded and extracted to {data_dir}")
    return image_file_list, image_label_list