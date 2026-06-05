import argparse
import math
import os

import torch

os.environ['FBM_RESEARCHER_COMPONENT_ROOT'] = './fbm-researcher'

import monai
import torch.nn.functional as F
from torch.optim import Adam

# MONAI import
from monai.networks.nets import resnet10

from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.metrics import MetricTypes
from fedbiomed.researcher.federated_workflows import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.common.logger import logger
from monai.transforms import Compose, EnsureChannelFirst, LoadImage, RandFlip, RandRotate, RandZoom, ScaleIntensity, ToTensor

parser = argparse.ArgumentParser(description="Fedbiomed Client Training Script")
parser.add_argument("--batch_size", type=int, default=32, help="Input batch size for training")
parser.add_argument("--epochs", type=int, default=1, help="Input batch size for training")
parser.add_argument("--num_of_rounds", type=int, default=2, help="Input batch size for training")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()

print(f"Successfully loaded arguments!")
print(f"Batch Size: {args.batch_size}")
print(f"Epochs: {args.epochs}")
print(f"Number of Rounds: {args.num_of_rounds}")
print(f"Seed: {args.seed}")

class MyTrainingPlan(TorchTrainingPlan):

    def init_model(self, model_args):
        monai.utils.set_determinism(seed=model_args.get('seed'))
        return resnet10(
            spatial_dims=2,
            n_input_channels=1,
            num_classes=6
        )

    def init_optimizer(self, optimizer_args):
        return Adam(self.model().parameters(), lr=optimizer_args.get("lr", 1e-5))

    def init_dependencies(self):
        return [
            "import monai",
            "import os",
            "import torch",
            "import torch.nn.functional as F",
            "from torch.utils.data import Dataset",
            "from torch.optim import Adam",
            "from monai.networks.nets import resnet10",
            "from monai.transforms import Compose, EnsureChannelFirst, LoadImage, RandFlip, RandRotate, RandZoom, ScaleIntensity, ToTensor",
            "from fedbiomed.common.datamanager import DataManager",
            "from fedbiomed.common.logger import logger",
            "import math"
        ]

    def training_data(self):
        from torch.utils.data import Dataset
        from monai.transforms import Compose, EnsureChannelFirst, LoadImage, RandFlip, RandRotate, RandZoom, ScaleIntensity, ToTensor
        import os

        class MedNISTDataset(Dataset):
            def __init__(self, data_dir, transform=None):
                self.data_dir = data_dir
                self.transform = transform

                self.class_names = sorted(
                    [x for x in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, x)) and (not x.startswith('.') and not x.startswith('MedNIST'))]
                )

                valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
                image_files = [
                    [
                        os.path.join(data_dir, class_name, x)
                        for x in os.listdir(os.path.join(data_dir, class_name))
                        if x.lower().endswith(valid_exts)
                    ]
                    for class_name in self.class_names
                ]
                
                self.global_label_map = {
                    'AbdomenCT': 0,
                    'BreastMRI': 1,
                    'ChestCT': 2,
                    'CXR': 3,
                    'Hand': 4,
                    'HeadCT': 5 
                }

                self.image_file_list = []
                self.image_label_list = []
                for i, class_name in enumerate(self.class_names):
                    self.image_file_list.extend(image_files[i])
                    global_label = self.global_label_map[class_name]
                    self.image_label_list.extend([global_label] * len(image_files[i]))
                    logger.info(f"Found {len(image_files[i])} images for class '{class_name}' mapped to global label {global_label}")
                logger.info(f"Loaded {len(self.image_file_list)} images from {data_dir}")

            def __len__(self):
                return len(self.image_file_list)

            def __getitem__(self, idx):
                img_path = self.image_file_list[idx]
                label = self.image_label_list[idx]

                if self.transform:
                    try:
                        img = self.transform(img_path)
                    except RuntimeError as e:
                        logger.error(f"Error occurred while transforming image {img_path}: {e}")
                        raise e
                else:
                    img = img_path

                return img, label

        data_dir = getattr(self, 'dataset_path', "/app/MedNIST")

        train_transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensity()
        ])
        dataset = MedNISTDataset(data_dir=data_dir, transform=train_transforms)

        loader_arguments = {'shuffle': True}
        return DataManager(dataset, **loader_arguments)
    def testing_step(self, data, target):
        transform = ToTensor()
        data = transform(data)
        out = self.model()(data)
        pred = out.argmax(dim=1)
        acc = torch.sum(pred == target)
        accuracy = acc / len(target)
        logger.info(f"Accuracy  {accuracy:.4f} and samples {len(target)}")
        return {'ACCURACY': accuracy}
    def training_step(self, data, target):
        train_transforms =Compose([RandRotate(range_x=math.pi/12, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
            ToTensor()
        ])
        data = train_transforms(data)
        output = self.model()(data)
        loss = F.cross_entropy(output, target.long())
        return loss
    
model_args = {
    'seed' : args.seed
}
training_args = {
    'loader_args': {'batch_size': args.batch_size},
    'optimizer_args': {
        "lr": 1e-5
    },
    'test_ratio': 0.2,
    'test_batch_size': args.batch_size,
    'test_metric': MetricTypes.ACCURACY,
    'test_on_global_updates': True,
    'use_gpu': True,
    'epochs': args.epochs,
    'dry_run': False,
    'random_seed': args.seed
}

tags = ['#MEDNIST', '#dataset']
rounds = args.num_of_rounds

exp = Experiment(
    tags=tags,
    model_args=model_args,
    training_plan_class=MyTrainingPlan,
    training_args=training_args,
    round_limit=rounds,
    aggregator=FedAverage(),
    node_selection_strategy=None
)

exp.run()

try:
    exp.training_plan().export_model('./trained_model')
except Exception as e:
    print(e)

print("\nList the training rounds : ", exp.training_replies().keys())

print("\nList the nodes for the last training round and their timings : ")
round_data = exp.training_replies()[rounds - 1]
for r in round_data.values():
    print(
        "\t- {id} :\n"
        "\t\trtime_training={rtraining:.2f} seconds\n"
        "\t\tptime_training={ptraining:.2f} seconds\n"
        "\t\trtime_total={rtotal:.2f} seconds".format(
            id=r['node_id'],
            rtraining=r['timing']['rtime_training'],
            ptraining=r['timing']['ptime_training'],
            rtotal=r['timing']['rtime_total']
        )
    )
print('\n')

print("\nList the training rounds : ", exp.aggregated_params().keys())

print("\nAccess the federated params for the last training round :")
print("\t- parameter data: ", exp.aggregated_params()[rounds - 1]['params'].keys())