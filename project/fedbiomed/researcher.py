import os
os.environ['FBM_RESEARCHER_COMPONENT_ROOT'] = './fbm-researcher'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.datamanager import DataManager
from torchvision import transforms
from fedbiomed.common.dataset import MnistDataset

# Here we define the training plan to be used.
# You can use any class name (here 'MyTrainingPlan')
class MyTrainingPlan(TorchTrainingPlan):
    class Net(nn.Module):
        def __init__(self, model_args):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

    def init_model(self, model_args):
        return self.Net(model_args = model_args)

    def init_optimizer(self, optimizer_args):
        return Adam(self.model().parameters(), lr = optimizer_args["lr"])

    def init_dependencies(self):
        return ["from fedbiomed.common.dataset import MnistDataset",
                "from torchvision import transforms",
                "from torch.optim import Adam"]

    def training_data(self):
        transform = transforms.Normalize((0.1307,), (0.3081,))
        dataset1 = MnistDataset(transform=transform)
        loader_arguments = {'shuffle': True}
        return DataManager(dataset1, **loader_arguments)

    def training_step(self, data, target):
        output = self.model().forward(data)
        loss   = torch.nn.functional.nll_loss(output, target)
        return loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import  transforms

from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.dataset import MnistDataset

from fedbiomed.common.metrics import MetricTypes
model_args = {}

training_args = {
    'loader_args': { 'batch_size': 1, }, 
    'optimizer_args': {
        "lr" : 1e-3
    },
    # 'test_ratio' : 0.25,
    # 'test_batch_size': 64,
    # 'test_metric': MetricTypes.F1_SCORE,
    # 'test_on_global_updates': True,
    # 'test_on_local_updates': True,
    # 'test_metric_args': {'average': 'marco'},
    'use_gpu': True,  # automatically falls back to cpu on nodes that don't support gpu
    'epochs': 1, 
    'dry_run': False,  
    'batch_maxnum': 100 # Fast pass for development : only use ( batch_maxnum * batch_size ) samples
}


from fedbiomed.researcher.federated_workflows import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage

tags =  ['#MNIST', '#dataset']
rounds = 5

exp = Experiment(tags=tags,
                 model_args=model_args,
                 training_plan_class=MyTrainingPlan,
                 training_args=training_args,
                 round_limit=rounds,
                 aggregator=FedAverage(),
                 node_selection_strategy=None)

exp.run()
exp.run_once(increase=True)
try: 
    exp.training_plan().export_model('./trained_model')
except Exception as e:
    print(e)
    
print("\nList the training rounds : ", exp.training_replies().keys())

print("\nList the nodes for the last training round and their timings : ")
round_data = exp.training_replies()[rounds - 1]
for r in round_data.values():
    print("\t- {id} :\
    \n\t\trtime_training={rtraining:.2f} seconds\
    \n\t\tptime_training={ptraining:.2f} seconds\
    \n\t\trtime_total={rtotal:.2f} seconds".format(id = r['node_id'],
        rtraining = r['timing']['rtime_training'],
        ptraining = r['timing']['ptime_training'],
        rtotal = r['timing']['rtime_total']))
print('\n')

print("\nList the training rounds : ", exp.aggregated_params().keys())

print("\nAccess the federated params for the last training round :")
print("\t- parameter data: ", exp.aggregated_params()[rounds - 1]['params'].keys())
