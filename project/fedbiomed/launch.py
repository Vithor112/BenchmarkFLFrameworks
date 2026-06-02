import argparse
import subprocess
import os
import sys
import time
import json
import shutil
import configparser
from pathlib import Path
import stop
import datasets
from flwr_datasets.partitioner import PathologicalPartitioner

script_path = Path(__file__).resolve()
RESEARCHER_IP = "172.25.0.10"
NETWORK_NAME = "fedbiomed_fbm-network"

def _download_data():
    """Download and extract dataset."""
    print("Downloading data...")
    data_dir = script_path.parent / "MedNIST"
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
    for i, _ in enumerate(class_names):
        image_file_list.extend(image_files[i])
        image_label_list.extend([i] * len(image_files[i]))
    print(f"Data downloaded and extracted to {data_dir}")
    return image_file_list, image_label_list


parser = argparse.ArgumentParser(description="Fedbiomed Client Training Script")
parser.add_argument("--batch_size", type=int, default=32, help="Input batch size for training")
parser.add_argument("--num_of_clients", type=int, default=2, help="Number of clients for training")
parser.add_argument("--epochs", type=int, default=1, help="Input batch size for training")
parser.add_argument("--num_of_rounds", type=int, default=2, help="Input batch size for training")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()
print(f"Successfully loaded arguments!")
print(f"Batch Size: {args.batch_size}")
print(f"Epochs: {args.epochs}")
print(f"Number of Rounds: {args.num_of_rounds}")
print(f"Number of Clients: {args.num_of_clients}")
print(f"Seed: {args.seed}")

num_nodes = args.num_of_clients
stop.stop()
subprocess.run(["docker", "compose", "up", "-d"])

print("Partitioning MedNIST dataset...")
image_files, image_labels = _download_data()
ds = datasets.Dataset.from_dict({"img_file": image_files, "label": image_labels})
ds = ds.shuffle(seed=args.seed)
partitioner = PathologicalPartitioner(
    num_partitions=args.num_of_clients,
    partition_by="label",
    num_classes_per_partition=3,
    class_assignment_mode="first-deterministic", seed=args.seed
)
partitioner.dataset = ds

print("Preparing researcher component...")
os.environ["FBM_SERVER_HOST"] = RESEARCHER_IP
subprocess.run(["fedbiomed", "component", "create", "-c", "researcher", "-p", "./fbm-researcher", "--exist-ok"])
abs_researcher_dir = os.path.abspath("./fbm-researcher")
res_config_path = os.path.join(abs_researcher_dir, "etc", "config.ini")
config = configparser.ConfigParser()
config.read(res_config_path)
if 'server' in config:
    config['server']['host'] = '0.0.0.0'
with open(res_config_path, 'w') as f:
    config.write(f)

for i in range(num_nodes):
    print(f"Preparing node {i+1}...")
    node_dir = f"./fbm-node_{i+1}"
    abs_node_dir = os.path.abspath(node_dir)
    
    os.environ["FBM_RESEARCHER_IP"] = RESEARCHER_IP
    subprocess.run(["fedbiomed", "component", "create", "-c", "node", "-p", node_dir, "--exist-ok"])
    
    node_mednist_dir = os.path.join(abs_node_dir, "data", "MedNIST")
    if os.path.exists(node_mednist_dir):
        shutil.rmtree(node_mednist_dir)
    os.makedirs(node_mednist_dir, exist_ok=True)
    
    partition = partitioner.load_partition(i)
    print(f"Node {i+1} assigned {len(partition)} images.")
    for row in partition:
        src_path = row["img_file"]
        class_name = os.path.basename(os.path.dirname(src_path))
        dest_dir = os.path.join(node_mednist_dir, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(src_path, os.path.join(dest_dir, os.path.basename(src_path)))
    
    dataset_path = os.path.join(abs_node_dir, "data", "MedNIST")
    mednist_config = {
        "name": "MedNIST",
        "data_type": "mednist",
        "tags": "#MEDNIST,#dataset", 
        "description": "MedNIST Medical Imaging Dataset",
        "path": dataset_path
    }
    
    mednist_file = os.path.join(abs_node_dir, "mednist.json")
    with open(mednist_file, "w") as f:
        json.dump(mednist_config, f)
    
    subprocess.run(["fedbiomed", "node", "-p", node_dir, "dataset", "add", "--file", mednist_file])

print("Exchanging certificates via certificate-dev-setup...")
subprocess.run(["fedbiomed", "certificate-dev-setup"])

current_uid = os.getuid()
current_gid = os.getgid()
current_user = os.environ.get("USER")  
print("Launching researcher container...")
abs_researcher_script = os.path.abspath("researcher.py")
subprocess.run([
    "docker", "run", "-d", 
    "--name", "fbm-researcher", "--network", NETWORK_NAME, "--ip", RESEARCHER_IP,
    "-u", f"{current_uid}:{current_gid}",
    "-e", f"USER={current_user}",     
    "-v", f"{abs_researcher_dir}:/app/fbm-researcher", 
    "-v", f"{abs_researcher_script}:/app/researcher.py",
    "fbm-researcher:6.2.0",
    "--batch_size", str(args.batch_size),
    "--epochs", str(args.epochs),
    "--num_of_rounds", str(args.num_of_rounds),
    "--seed", str(args.seed)
])

print("Launching node containers...")
for i in range(num_nodes):
    node_dir = f"./fbm-node_{i+1}"
    abs_node_dir = os.path.abspath(node_dir)
    subprocess.run([
        "docker", "run", "-d", 
        "--name", f"fbm-node-{i+1}", 
        "--gpus", "all", "--network", NETWORK_NAME,
        "-u", f"{current_uid}:{current_gid}",
        "-e", f"USER={current_user}",          
        "-v", f"{abs_node_dir}:/app/fbm-node", 
        "-v", f"{abs_node_dir}/data:{abs_node_dir}/data",
        "fbm-node:6.2.0"
    ])
