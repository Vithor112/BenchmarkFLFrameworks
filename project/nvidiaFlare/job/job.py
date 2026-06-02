import argparse

from nvflare.app_common.workflows.fedavg import FedAvg

from nvflare.job_config.api import  FedJob
from nvflare.job_config.script_runner import ScriptRunner
from prometheus import PrometheusMetricExporter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NVFlare Client Training Script")
    parser.add_argument("--batch_size", type=int, default=32, help="Input batch size for training")
    parser.add_argument("--num_of_clients", type=int, default=2, help="Input batch size for training")
    parser.add_argument("--epochs", type=int, default=1, help="Input batch size for training")
    parser.add_argument("--num_of_rounds", type=int, default=2, help="Input batch size for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    print(f"Successfully loaded arguments!")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Clients: {args.num_of_clients}")
    print(f"Epochs: {args.epochs}")
    print(f"Number of Rounds: {args.num_of_rounds}")
    print(f"Seed: {args.seed}")
    print("Creating and configuring NVFlare job...")
    job = FedJob(name="job_test")

    controller = FedAvg(
        num_clients=args.num_of_clients,
        num_rounds=args.num_of_rounds + 1, # add 1 to account for round final where no training happens, just evaluation of the global model
    )
    job.to_server(controller)
    job.to_server(PrometheusMetricExporter(port=18000, metric_name="accuracy", number_of_clients=args.num_of_clients))
    runner = ScriptRunner(
        script="job/client.py", 
        script_args=f"--batch_size {args.batch_size} --num_of_clients {args.num_of_clients} --epochs {args.epochs} --seed {args.seed} --num_of_rounds {args.num_of_rounds}"
    )
    job.to_clients(runner)
    job.export_job("/tmp/nvflare/jobs/job_config")
    print("Job created and exported successfully!")
