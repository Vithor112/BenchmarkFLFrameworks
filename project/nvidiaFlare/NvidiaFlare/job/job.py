from model import SimpleNetwork

from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.model import PTModel

from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import ScriptRunner

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 2
    train_script = "job/client.py"

    job = FedJob(name="job_test")

    controller = FedAvg(
        num_clients=n_clients,
        num_rounds=num_rounds,
    )
    job.to_server(controller)

    job.to_server(PTModel(SimpleNetwork()))

    job.to_server(IntimeModelSelector(key_metric="accuracy"))

    runner = ScriptRunner(
        script=train_script, script_args="f--batch_size 32 --data_path /tmp/data/site-{i}"
    )
    job.to_clients(runner)
    job.export_job("/tmp/nvflare/jobs/job_config")
