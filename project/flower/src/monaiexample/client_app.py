"""monaiexample: A Flower / MONAI app."""

import logging

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from monaiexample.task import set_seed, load_data, load_model, test_func, train_func

app = ClientApp()

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s"
)


logger = logging.getLogger(__name__)

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    seed = context.run_config["seed"]
    set_seed(seed)
    model = load_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Torch CUDA version: {torch.version.cuda}")
    logger.info(f"Loaded model on device {device}.")
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch_size"]
    epochs = context.run_config["epochs"]
    trainloader, _ = load_data(num_partitions, partition_id, batch_size)
    logger.info(f"Starting training on partition {partition_id} with {len(trainloader.dataset)} examples.")
    train_loss = train_func(model, trainloader, epoch_num=epochs, device=device)
    logger.info(f"Finished training on partition {partition_id} with average loss {train_loss:.4f}.")
    model_record = ArrayRecord(model.state_dict())
    metrics = {"train_loss": train_loss, "num-examples": len(trainloader)}
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    seed = context.run_config["seed"]
    set_seed(seed)  
    model = load_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch_size"]
    _, valloader = load_data(num_partitions, partition_id, batch_size)

    eval_loss, eval_acc = test_func(
        model,
        valloader,
        device,
    )

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)