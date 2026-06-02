"""monaiexample: A Flower / MONAI app."""

import time

from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from prometheus_client import start_http_server, Gauge

from monaiexample.task import load_model, set_seed

app = ServerApp()

FLOWER_EVAL_ACC = Gauge(
    "flower_evaluate_metrics_clientapp_eval_acc",
    "Evaluation accuracy metrics from Flower clients",
    ["round"]
)

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    num_rounds: int = context.run_config["num_of_rounds"]
    print(f"Starting ServerApp with num_rounds={num_rounds}...")

    seed: int = context.run_config["seed"]
    set_seed(seed)
    model = load_model()
    arrays = ArrayRecord(model.state_dict())

    strategy = FedAvg()

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )
    start_http_server(18000)
    for round in result.evaluate_metrics_clientapp:
        metrics = result.evaluate_metrics_clientapp[round]
        print(f"Round {round} - Accuracy metrics: {metrics['eval_acc']}")
        FLOWER_EVAL_ACC.labels(round=str(round)).set(metrics['eval_acc'])
    time.sleep(180)