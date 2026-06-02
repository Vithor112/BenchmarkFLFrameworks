from datetime import datetime

from prometheus_client import start_http_server, Gauge, Counter
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.dxo import from_shareable
from nvflare.widgets.widget import Widget

class PrometheusMetricExporter(Widget):
    def __init__(self, port=18000, metric_name="accuracy", number_of_clients=1):
        super().__init__()
        self.port = port
        self.metric_name = metric_name
        self.number_of_clients = number_of_clients
        self.clients_reached_final_round = 0
        self.server_started = False
        
        self.metric_gauge = Gauge(
            f'nvflare_client_{self.metric_name}', 
            f'Federated learning client {self.metric_name}', 
            ['client_name', 'instance_count', 'round', 'timestamp']
        )


    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.log_info(fl_ctx, f"Initializing PrometheusMetricExporter with port={self.port}, metric_name={self.metric_name}, number_of_clients={self.number_of_clients}")
            if not self.server_started:
                try:
                    start_http_server(self.port)
                    self.server_started = True
                    self.log_info(fl_ctx, f"Started Prometheus metrics server on port {self.port}")
                except Exception as e:
                    self.log_error(fl_ctx, f"Failed to start Prometheus server on port {self.port}: {e}")
        elif event_type == EventType.BEFORE_PROCESS_SUBMISSION:
            peer_ctx = fl_ctx.get_peer_context()
            if not peer_ctx:
                self.log_warning(fl_ctx, "No peer context available, skipping metric export.")
                return
                
            peer_name = peer_ctx.get_identity_name()
            shareable = fl_ctx.get_prop(FLContextKey.TASK_RESULT)            
            if shareable:
                dxo = from_shareable(shareable)
                dict = dxo.get_meta_prop("data")
                if dict:
                    metric_val = dict.get(self.metric_name)
                    instance_count = dict.get("instance_count")
                    round = dict.get("round")
                    final_round = dict.get("final_round", False)
                    timestamp = datetime.now().isoformat()
                    if metric_val is not None and round is not None and instance_count is not None:
                        self.metric_gauge.labels(client_name=peer_name, instance_count=instance_count, round=round, timestamp=timestamp).set(float(metric_val))
                        self.log_info(fl_ctx, f"Exported {self.metric_name}={metric_val} for {peer_name} round {round} and instance_count {instance_count}")
                    else:
                        self.log_warning(fl_ctx, f"No {self.metric_name} found in DXO meta for {peer_name}")
                    if final_round:
                        self.clients_reached_final_round += 1
                        if self.clients_reached_final_round == self.number_of_clients:
                            self.log_info(fl_ctx, f"Final round reached for all {self.clients_reached_final_round} clients. Sleeping for 180s.")
                            import time
                            time.sleep(180) # Sleep to allow Prometheus to scrape final metrics before the client shuts down

            else:
                self.log_warning(fl_ctx, f"No shareable found in context for {peer_name}, skipping metric export.")