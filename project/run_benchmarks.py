import random
import subprocess
import json
import time
import csv
import os
import sys
import re
from datetime import datetime, timedelta, timezone
import requests

if os.geteuid() != 0:
    print("ERROR: This script requires sudo privileges to capture hardware metrics.")
    print("Please run it again using 'sudo python3 script.py'.")
    sys.exit(1)

# --- Configuration ---
PROMETHEUS_URL = "http://127.0.0.1:9090/api/v1/query"
CSV_FILENAME = "run_metrics.csv"
SCRAPE_BUFFER_SECONDS_FLOWER = 180 
RUN_TIMEOUT_SECONDS = 10800          # 3 Hours Timeout for each run to prevent infinite waiting in case of issues

# Framework Scripts Configurations
FLOWER_START_SCRIPT = "./flower/src/start.sh"
FLOWER_STOP_SCRIPT = "./flower/src/stop.sh"

NVFLARE_START_SCRIPT = "./nvidiaFlare/start.sh"
NVFLARE_STOP_SCRIPT = "./nvidiaFlare/stop.sh"
NVFLARE_SERVER_CONTAINER_NAME = "server"

FEDBIOMED_START_SCRIPT = "./fedbiomed/start.sh"
FEDBIOMED_STOP_SCRIPT = "./fedbiomed/stop.sh"
FEDBIOMED_SERVER_CONTAINER_NAME = "fbm-researcher"

def is_nvflare_server_stopped():
    """Checks if the NVFlare server container has exited or logged the 'MPM: Good Bye!' shutdown signal."""
    try:
        status_output = subprocess.check_output(
            ['docker', 'inspect', '-f', '{{.State.Running}}', NVFLARE_SERVER_CONTAINER_NAME],
            stderr=subprocess.STDOUT,
            text=True
        ).strip()
        
        if status_output == "false":
            return True
            
        logs = subprocess.check_output(
            ['docker', 'logs', '--tail', '500', NVFLARE_SERVER_CONTAINER_NAME], 
            stderr=subprocess.STDOUT, 
            text=True
        )
        return "MPM: Good Bye!" in logs
    except Exception:
        return True

def get_flower_run_stats():
    """Polls the Flower CLI until the run finishes and returns run details."""
    print("Waiting for the Flower run to finish...")
    start_wait = time.time()
    
    while True:
        if time.time() - start_wait > RUN_TIMEOUT_SECONDS:
            raise TimeoutError("Flower run exceeded the 3-hour timeout limit.")
            
        try:
            output = subprocess.check_output(
                ['flower/env/bin/flwr', 'list', 'local-deployment', '--format', 'json'], 
                stderr=subprocess.DEVNULL
            )
            data = json.loads(output)
            
            if data and "runs" in data and len(data["runs"]) > 0:
                run = data["runs"][0]
                if run.get("finished-at") and run["finished-at"] != "N/A":
                    print(run)
                    return run
        except Exception as e:
            print(f"Error parsing flwr CLI output: {e}. Retrying...")
            
        time.sleep(10)

def poll_nvflare_status(num_clients, target_rounds, start_time_dt):
    """Polls NVFlare client accuracies from Prometheus until the last round is complete."""
    print(f"Waiting for NvidiaFlare run to finish ({target_rounds} rounds, {num_clients} clients)...")
    start_wait = time.time()
    shutdown_detected_time = None
    accuracies = {}
    end_time_dt = None

    while True:
        if time.time() - start_wait > RUN_TIMEOUT_SECONDS:
            raise TimeoutError("NvidiaFlare run exceeded the 3-hour timeout limit.")
            
        try:
            query = "nvflare_client_accuracy"
            response = requests.get(PROMETHEUS_URL, params={'query': query})
            response.raise_for_status()
            results = response.json().get('data', {}).get('result', [])

            round_data = {}
            for res in results:
                labels = res['metric']
                r = int(labels.get('round'))
                acc = float(res['value'][1])
                instance_count = int(labels.get('instance_count', 0))
                ts_str = labels.get('timestamp')

                if ts_str:
                    safe_ts = ts_str.replace("Z", "+00:00")
                    metric_dt = datetime.fromisoformat(safe_ts).astimezone(timezone.utc)
                    
                    if metric_dt > start_time_dt:
                        if r not in round_data:
                            round_data[r] = []
                        round_data[r].append({
                            'acc': acc,
                            'instance_count': instance_count,
                            'timestamp': safe_ts,
                            'client': labels.get('client_name')
                        })

            if target_rounds in round_data:
                clients_in_last_round = {c['client'] for c in round_data[target_rounds] if c['client']}
                
                if len(clients_in_last_round) >= num_clients or len(round_data[target_rounds]) >= num_clients:
                    for r, clients_data in round_data.items():
                        total_instances = sum(c['instance_count'] for c in clients_data)
                        if total_instances > 0:
                            weighted_acc = sum(c['acc'] * c['instance_count'] for c in clients_data) / total_instances
                        else:
                            weighted_acc = 0.0
                        accuracies[f"Round {r}"] = round(weighted_acc, 4)

                    final_timestamps = [c['timestamp'] for c in round_data[target_rounds]]
                    max_ts_str = max(final_timestamps)
                    end_time_dt = datetime.fromisoformat(max_ts_str).astimezone(timezone.utc) - timedelta(hours=3)
                    break
        except Exception as e:
            print(f"Error querying Prometheus for NVFlare accuracy metrics: {e}")

        if not shutdown_detected_time and is_nvflare_server_stopped():
            print("\nNVFlare server container stopped or 'MPM: Good Bye!' shutdown detected. Waiting up to 1 minute for final Prometheus scrape...")
            shutdown_detected_time = time.time()
            
        if shutdown_detected_time and (time.time() - shutdown_detected_time > 60):
            raise RuntimeError("NVFlare server shut down, but final metrics were not found in Prometheus within the 60s grace period.")

        time.sleep(10)

    run_stats = {
        "starting-at": start_time_dt.strftime("%Y-%m-%d %H:%M:%SZ"),
        "finished-at": end_time_dt.strftime("%Y-%m-%d %H:%M:%SZ"),
        "start_dt": start_time_dt,
        "end_dt": end_time_dt
    }
    return run_stats, accuracies

def clean_ansi(text):
    """Removes ANSI escape sequences (colors, etc.) from a string."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def poll_fedbiomed_status(num_clients, target_rounds, start_time_dt):
    """Polls FedBioMed logs to extract accuracies at 100% iteration and detects run completion."""
    print(f"Waiting for FedBioMed run to finish ({target_rounds} rounds, {num_clients} clients)...")
    start_wait = time.time()
    end_time_dt = None
    accuracies_per_round = {}
    
    round_data = {}
    
    processed_blocks_count = 0 
    while True:
        if time.time() - start_wait > RUN_TIMEOUT_SECONDS:
            raise TimeoutError("FedBioMed run exceeded the 3-hour timeout limit.")
            
        try:
            raw_logs = subprocess.check_output(
                ['docker', 'logs', FEDBIOMED_SERVER_CONTAINER_NAME],
                stderr=subprocess.STDOUT, text=True
            )
            
            logs = clean_ansi(raw_logs)
            blocks = logs.split("VALIDATION ON GLOBAL UPDATES")
            
            new_blocks = blocks[1:][processed_blocks_count:]
            
            if new_blocks:
                print(f"\n[FEDBIOMED PARSER] Found {len(new_blocks)} new validation block(s). Analyzing...")
            
            for i, block in enumerate(new_blocks):
                current_block_idx = processed_blocks_count + i + 1
                
                node_match = re.search(r"NODE_ID.*?NODE_([a-fA-F0-9\-]+)", block)
                round_match = re.search(r"Round\s+(\d+)", block)
                pct_match = re.search(r"\((\d+)%\)", block)
                acc_match = re.search(r"ACCURACY.*?([0-9]+\.[0-9]+)", block)
                
                if node_match and round_match and pct_match and acc_match:
                    node_id = f"NODE_{node_match.group(1).strip()}"
                    r = int(round_match.group(1))
                    pct = int(pct_match.group(1))
                    acc = float(acc_match.group(1))
                    
                    print(f"  -> [Block {current_block_idx}] MATCHED: Node: {node_id[-8:]}.. | Round: {r} | Progress: {pct}% | Acc: {acc}")
                    r -= 1
                    if pct == 100 and r >= 1:
                        if r not in round_data:
                            round_data[r] = {}
                        round_data[r][node_id] = acc
                        print(f"     [!] 100% REACHED. Saving Accuracy {acc} for round {r}.")
                    else:
                        print(f"     [-] Progress is {pct}% and round is {r}. Skipping save.")
                else:
                    print(f"  -> [Block {current_block_idx}] WARNING: Regex match failed. Incomplete log block.")
                    if "Round" in block or "ACCURACY" in block:
                         print(f"      [DEBUG FAIL] Block Content Snippet: {block[:200].replace(chr(10), ' ')}...")

            processed_blocks_count += len(new_blocks)

            if target_rounds in round_data and len(round_data[target_rounds]) >= num_clients:
                print(f"\n[FEDBIOMED PARSER] SUCCESS: All {num_clients} clients reached 100% in target Round {target_rounds}.")
                end_time_dt = datetime.now(timezone.utc)
                break
                
            status_output = subprocess.check_output(
                ['docker', 'inspect', '-f', '{{.State.Running}}', FEDBIOMED_SERVER_CONTAINER_NAME],
                stderr=subprocess.STDOUT, text=True
            ).strip()
            
            if status_output == "false":
                print(f"\n[FEDBIOMED PARSER] FALLBACK: fbm-researcher container has stopped running.")
                end_time_dt = datetime.now(timezone.utc)
                break
                
        except subprocess.CalledProcessError:
            pass 
        except Exception as e:
            print(f"[FEDBIOMED PARSER ERROR] {e}")
            
        time.sleep(10)

        
    for r, nodes_dict in round_data.items():
        if len(nodes_dict) > 0:
            avg_acc = sum(nodes_dict.values()) / len(nodes_dict)
            accuracies_per_round[f"Round {r}"] = round(avg_acc, 6)
            
    if not end_time_dt:
        end_time_dt = datetime.now(timezone.utc)
        
    run_stats = {
        "starting-at": start_time_dt.strftime("%Y-%m-%d %H:%M:%SZ"),
        "finished-at": end_time_dt.strftime("%Y-%m-%d %H:%M:%SZ"),
        "start_dt": start_time_dt,
        "end_dt": end_time_dt
    }
    
    return run_stats, accuracies_per_round

def query_prometheus(query, target_timestamp):
    """Executes a PromQL instant query."""
    try:
        response = requests.get(
            PROMETHEUS_URL, 
            params={'query': query, 'time': target_timestamp}
        )
        response.raise_for_status()
        results = response.json()['data']['result']
        if results:
            return float(results[0]['value'][1])
        return 0.0
    except Exception as e:
        print(f"Prometheus query failed: {query} -> {e}")
        return 0.0

def query_flower_accuracies(target_timestamp):
    """Fetches the accuracies per round specifically for Flower."""
    query = "flower_evaluate_metrics_clientapp_eval_acc"
    accuracies = {}
    try:
        response = requests.get(
            PROMETHEUS_URL, 
            params={'query': query, 'time': target_timestamp}
        )
        response.raise_for_status()
        results = response.json()['data']['result']
        for res in results:
            round_num = res['metric'].get('round', 'unknown')
            acc_val = float(res['value'][1])
            accuracies[f"Round {round_num}"] = round(acc_val, 4)
    except Exception as e:
        print(f"Failed to fetch Flower accuracies: {e}")
    return accuracies

def build_container_query(metric_type, containers_regex, window_s):
    """Constructs PromQL queries for container-specific cAdvisor metrics."""
    if metric_type == "cpu":
        return f'sum(rate(container_cpu_usage_seconds_total{{name=~"{containers_regex}"}}[{window_s}s]))'
    elif metric_type == "memory":
        return f'sum(avg_over_time(container_memory_usage_bytes{{name=~"{containers_regex}"}}[{window_s}s]))'
    elif metric_type == "net_rx":
        return f'sum(rate(container_network_receive_bytes_total{{name=~"{containers_regex}"}}[{window_s}s]))'
    elif metric_type == "net_tx":
        return f'sum(rate(container_network_transmit_bytes_total{{name=~"{containers_regex}"}}[{window_s}s]))'

def build_global_gpu_query(window_s):
    """Constructs PromQL query for total/host GPU utilization from DCGM exporter."""
    return f'sum(avg_over_time(DCGM_FI_DEV_GPU_UTIL[{window_s}s]))'

def get_server_metrics(server_regex, window_s, target_timestamp):
    """Fetches all server metrics depending on the regex passed."""
    return {
        "cpu": query_prometheus(build_container_query("cpu", server_regex, window_s), target_timestamp),
        "memory": query_prometheus(build_container_query("memory", server_regex, window_s), target_timestamp),
        "net_rx": query_prometheus(build_container_query("net_rx", server_regex, window_s), target_timestamp),
        "net_tx": query_prometheus(build_container_query("net_tx", server_regex, window_s), target_timestamp),
        "gpu": 0.0
    }

def get_node_container_metrics(containers_regex, window_s, target_timestamp):
    """Fetches non-GPU metrics for specific client container groupings."""
    return {
        "cpu": query_prometheus(build_container_query("cpu", containers_regex, window_s), target_timestamp),
        "memory": query_prometheus(build_container_query("memory", containers_regex, window_s), target_timestamp),
        "net_rx": query_prometheus(build_container_query("net_rx", containers_regex, window_s), target_timestamp),
        "net_tx": query_prometheus(build_container_query("net_tx", containers_regex, window_s), target_timestamp)
    }

def generate_experiment_matrix():
    """Generates the unique set of configuration definitions to test."""
    configs = []

    for c in [2, 3, 5, 7, 10]:
        configs.append({'clients': c, 'rounds': 3, 'epochs': 2, 'batch_size': 32})
    for r in [3, 5, 7]:
        configs.append({'clients': 3, 'rounds': r, 'epochs': 2, 'batch_size': 64})     
    for b in [64, 128]:
        configs.append({'clients': 3, 'rounds': 3, 'epochs': 1, 'batch_size': b})

    unique_configs = []
    seen = set()
    for cfg in configs:
        tup = (cfg['clients'], cfg['rounds'], cfg['epochs'], cfg['batch_size'])
        if tup not in seen:
            seen.add(tup)
            unique_configs.append(cfg)
            
    return unique_configs

def get_completed_runs(csv_filename):
    """Reads the CSV file to find which configurations have already been successfully run."""
    completed = set()
    if not os.path.exists(csv_filename):
        return completed

    try:
        with open(csv_filename, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    fw = row.get('Framework', 'flower')
                    c = int(row['Param_Clients'])
                    r = int(row['Param_Rounds'])
                    e = int(row['Param_Epochs'])
                    b = int(row['Param_Batch_Size'])
                    s = int(row['Param_Seed'])
                    completed.add((fw, c, r, e, b, s))
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"Warning: Could not read existing CSV to resume state: {e}")
        
    return completed

def collect_and_save_run(framework, cfg, iteration, seed):
    """Collects all run metrics and handles timeout and early exit fallbacks."""
    start_dt_local = datetime.now(timezone.utc)
    num_clients = cfg['clients']
    target_rounds = cfg['rounds']
    print(f"\n=== Starting Run: Framework={framework.upper()} | Clients={num_clients} | Rounds={target_rounds} | Epochs={cfg['epochs']} | Batch Size={cfg['batch_size']} | Seed={seed} ===")

    file_needs_header = not os.path.exists(CSV_FILENAME) or os.path.getsize(CSV_FILENAME) == 0
    fieldnames = [
        "Framework", "Param_Clients", "Param_Rounds", "Param_Epochs", "Param_Batch_Size", "Param_Seed", "Iteration", "Start_Time", "End_Time", "Raw_Duration_s", "Adjusted_Window_s", "Detected_Client_Count",
        "Server_CPU_Cores", "Server_Memory_Bytes", "Server_Net_Rx_Bps", "Server_Net_Tx_Bps", "Server_GPU_Util",
        "Clients_Avg_CPU_Cores", "Clients_Avg_Memory_Bytes", "Clients_Avg_Net_Rx_Bps", "Clients_Avg_Net_Tx_Bps", "Clients_Avg_GPU_Util",
        "Accuracies_Per_Round"
    ]

    def write_row(row_data):
        with open(CSV_FILENAME, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if file_needs_header:
                writer.writeheader()
            writer.writerow(row_data)

    def write_fallback_row(status_msg):
        """Writes a row utilizing the active configuration, but placing a status message in the metric fields."""
        write_row({
            "Framework": framework,
            "Param_Clients": cfg['clients'],
            "Param_Rounds": cfg['rounds'],
            "Param_Epochs": cfg['epochs'],
            "Param_Batch_Size": cfg['batch_size'],
            "Param_Seed": seed,
            "Iteration": iteration,
            "Start_Time": status_msg,
            "End_Time": status_msg,
            "Raw_Duration_s": status_msg,
            "Adjusted_Window_s": status_msg,
            "Detected_Client_Count": status_msg,
            "Server_CPU_Cores": status_msg,
            "Server_Memory_Bytes": status_msg,
            "Server_Net_Rx_Bps": status_msg,
            "Server_Net_Tx_Bps": status_msg,
            "Server_GPU_Util": status_msg,
            "Clients_Avg_CPU_Cores": status_msg,
            "Clients_Avg_Memory_Bytes": status_msg,
            "Clients_Avg_Net_Rx_Bps": status_msg,
            "Clients_Avg_Net_Tx_Bps": status_msg,
            "Clients_Avg_GPU_Util": status_msg,
            "Accuracies_Per_Round": status_msg
        })

    try:
        if framework == "flower":
            run_data = get_flower_run_stats()
            start_str = run_data["starting-at"]
            end_str = run_data["finished-at"]
            
            start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%SZ").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%SZ").replace(tzinfo=timezone.utc)
            
            accuracies = query_flower_accuracies(end_dt.timestamp())
            server_regex = "superlink|superexec-serverapp"
            
            def get_client_regex(cid):
                return f"supernode-{cid}|superexec-clientapp-{cid}"

        elif framework == "nvidiaFlare":
            run_data, accuracies = poll_nvflare_status(num_clients, target_rounds, start_dt_local)
            start_str = run_data["starting-at"]
            end_str = run_data["finished-at"]
            
            start_dt = run_data["start_dt"]
            end_dt = run_data["end_dt"]
            server_regex = NVFLARE_SERVER_CONTAINER_NAME
            
            def get_client_regex(cid):
                return f"site-{cid}"
                
        elif framework == "fedbiomed":
            run_data, accuracies = poll_fedbiomed_status(num_clients, target_rounds, start_dt_local)
            start_str = run_data["starting-at"]
            end_str = run_data["finished-at"]
            
            start_dt = run_data["start_dt"]
            end_dt = run_data["end_dt"]
            server_regex = FEDBIOMED_SERVER_CONTAINER_NAME
            
            def get_client_regex(cid):
                return f"fbm-node-{cid}"

        total_duration_s = max(int((end_dt - start_dt).total_seconds()), 1)
        
        if framework == "flower":
            adjusted_window_s = max(total_duration_s - SCRAPE_BUFFER_SECONDS_FLOWER, 1)
            adjusted_end_timestamp = end_dt.timestamp() - SCRAPE_BUFFER_SECONDS_FLOWER
        else:
            adjusted_window_s = total_duration_s
            adjusted_end_timestamp = end_dt.timestamp()

        print(f"[{framework.upper()}] Run complete. Raw: {total_duration_s}s. Metric Evaluation Window: {adjusted_window_s}s.")

        server_metrics = get_server_metrics(server_regex, adjusted_window_s, adjusted_end_timestamp)
        
        client_ids = [str(i) for i in range(1, num_clients + 1)]
        all_clients_metrics = {"cpu": 0.0, "memory": 0.0, "net_rx": 0.0, "net_tx": 0.0}
        
        for cid in client_ids:
            node_regex = get_client_regex(cid)
            node_metrics = get_node_container_metrics(node_regex, adjusted_window_s, adjusted_end_timestamp)
            
            for key in all_clients_metrics:
                all_clients_metrics[key] += node_metrics[key]
                
        clients_avg_metrics = {k: v / num_clients for k, v in all_clients_metrics.items()}

        gpu_query = build_global_gpu_query(adjusted_window_s)
        clients_avg_metrics["gpu"] = query_prometheus(gpu_query, adjusted_end_timestamp)

        write_row({
            "Framework": framework,
            "Param_Clients": cfg['clients'],
            "Param_Rounds": cfg['rounds'],
            "Param_Epochs": cfg['epochs'],
            "Param_Batch_Size": cfg['batch_size'],
            "Param_Seed": seed,
            "Iteration": iteration,
            "Start_Time": start_str,
            "End_Time": end_str,
            "Raw_Duration_s": total_duration_s,
            "Adjusted_Window_s": adjusted_window_s,
            "Detected_Client_Count": num_clients,
            "Server_CPU_Cores": round(server_metrics["cpu"], 4),
            "Server_Memory_Bytes": round(server_metrics["memory"], 2),
            "Server_Net_Rx_Bps": round(server_metrics["net_rx"], 2),
            "Server_Net_Tx_Bps": round(server_metrics["net_tx"], 2),
            "Server_GPU_Util": round(server_metrics["gpu"], 2),
            "Clients_Avg_CPU_Cores": round(clients_avg_metrics["cpu"], 4),
            "Clients_Avg_Memory_Bytes": round(clients_avg_metrics["memory"], 2),
            "Clients_Avg_Net_Rx_Bps": round(clients_avg_metrics["net_rx"], 2),
            "Clients_Avg_Net_Tx_Bps": round(clients_avg_metrics["net_tx"], 2),
            "Clients_Avg_GPU_Util": round(clients_avg_metrics["gpu"], 2),
            "Accuracies_Per_Round": json.dumps(accuracies)
        })

        print(f"Data logged to {CSV_FILENAME}\n" + "-"*50)

    except TimeoutError as e:
        print(f"\n[{framework.upper()}] WARNING: {e}")
        write_fallback_row("timeout")
        print(f"Timeout fallback registered in {CSV_FILENAME}\n" + "-"*50)
        
    except RuntimeError as e:
        print(f"\n[{framework.upper()}] ERROR: {e}")
        write_fallback_row("ERROR")
        print(f"Error fallback registered in {CSV_FILENAME}\n" + "-"*50)

def main():
    frameworks = ["fedbiomed", "nvidiaFlare", "flower"]
    configs = generate_experiment_matrix()
    seeds = [42, 69, 420]
    
    total_runs = len(frameworks) * len(configs) * len(seeds)
    current_run_idx = 1
    
    completed_runs = get_completed_runs(CSV_FILENAME)
    if completed_runs:
        print(f"Found {len(completed_runs)} previously completed runs in {CSV_FILENAME}. These will be skipped.")

    print(f"Starting experimental suite. {len(configs)} configurations across {len(seeds)} seeds mapped to {len(frameworks)} frameworks ({total_runs} total runs).")
    signatures = set()
    for framework in frameworks:
        for cfg in configs:
            for iteration, seed in enumerate(seeds, start=1):
                run_signature = (framework, cfg['clients'], cfg['rounds'], cfg['epochs'], cfg['batch_size'], seed)
                if run_signature in completed_runs:
                    print(f"\n[{current_run_idx}/{total_runs}] SKIPPING ALREADY COMPLETED...")
                    current_run_idx += 1
                    continue
                signatures.add(run_signature)
    
    shuffled_signatures = list(signatures)
    random.shuffle(shuffled_signatures)  
    
    for signature in shuffled_signatures:
        framework, clients, rounds, epochs, batch_size, seed = signature
        
        cfg = {
            'clients': clients, 
            'rounds': rounds, 
            'epochs': epochs, 
            'batch_size': batch_size
        }
        iteration = seeds.index(seed) + 1 

        print(f"\n[{current_run_idx}/{total_runs}] EXEC_CONFIG: Framework={framework}, Clients={clients}, Rounds={rounds}, Epochs={epochs}, Batch={batch_size}, Seed={seed} | Iteration {iteration}/3")
        
        if framework == "flower":
            start_script = FLOWER_START_SCRIPT
            stop_script = FLOWER_STOP_SCRIPT
        elif framework == "nvidiaFlare":
            start_script = NVFLARE_START_SCRIPT
            stop_script = NVFLARE_STOP_SCRIPT
        elif framework == "fedbiomed":
            start_script = FEDBIOMED_START_SCRIPT
            stop_script = FEDBIOMED_STOP_SCRIPT
            rounds += 1 # adding one because validation phase starts in the beggining of the round
        
        start_cmd = [
            start_script,
            "-c", str(clients),
            "-r", str(rounds),
            "-e", str(epochs),
            "-b", str(batch_size),
            "-s", str(seed)
        ]
        
        while True:
            try:
                print(f"Executing: {' '.join(start_cmd)}")
                subprocess.run(start_cmd, check=True)
                break
            except Exception as e:
                print(f"CRITICAL ERROR launching the framework: {e}")
                time.sleep(180)
        
        try:
            collect_and_save_run(framework, cfg, iteration, seed)
        except Exception as e:
            print(f"CRITICAL ERROR gathering stats during current run window: {e}")
        finally:
            print(f"Executing cleanup: {stop_script}")
            subprocess.run([stop_script], check=True)
            time.sleep(5)
            
        current_run_idx += 1

    print("\nAll benchmarking conditions have completed successfully! Results are saved in:", CSV_FILENAME)

if __name__ == "__main__":
    main()