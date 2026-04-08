import yaml
import sys
import subprocess


def is_port_in_use(port):
    """Checks if a port is already in use on the host."""
    result = subprocess.run(f"netstat -lant | grep {port}", capture_output=True, shell=True)
    return result.stdout != b''

def generate_flower_compose(num_nodes):
    total_partitions = 200
    start_port = 9094
    current_port = start_port
    
    compose_dict = {
        "services": {
            "superlink": {
                "image": "flwr/superlink:1.28.0",
                "container_name": "superlink",
                "ports": ["9091:9091", "9092:9092", "9093:9093"],
                "command": ["--insecure", "--isolation", "process"],
                "networks": ["flwr-network"]
            },
            "superexec-serverapp": {
                "build": {"context": ".", "dockerfile": "superexec.Dockerfile"},
                "image": "flwr_superexec:0.0.1",
                "container_name": "superexec-serverapp",
                "depends_on": ["superlink"],
                "command": ["--insecure", "--plugin-type", "serverapp", "--appio-api-address", "superlink:9091"],
                "networks": ["flwr-network"],
                "environment": ["FLWR_LOG_LEVEL=DEBUG"]
            }
        },
        "networks": {
            "flwr-network": {"driver": "bridge"}
        }
    }

    nodes_created = 0
    while nodes_created < num_nodes:
        if is_port_in_use(current_port):
            # Thank you copilot for the logs with emojis
            print(f"⚠️ Port {current_port} is in use, skipping...")
            current_port += 1
            continue

        nodes_created += 1
        partition_id = nodes_created - 1
        node_name = f"supernode-{nodes_created}"
        client_name = f"superexec-clientapp-{nodes_created}"

        compose_dict["services"][node_name] = {
            "image": "flwr/supernode:1.28.0",
            "container_name": node_name,
            "ports": [f"{current_port}:{current_port}"],
            "depends_on": ["superlink"],
            "command": [
                "--insecure",
                "--superlink", "superlink:9092",
                "--node-config", f"partition-id={partition_id} num-partitions={total_partitions}",
                "--clientappio-api-address", f"0.0.0.0:{current_port}",
                "--isolation", "process"
            ],
            "networks": ["flwr-network"],
            "environment": ["FLWR_LOG_LEVEL=DEBUG"]
        }

        compose_dict["services"][client_name] = {
            "build": {"context": ".", "dockerfile": "superexec.Dockerfile"},
            "image": "flwr_superexec:0.0.1",
            "container_name": client_name,
            "depends_on": [node_name],
            "command": [
                "--insecure",
                "--plugin-type", "clientapp",
                "--appio-api-address", f"{node_name}:{current_port}"
            ],
            "networks": ["flwr-network"],
            "volumes": ["./MedNIST:/app/MedNIST"],
            "environment": ["FLWR_LOG_LEVEL=DEBUG"]
        }
        
        current_port += 1

    with open('docker-compose.yml', 'w') as f:
        yaml.dump(compose_dict, f, sort_keys=False, default_flow_style=False)
    
    print(f"✅ Successfully generated docker-compose.yml with {num_nodes} nodes.")

if __name__ == "__main__":
    try:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
        generate_flower_compose(n)
    except ValueError:
        print("Error: Please provide an integer for the number of nodes.")