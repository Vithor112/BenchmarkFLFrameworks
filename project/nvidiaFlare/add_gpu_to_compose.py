import yaml
import sys
import os
import copy

def add_gpu_to_compose(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    with open(file_path, 'r') as f:
        try:
            compose_data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML: {exc}")
            return

    if not compose_data or 'services' not in compose_data:
        print("No services found in compose file.")
        return

    gpu_deploy_config = {
        'resources': {
            'reservations': {
                'devices': [
                    {
                        'count': 'all',
                        'driver': 'nvidia',
                        'capabilities': ['gpu']
                    }
                ]
            }
        }
    }
    services = {
        "prometheus": {
            "image": "prom/prometheus:latest",
            "container_name": "prometheus",
            "ports": ["9090:9090"],
            "command": ["--config.file=/etc/prometheus/prometheus.yml"],
            "volumes": ["../../../prometheus.yml:/etc/prometheus/prometheus.yml:ro"],
            "depends_on": ["cadvisor", "dcgm_exporter"],
        },
        "cadvisor": {
            "image": "gcr.io/cadvisor/cadvisor:latest",
            "container_name": "cadvisor",
            "ports": ["8080:8080"],
            "volumes": [
                "/:/rootfs:ro",
                "/var/run:/var/run:rw",
                "/sys:/sys:ro",
                "/var/lib/docker/:/var/lib/docker:ro"
            ]
        },
        "dcgm_exporter": {
            "image": "nvidia/dcgm-exporter:4.5.2-4.8.1-ubuntu22.04",
            "container_name": "dcgm_exporter",
            "cap_add": ["SYS_ADMIN"],
            "deploy": {
                "resources": {
                    "reservations": {
                        "devices": [
                            {
                                "driver": "nvidia",
                                "count": "all",
                                "capabilities": ["gpu"]
                            }
                        ]
                    }
                }
            },
            "ports": ["9400:9400"],
            "command": "-f /etc/dcgm-exporter/dcp-metrics-included.csv"
        }
    }

    for service_name, service_config in compose_data['services'].items():
        service_config['deploy'] = copy.deepcopy(gpu_deploy_config)
    for new_service_name, new_service_config in services.items():
        compose_data['services'][new_service_name] = new_service_config
    with open(file_path, 'w') as f:
        yaml.dump(compose_data, f, default_flow_style=False, sort_keys=False)
        print(f"Successfully added GPU configuration to {file_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        add_gpu_to_compose(sys.argv[1])
    else:
        print("Usage: python add_gpu_to_compose.py <path_to_compose_yaml>")
