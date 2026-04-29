import subprocess
import os
import sys
import time


if len(sys.argv) != 2:
    print("Usage: python launch.py <number_of_nodes>")
    sys.exit(1)

num_nodes = int(sys.argv[1])

subprocess.run(["sudo", "nft", "delete", "table", "ip", "eset_eea_wap"])
subprocess.run(["docker", "compose", "up", "-d"])
for i in range(num_nodes):
    print(f"Creating node {i+1}...")
    node_dir = f"./fbm-node_{i+1}"
    
    subprocess.run(["fedbiomed", "component", "create", "-c", "node", "-p", node_dir])
    subprocess.run(["fedbiomed", "node", "-p", node_dir, "dataset", "add", "--mnist"])
    
    abs_node_dir = os.path.abspath(node_dir)
    
    subprocess.run([
        "docker", "run", "-d", 
        "--name", f"fbm-node-{i+1}", "--rm", 
        "--gpus", "all", "--network", "host",
        "-v", f"{abs_node_dir}:/app/fbm-node", 
        "-v", f"{abs_node_dir}/data:{abs_node_dir}/data",
        "fbm-node:6.2.0"
    ])
time.sleep(5)
print("Creating researcher")
subprocess.run(["fedbiomed", "component", "create", "-c", "researcher", "-p", "./fbm-researcher"])
abs_researcher_dir = os.path.abspath("./fbm-researcher")
abs_researcher_script = os.path.abspath("researcher.py")
subprocess.run([
    "docker", "run", "-d", 
    "--name", f"fbm-researcher", "--rm", "--network", "host",
    "-v", f"{abs_researcher_dir}:/app/fbm-researcher", 
    "-v", f"{abs_researcher_script}:/app/researcher.py",
    "fbm-researcher:6.2.0"
])
subprocess.run(["python", "researcher.py"])

print(f"Cleaning up folders...")
for file in os.listdir("."):
    if file.startswith("fbm-node_") and os.path.isdir(file):
        subprocess.run(["sudo", "rm", "-rf", file])
    elif file == "fbm-researcher" and os.path.isdir(file):
        subprocess.run(["sudo", "rm", "-rf", file])

result = subprocess.run(
    ["docker", "ps", "-q", "--filter", "name=fbm-node-"],
    capture_output=True,
    text=True
)
container_ids = result.stdout.split()
if container_ids:
    subprocess.run(["docker", "stop"] + container_ids)
else:
    print("No matching containers found.")
subprocess.run(["docker", "compose", "down"])