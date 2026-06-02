import subprocess
import os

def stop():

    result_nodes = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "name=fbm-node-"],
        capture_output=True,
        text=True
    )
    node_ids = result_nodes.stdout.split()
    if node_ids:
        print(f"Stopping and removing node containers...")
        subprocess.run(["docker", "stop"] + node_ids)
        subprocess.run(["docker", "rm"] + node_ids)
    else:
        print("No matching node containers found.")

    result_researcher = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "name=fbm-researcher"],
        capture_output=True,
        text=True
    )
    researcher_ids = result_researcher.stdout.split()
    if researcher_ids:
        print(f"Stopping and removing researcher containers...")
        subprocess.run(["docker", "stop"] + researcher_ids)
        subprocess.run(["docker", "rm"] + researcher_ids)
    else:
        print("No matching researcher containers found.")
        
    subprocess.run(["docker", "compose", "down"])

    print(f"Cleaning up folders...")
    for file in os.listdir("."):
        if file.startswith("fbm-node_") and os.path.isdir(file):
            subprocess.run(["rm", "-rf", file])
        elif file == "fbm-researcher" and os.path.isdir(file):
            subprocess.run(["rm", "-rf", file])


if __name__ == "__main__":
    stop()