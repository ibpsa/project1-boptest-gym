import os
import socket
import shutil
import yaml
import sys



if len(sys.argv) >= 2:
    # Check if there is an argument when calling the script defining the yaml target directory
    # This is just convenient for dropping the generated script directly in the BOPTEST root directory,
    # but if not provided it can be manually moved once it's generated
    yaml_target_dir = sys.argv[1]
else:
    # Otherwise use parent directory as default
    yaml_target_dir = os.path.dirname(os.path.abspath(__file__))

num_services = 2  # Total Services needed
base_port = 5000  # Start Port number


# Function to check if a port is available
def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(('localhost', port)) != 0


services = {}
last_assigned_port = base_port - 1  # Initial value set to one less than the first port to be checked
for _ in range(num_services):
    port = last_assigned_port + 1  # Start checking from the next port after the last assigned one

    # If the port is unavailable, continue to check the next one
    while not is_port_available(port):
        print(f"Port {port} is occupied.")
        port += 1
        if port > base_port + num_services:
            raise Exception("Too many ports are occupied.")

    last_assigned_port = port  # Update the last assigned port

    service_name = f"boptest{port}"
    service_config = {
        "image": "boptest_base",
        "build": {"context": "."},
        "volumes": [
            "./testcases/${TESTCASE}/models/wrapped.fmu:${APP_PATH}/models/wrapped.fmu",
            "./testcases/${TESTCASE}/doc/:${APP_PATH}/doc/",
            "./restapi.py:${APP_PATH}/restapi.py",
            "./testcase.py:${APP_PATH}/testcase.py",
            "./version.txt:${APP_PATH}/version.txt",
            "./data:${APP_PATH}/data/",
            "./forecast:${APP_PATH}/forecast/",
            "./kpis:${APP_PATH}/kpis/",
        ],
        "ports": [f"127.0.0.1:{port}:5000"],
        "networks": ["boptest-net"],
        "restart": "on-failure"  # restart on-failure
    }
    services[service_name] = service_config

docker_compose_content = {
    "version": "3.7",
    "services": services,
    "networks": {
        "boptest-net": {
            "name": "boptest-net",
            "attachable": True,
        }
    },
}

# Check whether the docker-compose.yml file exists in the BOPTEST root directory
docker_compose_path = os.path.join(yaml_target_dir, 'docker-compose.yml')
if os.path.exists(docker_compose_path):
    # If it exists, rename to docker-compose_origin.yml
    shutil.move(docker_compose_path, os.path.join(yaml_target_dir, 'docker-compose_origin.yml'))

# Create a new docker-compose.yml file in the BOPTEST root directory
with open(docker_compose_path, "w") as file:
    yaml.dump(docker_compose_content, file, default_flow_style=False)
