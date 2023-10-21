import socket
import yaml


num_services = 5  # Total Services needed
base_port = 5000  # Start Port number


# Function to check if a port is available
def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(('localhost', port)) != 0


services = {}
for i in range(num_services):
    port = base_port + i

    # If the port is unavailable, continue to check the next one
    while not is_port_available(port):
        print(f"Port {port} is occupied.")
        port += 1
        if port > base_port + num_services:
            raise Exception("Too many ports are occupied.")

    service_name = f"boptest{i}"
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

with open("docker-compose.yml", "w") as file:
    yaml.dump(docker_compose_content, file, default_flow_style=False)



