# Set $ROOT to top-level directory of the repository
ROOT ?= $(shell dirname \
  $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST)))))

# Local image name and home
IMG_NAME=boptestgym
IMG_HOME=/home/developer/boptestgym

# Name of remote registry image
IMG_REGI=javierarroyo/boptestgym

# BOPTEST commit used for the tests
BOPTEST_COMMIT=78665506a620fc07bc2a8f301661aef3dd6c88c2

# Define current BOPTEST-Gym version (should be even with BOPTEST version defined in commit above)
VERSION = 0.6.0

build-boptestgym:
	docker build -f ${ROOT}/testing/Dockerfile \
		--progress=plain --rm -t ${IMG_NAME} .

build-boptestgym-no-cache:
	docker build -f ${ROOT}/testing/Dockerfile \
		--progress=plain --no-cache --rm -t ${IMG_NAME} .
	
run-boptestgym:
	docker run \
		--name ${IMG_NAME} \
		--detach=false \
		--network=host \
		--rm \
		--user $(id -u):$(id -g) \
		-v ${ROOT}:${IMG_HOME}:rw \
		-w ${IMG_HOME}/testing \
		-it \
		${IMG_NAME}

run-boptestgym-detached:
	docker run \
		--name ${IMG_NAME} \
		--detach=true \
		--network=host \
		--rm \
		--user $(id -u):$(id -g) \
		-v ${ROOT}:${IMG_HOME}:rw \
		-w ${IMG_HOME}/testing \
		-it \
		${IMG_NAME}

stop-boptestgym:
	docker stop ${IMG_NAME}

exec-boptestgym:
	docker exec \
		-i \
		${IMG_NAME} \
		/bin/bash -c "${ARGS} && exit"

push-boptestgym:
# requires `docker login` first
	docker tag ${IMG_NAME} ${IMG_REGI}:${VERSION}
	docker push ${IMG_REGI}:${VERSION}

pull-boptestgym:
	docker pull ${IMG_REGI}:${VERSION}
	docker tag ${IMG_REGI}:${VERSION} ${IMG_NAME}

make download-boptest:
	curl -L -o boptest.zip https://github.com/ibpsa/project1-boptest/archive/${BOPTEST_COMMIT}.zip
	unzip -o -q boptest.zip
	
run-boptest-case:
	make download-boptest
	cd project1-boptest-${BOPTEST_COMMIT} && \
	TESTCASE=bestest_hydronic_heat_pump docker-compose up -d --quiet-pull

run-boptest-case-no-cache:
	make download-boptest
	cd project1-boptest-${BOPTEST_COMMIT} && \
	TESTCASE=bestest_hydronic_heat_pump docker-compose up -d --force-recreate --build

run-boptest-vectorized:
	make download-boptest && \
	cd .. && python3 generateDockerComposeYml.py testing/project1-boptest-${BOPTEST_COMMIT} && \
	cd testing/project1-boptest-${BOPTEST_COMMIT} && \
	TESTCASE=bestest_hydronic_heat_pump docker-compose up -d --quiet-pull

stop-boptest-case:
	cd project1-boptest-${BOPTEST_COMMIT} && docker-compose down

cleanup-boptest:
	rm boptest.zip
	rm -r project1-boptest-${BOPTEST_COMMIT}

# All tests except test_tutorial use the master version of boptest-gym
# Assumes bestest_hydronic_heat_pump test case is running
test-local:
	python3 -m unittest test_boptestGymEnv.BoptestGymEnvTest

# Vectorized needs to run separate since modifies docker-compose.yml to have multiple boptest instances 
test-vectorized:
	python3 -m unittest test_boptestGymEnv.BoptestGymVecTest project1-boptest-${BOPTEST_COMMIT}

# The tutorial is using boptest-gym-service and covers most of the functionality of boptest-gym 
test-service:
	python3 -m unittest test_boptestGymEnv.BoptestGymServiceTest

test-local-in-container:
	make run-boptest-case
	make run-boptestgym-detached
	make exec-boptestgym ARGS="make test-local"
	make stop-boptestgym
	make stop-boptest-case
	make cleanup-boptest

test-vectorized-in-container:
	make run-boptest-vectorized
	make run-boptestgym-detached
	make exec-boptestgym ARGS="make test-vectorized"
	make stop-boptestgym
	make stop-boptest-case
	make cleanup-boptest

test-service-in-container:
	make run-boptestgym-detached
	make exec-boptestgym ARGS="make test-service"
	make stop-boptestgym

