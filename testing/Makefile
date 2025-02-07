# Set $ROOT to top-level directory of the repository
ROOT ?= $(shell dirname \
  $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST)))))

# Local image name, home directory, and remote registry for BOPTEST-Gym
IMG_NAME_BOPTESTGYM=boptestgym
IMG_HOME_BOPTESTGYM=/home/developer/boptestgym
IMG_REGI_BOPTESTGYM=javierarroyo/boptestgym

# Local image name and remote registry for the BOPTEST test case
IMG_NAME_BOPTESTBASE=boptest_base
IMG_REGI_BOPTESTBASE=javierarroyo/boptest_base

# BOPTEST commit used for the tests
BOPTEST_COMMIT=e7602c766932ca140459609a66e6b999bbb2bed7

# Define current BOPTEST-Gym version (should be even with BOPTEST version defined in commit above)
VERSION = 0.7.0

build-boptestgym:
	docker build -f ${ROOT}/testing/Dockerfile \
		--progress=plain --rm -t ${IMG_NAME_BOPTESTGYM} .

build-boptestgym-no-cache:
	docker build -f ${ROOT}/testing/Dockerfile \
		--progress=plain --no-cache --rm -t ${IMG_NAME_BOPTESTGYM} .

# Build all needed services without selecting any test case
build-boptestbase:
	make download-boptest
	cd project1-boptest-${BOPTEST_COMMIT} && \
	docker compose build

run-boptestgym:
	docker run \
		--name ${IMG_NAME_BOPTESTGYM} \
		--detach=false \
		--network=host \
		--rm \
		--user $(id -u):$(id -g) \
		-v ${ROOT}:${IMG_HOME_BOPTESTGYM}:rw \
		-w ${IMG_HOME_BOPTESTGYM}/testing \
		-it \
		${IMG_NAME_BOPTESTGYM}

run-boptestgym-detached:
	docker run \
		--name ${IMG_NAME_BOPTESTGYM} \
		--detach=true \
		--network=host \
		--rm \
		--user $(id -u):$(id -g) \
		-v ${ROOT}:${IMG_HOME_BOPTESTGYM}:rw \
		-w ${IMG_HOME_BOPTESTGYM}/testing \
		-it \
		${IMG_NAME_BOPTESTGYM}

stop-boptestgym:
	docker stop ${IMG_NAME_BOPTESTGYM}

exec-boptestgym:
	docker exec \
		-i \
		${IMG_NAME_BOPTESTGYM} \
		/bin/bash -c "${ARGS} && exit"

push-boptestgym:
# requires `docker login` first
	docker tag ${IMG_NAME_BOPTESTGYM} ${IMG_REGI_BOPTESTGYM}:${VERSION}
	docker push ${IMG_REGI_BOPTESTGYM}:${VERSION}

pull-boptestgym:
	docker pull ${IMG_REGI_BOPTESTGYM}:${VERSION}
	docker tag ${IMG_REGI_BOPTESTGYM}:${VERSION} ${IMG_NAME_BOPTESTGYM}

push-boptestbase:
# requires `docker login` first
	docker tag ${IMG_NAME_BOPTESTBASE} ${IMG_REGI_BOPTESTBASE}:${VERSION}
	docker push ${IMG_REGI_BOPTESTBASE}:${VERSION}

pull-boptestbase:
	docker pull ${IMG_REGI_BOPTESTBASE}:${VERSION}
	docker tag ${IMG_REGI_BOPTESTBASE}:${VERSION} ${IMG_NAME_BOPTESTBASE}

download-boptest:
	curl -L -o boptest.zip https://github.com/ibpsa/project1-boptest/archive/${BOPTEST_COMMIT}.zip
	unzip -o -q boptest.zip
	
run-boptest:
	make download-boptest
	cd project1-boptest-${BOPTEST_COMMIT} && \
	docker compose up -d web worker provision && \
	sleep 10

restart-boptest-app:
	cd project1-boptest-${BOPTEST_COMMIT} && \
	docker compose restart

run-boptest-vectorized:
	make download-boptest && \
	cd project1-boptest-${BOPTEST_COMMIT} && \
	docker compose up -d --scale worker=2 web worker provision && \
	sleep 10 

stop-boptest-case:
	cd project1-boptest-${BOPTEST_COMMIT} && docker compose down

cleanup-boptest:
	-rm boptest.zip
	-rm -r project1-boptest-${BOPTEST_COMMIT}

# Assumes bestest_hydronic_heat_pump test case is running
test-local:
	python3 -m unittest test_boptestGymEnv.BoptestGymEnvTest

test-multiaction:
	python3 -m unittest test_boptestGymEnv.BoptestGymEnvMultiActTest

# Vectorized needs to run separate since modifies docker-compose.yml to have multiple boptest instances 
test-vectorized:
	python3 -m unittest test_boptestGymEnv.BoptestGymVecTest project1-boptest-${BOPTEST_COMMIT}

# The tutorial covers most of the functionality of boptest-gym 
test-tutorial:
	python3 -m unittest test_boptestGymEnv.BoptestGymTutorialTest

test-local-in-container:
	make run-boptest
	make run-boptestgym-detached
	make exec-boptestgym ARGS="make test-local"
	make stop-boptestgym
	make stop-boptest-case
	make cleanup-boptest

test-multiaction-in-container:
	make run-boptest
	make run-boptestgym-detached
	make exec-boptestgym ARGS="make test-multiaction"
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

test-tutorial-in-container:
	make run-boptestgym-detached
	make exec-boptestgym ARGS="make test-tutorial"
	make stop-boptestgym

