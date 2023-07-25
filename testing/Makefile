# Set $ROOT to top-level directory of the repository
ROOT ?= $(shell dirname \
  $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST)))))

# Local image name and home
IMG_NAME=boptestgym
IMG_HOME=/home/developer/boptestgym

# BOPTEST commit used for the tests
BOPTEST_COMMIT=619d43cb39c1e6b19bd8c9c71beba259e4cca143

build-boptestgym:
	docker build -f ${ROOT}/testing/Dockerfile \
		--progress=plain --rm -t ${IMG_NAME} .

build-boptestgym-no-cache:
	docker build -f ${ROOT}/testing/Dockerfile \
		--progress=plain --no-cache --rm -t ${IMG_NAME}
	
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
		-it \
		${IMG_NAME} \
		/bin/bash -c "${ARGS} && exit"

push-boptestgym:
# requires `docker login` first
	docker tag boptestgym javierarroyo/boptestgym
	docker push javierarroyo/boptestgym

run-boptest-case:
	curl -L -o boptest.zip https://github.com/ibpsa/project1-boptest/archive/${BOPTEST_COMMIT}.zip
	unzip -o boptest.zip
	cd project1-boptest-${BOPTEST_COMMIT} && TESTCASE=bestest_hydronic_heat_pump docker-compose up -d

stop-boptest-case:
	cd project1-boptest-${BOPTEST_COMMIT} && docker-compose down

cleanup-boptest:
	rm boptest.zip
	rm -r project1-boptest-${BOPTEST_COMMIT}

# All tests except test_tutorial use the master version of boptest-gym
# Assumes bestest_hydronic_heat_pump test case is running
test-local:
	python3 -m unittest \
		test_boptestGymEnv.BoptestGymEnvTest.test_summary \
		test_boptestGymEnv.BoptestGymEnvTest.test_stable_baselines_check \
		test_boptestGymEnv.BoptestGymEnvTest.test_reset_fixed \
		test_boptestGymEnv.BoptestGymEnvTest.test_reset_random \
		test_boptestGymEnv.BoptestGymEnvTest.test_get_reward_default \
		test_boptestGymEnv.BoptestGymEnvTest.test_get_reward_custom \
		test_boptestGymEnv.BoptestGymEnvTest.test_get_reward_clipping \
		test_boptestGymEnv.BoptestGymEnvTest.test_normalized_observation_wrapper \
		test_boptestGymEnv.BoptestGymEnvTest.test_set_scenario \
		test_boptestGymEnv.BoptestGymEnvTest.test_A2C_simple \
		test_boptestGymEnv.BoptestGymEnvTest.test_A2C_A \
		test_boptestGymEnv.BoptestGymEnvTest.test_A2C_B \
		test_boptestGymEnv.BoptestGymEnvTest.test_A2C_C \
		test_boptestGymEnv.BoptestGymEnvTest.test_save_callback \
		test_boptestGymEnv.BoptestGymEnvTest.test_variable_episode 

# The tutorial is using boptest-gym-service and covers most of the functionality of boptest-gym 
test-service:
	python3 -m unittest test_boptestGymEnv.BoptestGymEnvTest.test_tutorial

test-local-in-container:
	make run-boptest-case
	make run-boptestgym-detached
	make exec-boptestgym ARGS="make test-local"
	make stop-boptestgym
	make stop-boptest-case
	make cleanup-boptest

test-service-in-container:
	make run-boptestgym-detached
	make exec-boptestgym ARGS="make test-service"
	make stop-boptestgym

