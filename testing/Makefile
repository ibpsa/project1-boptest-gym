# Set $ROOT to top-level directory of the repository
ROOT ?= $(shell dirname \
  $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST)))))

# Local image name and home
IMG_NAME=boptestgym
IMG_HOME=/home/developer/boptestgym

build:
	docker build -f ${ROOT}/testing/Dockerfile \
		--progress=plain --rm -t ${IMG_NAME} .

build-no-cache:
	docker build -f ${ROOT}/testing/Dockerfile \
		--progress=plain --no-cache --rm -t ${IMG_NAME}
	
run:
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

run-detached:
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

stop:
	docker stop ${IMG_NAME}

exec-interactive:
	docker exec \
		-it \
		${IMG_NAME} \
		/bin/bash -c "${ARGS} && exit"

# All tests except test_tutorial use the master version of boptest-gym
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

