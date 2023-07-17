# Set $ROOT to top-level directory of the repository
ROOT ?= $(shell dirname \
  $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST)))))

# Local image name and home
IMG_NAME=boptestgym
IMG_HOME=/home/developer/boptestgym

build:
	docker build -f ${ROOT}/testing/Dockerfile \
		--platform=linux/amd64 --progress=plain --rm -t ${IMG_NAME} .

build-no-cache:
	docker build -f ${ROOT}/testing/Dockerfile \
		--platform=linux/amd64 --progress=plain --no-cache --rm -t ${IMG_NAME}
	
run:
	docker run \
		--platform=linux/amd64 \
		--name ${IMG_NAME} \
		--detach=false \
		--rm \
		-v ${ROOT}:${IMG_HOME} \
		-w ${IMG_HOME}/testing \
		-it \
		-p 127.0.0.1:5000:5000 \
		${IMG_NAME}

run-detached:
	docker run \
		--platform=linux/amd64 \
		--name ${IMG_NAME} \
		--detach=true \
		--rm \
		-v ${ROOT}:${IMG_HOME} \
		-w ${IMG_HOME}/testing \
		-it \
		-p 127.0.0.1:5000:5000 \
		${IMG_NAME}

stop:
	docker stop ${IMG_NAME}

exec-interactive:
	docker exec \
		-it \
		${IMG_NAME} \
		/bin/bash -c "${ARGS} && exit"
