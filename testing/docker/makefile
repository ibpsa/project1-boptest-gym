IMG_NAME=boptestgymtesting

build:
	docker build --no-cache=false --rm -t ${IMG_NAME} .

remove-image:
	docker rmi ${IMG_NAME}

run:
	docker run --name ${IMG_NAME} -it --rm --detach=false ${IMG_NAME} /bin/bash

stop:
	docker stop ${IMG_NAME}

# For pushing first it is needed to login: docker login --username=yourhubusername
push:
	docker tag ${IMG_NAME} javierarroyo/${IMG_NAME} && docker push javierarroyo/${IMG_NAME} 