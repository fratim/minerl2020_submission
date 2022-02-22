#!/bin/bash
# Expected Env variables : in environ.sh

REPO2DOCKER="$(which aicrowd-repo2docker)"

echo REPO2DOCKER

#sudo ${REPO2DOCKER} --no-run \
#  --user-id 1001 \
#  --user-name aicrowd \
#  --image-name minerl_image:v1 \
#  --debug .