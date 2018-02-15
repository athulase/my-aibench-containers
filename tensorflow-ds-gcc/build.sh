#!/bin/bash

DOCKER_REPO=10.105.15.44:5000
DATE=15022018154805
VERSIONING_REPO_ACTIVE=active
VERSIONING_REPO=github.com/athulase/my-aibench-versions.git
VERSIONING_REPO_PROTOCOL=https
VERSIONING_REPO_BRANCH=master
STANDALONE=False

###############################################################################

cd `dirname $0`

PROJECT=${PWD##*/}
UBID=`date +%s%N` # unique build id

docker build -t $UBID/$PROJECT .

if [ $? -ne 0 ]; then
    echo "Error running docker build"
    exit 1
else
    IMAGEID=`docker images | grep $UBID/$PROJECT | awk '{print $3}'`

    if [ $STANDALONE = "True" ]; then
        docker tag $IMAGEID $PROJECT:$DATE
        sleep 2
    else
        docker tag $IMAGEID $DOCKER_REPO/$PROJECT:$DATE
        sleep 2
        docker push $DOCKER_REPO/$PROJECT:$DATE
        sleep 2
    fi

    if [ $VERSIONING_REPO_ACTIVE = "active" ]; then
        VERSIONING_REPO_USERNAME=`cat versioning-repo.credentials | head -n 1`
        VERSIONING_REPO_PASSWORD=`cat versioning-repo.credentials | tail -n 1`
        cd /tmp
        git clone -b $VERSIONING_REPO_BRANCH $VERSIONING_REPO_PROTOCOL://$VERSIONING_REPO_USERNAME:$VERSIONING_REPO_PASSWORD@$VERSIONING_REPO versioning_repo
        cd versioning_repo
        echo $DOCKER_REPO/$PROJECT:$DATE > $PROJECT.version
        git add .
        git commit -m "$PROJECT version $DATE"
        git push -u origin $BRANCH
    fi


    echo "###############################################################################"
    if [ $STANDALONE = "True" ]; then
        echo "Image link: $PROJECT:$DATE"
    else
        echo "Image link: $DOCKER_REPO/$PROJECT:$DATE"
    fi
    echo "###############################################################################"
fi