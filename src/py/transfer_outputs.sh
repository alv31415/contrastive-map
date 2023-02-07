#!/bin/bash

source ~/.bashrc

USER=$(whomai)

SOURCE_DIR=s${STUDENT_ID}@mlp.inf.ed.ac.uk:/home/s${STUDENT_ID}/honours-project/contrastive-map/src/py/output
DESTINATION_DIR=/Users/${USER}/Desktop/ED/ED4/Honours\ Project/src/py/output

echo ${DESTINATION_DIR}

echo "Transferring files from ${SOURCE_DIR} to ${DESTINATION_DIR}"
rsync --archive --update --compress --progress ${SOURCE_DIR}/ ${DESTINATION_DIR}