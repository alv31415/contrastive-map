#!/bin/bash

source ~/.bashrc

SOURCE_DIR=s${STUDENT_ID}@mlp.inf.ed.ac.uk:/home/${STUDENT_ID}/honours-project/contrastive-map/src/py/output
DESTINATION_DIR=output

echo ${DESTINATION_DIR}

echo "Transferring files from ${SOURCE_DIR} to ${DESTINATION_DIR}"
scp -r ${SOURCE_DIR}/ ${DESTINATION_DIR}