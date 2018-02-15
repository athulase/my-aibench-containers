#!/bin/bash

#delete output so new run doesn't use existing features/match data
rm -f ${ROMAN_PATH}/output/*
rm -f ${ROMAN_PATH}/images/features/*

/theia-build/bin/build_reconstruction --flagfile=/flagfile.txt &> ${ROMAN_PATH}/output/output.log
/theia-build/bin/write_reconstruction_ply_file --reconstruction=${ROMAN_PATH}/output/output-0 --ply_file=${ROMAN_PATH}/output/output-0.ply &>> ${ROMAN_PATH}/output/output.log
/theia-build/bin/write_reconstruction_ply_file --reconstruction=${ROMAN_PATH}/output/output-1 --ply_file=${ROMAN_PATH}/output/output-1.ply &>> ${ROMAN_PATH}/output/output.log
