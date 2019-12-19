#!/bin/bash
echo "Running deep feature matching, getting features using Ground Truth annotations"
python run.py True
echo "Running deep feature matching, getting features using MTCNN face detection"
python run.py False
echo "Running deep feature matching, generating probe, gallery from ground truth"
python sample.py 
echo "Running deep feature matching, generating probe, gallery from MTCNN features"
python featuresMTCNN.py 
echo "Running deep feature matching, compute cosine similarity and ranking the matches"
python computeCosine.py
python eval.py
