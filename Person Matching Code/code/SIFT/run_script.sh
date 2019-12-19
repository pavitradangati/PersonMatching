#!/bin/bash
echo "Running sift feature matching, getting features using Ground Truth annotations"
python sift_run.py True
echo "Running sift feature matching, getting probe, gallery"
python generatePG_SIFT.py False
echo "Running sift feature matching,get scores and rank the matches"
python matching.py False
