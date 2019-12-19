#!/bin/bash
echo "Running celebrity matching, generating probe features"
python run_celeb.py True
echo "Running celebrity matching, generating gallery features"
python run_celeb.py False
echo "Running celebrity matching, to compute scores"
python celebAMatching.py
echo "Running celebrity matching, ranking the matches"
python eval.py