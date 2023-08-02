#!/bin/bash
# run froc.py
TYPE="2"
GT_CSV="/home/stat-zx/CTC_data/test_pos.csv"


python froc.py \
    --type $TYPE \
    $GT_CSV
    
