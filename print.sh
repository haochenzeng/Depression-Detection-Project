#!/bin/bash

grep 'Number of neurons:' ResNet_output.txt |tail -1

python3 extract_spikes.py \ --file_path = "/Users/mac/PycharmProjects/ResNet/ResNet_output.txt" \
--n_neurons = 22022208 \
--n_images = 10000