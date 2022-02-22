import re
import numpy as np

def extract_spikes(file_path, n_neurons, n_images):
    with open(file_path,'r' ) as f:
        text = f.read()

    spikes = []
    num = 0
    for line in text.split('\n'):
        if line.startswith('S') and line.endswith('0'):
            # print(int(float(line[7:-1])))
            spikes.append(int(float(line[7:-1])))
            num += int(float(line[7:-1]))

    # print(num)
    # print(np.sum(spikes))
    average_n_spikes = num / n_neurons / n_images
    print(f'Total number of spikes: {num}')
    print(f'Average number of spikes: {average_n_spikes}')

    pass

if __name__ == '__main__':
    print(f'number of neurons:{39424}')
    extract_spikes("resnet_n_spikes.txt", n_neurons= 39424, n_images= 10000)