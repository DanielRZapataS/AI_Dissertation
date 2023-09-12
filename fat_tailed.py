import numpy as np
import pandas as pd
import powerlaw
import argparse

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='VAE CONV1')
args = parser.parse_args()
model = args.model

gen_dir = 'generated_data/'
vaeconv1 = np.load(gen_dir + 'vaeconv1.npy')
vaeconv3 = np.load(gen_dir + 'vaeconv3.npy')
vaefd1 = np.load(gen_dir + 'vaefd1.npy')
vaefd2 = np.load(gen_dir + 'vaefd2.npy')
vqvaeconv2 = np.load(gen_dir + 'vqvaeconv2.npy')
vqvaefd1 = np.load(gen_dir + 'vqvaefd1.npy')
timevaebase = np.load(gen_dir + 'timevaebase.npy')


generated = {
            # 'VQ VAE CONV2':vqvaeconv2,
            'VAE CONV1 ':vaeconv1,
             'VAE CONV3':vaeconv3,
             'VAE FD1':vaefd1,
             'VAE FD2':vaefd2,
             'VQ VAE FD1':vqvaefd1,
            'TIME VAE BASE':timevaebase
             }

# run main 
if __name__ == '__main__':
    v = generated[model]
    data = v.flatten()
    # data = remove_outliers(data)
    results = powerlaw.Fit(data)
    # print(data.shape)
    alpha = results.power_law.alpha
    sigma = results.power_law.sigma
    print('Model: ', model)
    print('Alpha: ', alpha)
    print('Sigma: ', sigma)
