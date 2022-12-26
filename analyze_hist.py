#!/usr/bin/env python

from glob import glob
import numpy as np
import matplotlib.pyplot as plt

def cmp():
    data_rgb = np.load("tmp/rgb_codes.npy")
    data_recon = np.concatenate([ np.load(p) for p in glob("tmp/code_*.npy")])

    data_rgb = set(data_rgb)
    data_recon = set(data_recon)
    print(len(data_rgb))
    print(len(data_recon))
    print(data_rgb - data_recon)

def rgb_code():
    data = np.load("rgb_codes.npy")
    print(data)
    print(data.shape)

def main():
    paths = sorted(glob("code_*.npy"))
    datas = []
    for path in paths:
        data = np.load(path)
        datas.append(data)

    data = np.concatenate(datas)
    hist, bin_edges = np.histogram(data, bins=np.arange(0,2049))
    names = [str(i) for i in bin_edges[:-1]]

    pairs = list(zip(names, hist))
    pairs = list(filter(lambda x: x[1] !=0, pairs))

    plt.rcParams["figure.figsize"] = (50, 5)
    plt.bar([i[0] for i in pairs], [i[1] for i in pairs])

    # plt.hist(data, np.arange(0, 2049))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    cmp()
