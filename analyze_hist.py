#!/usr/bin/env python

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pycomar.io import save, load


def cmp():
    data_rgb = np.load("tmp/rgb_codes.npy")
    data_recon = np.concatenate([np.load(p) for p in glob("tmp/code_*.npy")])

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
    hist, bin_edges = np.histogram(data, bins=np.arange(0, 2049))
    names = [str(i) for i in bin_edges[:-1]]

    pairs = list(zip(names, hist))
    pairs = list(filter(lambda x: x[1] != 0, pairs))

    plt.rcParams["figure.figsize"] = (50, 5)
    plt.bar([i[0] for i in pairs], [i[1] for i in pairs])

    # plt.hist(data, np.arange(0, 2049))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def mkhist_imgnet16384():
    hist_acc = {}
    for i in range(0, 16384):
        hist_acc[str(i)] = 0

    for p in tqdm(
            sorted(glob("experiments/code-hist_imgnet16384/npys/code_*.npy"))):
        data = np.load(p)
        hist, bin_edges = np.histogram(data, bins=np.arange(0, 16385))
        labels = [str(i) for i in bin_edges[:-1]]
        pairs = zip(labels, hist)
        for label, cnt in zip(labels, hist):
            hist_acc[label] += cnt

    save(hist_acc, "experiments/code-hist_imgnet16384/histogram.pkl")


def mkhist_imgnet1024():
    hist_acc = {}
    for i in range(0, 1024):
        hist_acc[str(i)] = 0

    for p in tqdm(
            sorted(glob("experiments/code-hist_imgnet1024/npys/code_*.npy"))):
        data = np.load(p)
        hist, bin_edges = np.histogram(data, bins=np.arange(0, 1025))
        labels = [str(i) for i in bin_edges[:-1]]
        for label, cnt in zip(labels, hist):
            hist_acc[label] += cnt

    save(hist_acc, "experiments/code-hist_imgnet1024/histogram.pkl")


def viewhist_imgnet1024():
    hist: dict = load("experiments/code-hist_imgnet1024/histogram.pkl")
    acc = sum([i[1] for i in hist.items()])
    num_zerovisit = len(list(filter(lambda x: x[1] == 0, hist.items())))
    print("total count:", acc)
    print("visit:", 1024 - num_zerovisit)
    print("zero-visit:", num_zerovisit)


def viewhist_imgnet16384():
    hist: dict = load("experiments/code-hist_imgnet16384/histogram.pkl")
    acc = sum([i[1] for i in hist.items()])
    print("total count:", acc)
    num_zerovisit = len(list(filter(lambda x: x[1] == 0, hist.items())))
    print("visit:", 16384 - num_zerovisit)
    print("zero-visit:", num_zerovisit)


if __name__ == "__main__":
    # mkhist_imgnet16384()
    viewhist_imgnet1024()
    viewhist_imgnet16384()
