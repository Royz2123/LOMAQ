import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import argparse
import glob
from itertools import cycle
import os
from scipy.ndimage import uniform_filter1d
from scipy import interpolate

from envs import REGISTRY as env_REGISTRY

sns.set(style='darkgrid', rc={'figure.figsize': (7.2, 4.45),
                              # 'text.usetex': True,
                              'xtick.labelsize': 18,
                              'ytick.labelsize': 18,
                              'font.size': 20,
                              'figure.autolayout': True,
                              'axes.titlesize': 22,
                              'axes.labelsize': 20,
                              'lines.linewidth': 3,
                              'lines.markersize': 6,
                              'legend.fontsize': 18})
colors = sns.color_palette("colorblind", 15)
colors = colors[:5] + colors[7:9]
colors[1] = [0, 0, 0]
colors[6] = np.array(colors[6]) / 1.25

# colors = sns.color_palette("Set1", 2)
# colors = ['#FF4500','#e31a1c','#329932', 'b', 'b', '#6a3d9a','#fb9a99']
# dashes_styles = cycle(['-', '-.', '--', ':'])
sns.set_palette(colors)
colors = cycle(colors)
plt.rcParams["font.family"] = "serif"

DEFAULT_COLS = {
    "learner_data": ["loss", "grad_norm", "q_taken_mean"],
    "env_data": ["episode_reward"]
}


def moving_average(interval, window_size):
    if window_size == 1:
        return interval
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


def plot_df(df, color, xaxis, yaxis, ma=20, label=''):
    df[yaxis] = pd.to_numeric(df[yaxis], errors='coerce')  # convert NaN string to NaN value

    mean = df.groupby(xaxis).mean()[yaxis]
    std = df.groupby(xaxis).std()[yaxis]

    if ma > 1:
        mean = moving_average(mean, ma)
        std = moving_average(std, ma)

    x = df.groupby(xaxis)[xaxis].mean().keys().values
    plt.plot(x, mean, label=label, color=color)
    plt.fill_between(x, mean + std, mean - std, alpha=0.25, color=color, rasterized=True)

    return x, mean


def simple_plot_df(df, color, xaxis, yaxis, ma=1, label=''):
    df.dropna(subset=[yaxis], inplace=True)
    df = df.sort_values(by=[xaxis])

    x = df[xaxis]
    y = df[yaxis]

    if ma > 1:
        y_mean = uniform_filter1d(y, size=ma)

    # create stds
    datarep = np.tile(y, (ma, 1))
    for i in range(1, ma):
        datarep[i, i:] = datarep[i, :-i]
    y_std = np.sqrt(np.mean(np.square(datarep - y_mean[None, :]), 0))

    plot_by_mean_and_std(x, y_mean, y_std, color)

    return x, y_mean, y_std


def plot_by_mean_and_std(x, y_mean, top, bottom, color):
    plt.plot(x, y_mean, color=color)
    plt.fill_between(x, top, bottom, alpha=0.2, color=color, rasterized=True)


def displayable_name(name):
    return name.replace("_", " ").title()


def label_fig(labels, title, xname, yname):
    font = {}  # add custom font if needed

    if len(labels):
        plt.legend(labels, **font, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(title, **font)
    plt.ylabel(displayable_name(yname), **font)
    plt.xlabel(displayable_name(xname), **font)


def find_x_mean_max_min(data, ma=10, max_x=950000):
    new_x = np.linspace(0, max_x, 1000)
    new_ys = []

    # First, approximate data using a polynomial so they are on the same axis
    for x, y in data:
        # a1_coefs = np.polyfit(x, y, 5)
        # Get your new y coordinates from the coefficients of the above polynomial
        # new_ys.append(np.polyval(a1_coefs, new_x))

        if ma > 1:
            y = uniform_filter1d(y, size=ma)

        tck = interpolate.splrep(x, y, s=0)
        new_ys.append(interpolate.splev(new_x, tck, der=0))

    # Now approximate mean:
    mean = np.mean(np.array(new_ys), axis=0)
    max = np.max(np.array(new_ys), axis=0)
    min = np.min(np.array(new_ys), axis=0)
    return new_x, mean, max, min


ALG_NAMES = [
    "LOMAQ",
    "LOMAQ+RD",
    "QMIX",
    "IQL-local",
    "IQL",
    "VDN",
    "QTRAN"
]

ENV_NAMES = {
    "multi_cart": "Multi-Cart-Pole",
    "multi_particle": "Bounded-Cooperative-Navigation",
}

BASE_PATH = f"results/final_results/"


def plot_85():
    curr_path = f"{BASE_PATH}{85}/"
    plt.figure()

    # File reading and grouping
    # Return means is a dict of alg -> [return_seed1, return_seed2, ...]
    different_kappas = {}
    # for kappa in ["0", "1", '2']:
    #     different_kappas[alg_name] = list()

    fname = f"{curr_path}{os.listdir(curr_path)[0]}"
    df = pd.read_csv(fname)
    for column in df:
        print(column)
        if column == "Step" or "MIN" in column or "MAX" in column:
            continue
        kappa_num = int(column.split("-")[1])

        sub_df = df[["Step", column]].copy()
        sub_df.dropna(subset=[column], inplace=True)
        sub_df.sort_values(by=["Step"], inplace=True)
        # sub_df = sub_df[sub_df["Step"] <= 1000000]
        different_kappas[kappa_num] = (sub_df["Step"], sub_df[column])

    # Plot results
    alg_names = []
    for alg_name, results in different_kappas.items():
        x, mean, max, min = find_x_mean_max_min([results], max_x=3000000)
        plot_by_mean_and_std(x, mean, max, min, next(colors))
        alg_names.append(rf"$\kappa$ = {alg_name + 1}")

    label_fig(alg_names, r"Changing $\kappa$", "Timestep", "Test Return Mean")

    # plt.xticks([0, 200000, 400000, 600000, 800000],
    #            ["0", "200K", "400K", "600K", "800K"])

    plt.show()


def plot_main_results():

    for env_name in ["multi_cart", "multi_particle"]:
        curr_path = f"{BASE_PATH}{env_name}/"
        plt.figure()

        # File reading and grouping
        # Return means is a dict of alg -> [return_seed1, return_seed2, ...]
        test_return_means = {}
        for alg_name in ALG_NAMES:
            test_return_means[alg_name] = list()

        labels = []
        for seed_name in os.listdir(curr_path):
            seed_file = f"{curr_path}{seed_name}"

            if os.path.isfile(seed_file):
                df = pd.read_csv(seed_file)
                for column in df:
                    if column == "Step" or "MIN" in column or "MAX" in column:
                        continue
                    alg_num = int(column.split("-")[2])

                    sub_df = df[["Step", column]].copy()
                    sub_df.dropna(subset=[column], inplace=True)
                    sub_df.sort_values(by=["Step"], inplace=True)
                    sub_df = sub_df[sub_df["Step"] <= 1000000]
                    test_return_means[ALG_NAMES[alg_num]].append((sub_df["Step"], sub_df[column]))

        # Plot results
        for alg_name, results in test_return_means.items():
            x, mean, max, min = find_x_mean_max_min(results)
            plot_by_mean_and_std(x, mean, max, min, next(colors))

        title = ENV_NAMES[env_name]
        label_fig(ALG_NAMES, title, "Timestep", "Test Return Mean")

        plt.xticks([0, 200000, 400000, 600000, 800000],
                   ["0", "200K", "400K", "600K", "800K"])

        plt.show()

if __name__ == '__main__':
    # plot_main_results()
    plot_85()
