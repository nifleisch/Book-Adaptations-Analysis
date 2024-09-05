import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)

def parse_string_to_dict(string: str):
    try:
        return ast.literal_eval(string)
    except ValueError:
        return {}


def get_similarity(propensity_score1: float, propensity_score2: float):
    """Calculate similarity for instances with given propensity scores"""
    return 1 - np.abs(propensity_score1 - propensity_score2)


def plot_imdb_histogram(df: pd.DataFrame):
    df = df.assign(
        label=lambda x: x.movie_is_adaptation.map(
            {True: "Adapted", False: "Original"}
        ).astype("category"),
    )
    
    plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 6], hspace=0.05)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    sns.histplot(
        data=df,
        x="imdb_rating",
        hue="label",
        ax=ax1,
        bins=50,
        palette=["#8B0000", "#6a737b"]
    )
    sns.boxplot(
        data=df, 
        x="imdb_rating", 
        y="label", 
        ax=ax0, 
        palette=["#8B0000", "#6a737b"],
        fliersize=0,
    )

    ax0.set(xlabel="", ylabel="")
    ax0.set(xticklabels=[], yticklabels=[])
    ax0.set(xticks=[], yticks=[])
    ax0.spines["top"].set_visible(False)
    ax0.spines["bottom"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.spines["left"].set_visible(False)

    ax1.set(xlabel='IMDB Rating', ylabel="Number of Movies")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.get_legend().set_title("")
    ax1.get_legend()._legend_box.align = "left"

    plt.savefig(f"assets/imdb_histogram.pdf", bbox_inches="tight")
    plt.show()


def plot_revenue_histogram(df: pd.DataFrame):
    df = df.assign(
        label=lambda x: x.movie_is_adaptation.map(
            {True: "Adapted", False: "Original"}
        ).astype("category"),
    )
    
    plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 6], hspace=0.05)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    sns.histplot(
        data=df,
        x="movie_revenue",
        hue="label",
        ax=ax1,
        bins=50,
        palette=["#8B0000", "#6a737b"],
        log_scale=True,
    )
    sns.boxplot(
        data=df, 
        x="movie_revenue_log", 
        y="label", 
        ax=ax0, 
        palette=["#8B0000", "#6a737b"],
        fliersize=0,
    )

    ax0.set(xlabel="", ylabel="")
    ax0.set(xticklabels=[], yticklabels=[])
    ax0.set(xticks=[], yticks=[])
    ax0.spines["top"].set_visible(False)
    ax0.spines["bottom"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.spines["left"].set_visible(False)

    ax1.set(xlabel='Box Office Revenue [$US]', ylabel="Number of Movies")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.get_legend().set_title("")
    ax1.get_legend()._legend_box.align = "left"

    plt.savefig(f"assets/revenue_histogram.pdf", bbox_inches="tight")
    plt.show()

def make_book_histplot_no_stack(df: pd.DataFrame, col: str, x_label: str, log=False, ylog=False):
    if log:
        df["log"] = np.log(df[col])
    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 6], hspace=0.05)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    if ylog==True:
        ax1.set_yscale('log')
    if log:
        sns.histplot(data=df, x=col, hue='label', ax=ax1, bins=50, palette=['#6a737b', '#8B0000'], log_scale=True)
        sns.boxplot(data=df, x="log", y='label', ax=ax0, palette=['#6a737b', '#8B0000'], fliersize=0)
    else:
        sns.histplot(data=df, x=col, hue='label', ax=ax1, bins=50, palette=['#6a737b', '#8B0000'])
        sns.boxplot(data=df, x=col, y='label', ax=ax0, palette=['#6a737b', '#8B0000'], fliersize=0)

    ax0.set(xlabel="", ylabel="")
    ax0.set(xticklabels=[], yticklabels=[])
    ax0.set(xticks=[], yticks=[])
    ax0.spines["top"].set_visible(False)
    ax0.spines["bottom"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.spines["left"].set_visible(False)

    ax1.set(xlabel=x_label, ylabel="Number of Books")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.get_legend().set_title("")
    ax1.get_legend()._legend_box.align = "left"

    plt.savefig(f"assets/{col}_histogram.pdf", bbox_inches="tight")
    plt.show()


def make_book_histplot(df: pd.DataFrame, col: str, x_label: str, log=False, ylog=False):
    if log:
        df["log"] = np.log(df[col])
    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 6], hspace=0.05)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    if ylog==True:
        ax1.set_yscale('log')
    if log:
        sns.histplot(data=df, x=col, hue='label', ax=ax1, bins=50, palette=['#6a737b', '#8B0000'], log_scale=True, multiple="stack")
        sns.boxplot(data=df, x="log", y='label', ax=ax0, palette=['#6a737b', '#8B0000'], fliersize=0)
    else:
        sns.histplot(data=df, x=col, hue='label', ax=ax1, bins=50, palette=['#6a737b', '#8B0000'], multiple="stack")
        sns.boxplot(data=df, x=col, y='label', ax=ax0, palette=['#6a737b', '#8B0000'], fliersize=0)

    ax0.set(xlabel="", ylabel="")
    ax0.set(xticklabels=[], yticklabels=[])
    ax0.set(xticks=[], yticks=[])
    ax0.spines["top"].set_visible(False)
    ax0.spines["bottom"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.spines["left"].set_visible(False)

    ax1.set(xlabel=x_label, ylabel="Number of Books")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.get_legend().set_title("")
    ax1.get_legend()._legend_box.align = "left"

    plt.savefig(f"assets/{col}_histogram.pdf", bbox_inches="tight")
    plt.show()


def make_book_revenue_histplot(df: pd.DataFrame, col: str, labels: list):
    df = df.assign(label= lambda x: x[col].map({1: labels[0], 0: labels[1]}))

    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 6], hspace=0.05)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    
    sns.histplot(data=df, x='movie_revenue', hue='label', ax=ax1, bins=50, palette=['#6a737b', '#8B0000'], log_scale=True, multiple="stack")
    sns.boxplot(data=df, x="movie_revenue_log", y='label', ax=ax0, palette=['#6a737b', '#8B0000'], fliersize=0)

    ax0.set(xlabel="", ylabel="")
    ax0.set(xticklabels=[], yticklabels=[])
    ax0.set(xticks=[], yticks=[])
    ax0.spines["top"].set_visible(False)
    ax0.spines["bottom"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.spines["left"].set_visible(False)

    ax1.set(xlabel="Box Office Revenue", ylabel="Number of Books Adaptations")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.get_legend().set_title("")
    ax1.get_legend()._legend_box.align = "left"

    plt.savefig(f"assets/{col}_histogram.pdf", bbox_inches="tight")
    plt.show()

def make_book_revenue_histplot_no_stacking(df: pd.DataFrame, col: str, labels: list):
    df = df.assign(label= lambda x: x[col].map({1: labels[0], 0: labels[1]}))

    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 6], hspace=0.05)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    
    sns.histplot(data=df, x='movie_revenue', hue='label', ax=ax1, bins=20, palette=['#6a737b', '#8B0000'], log_scale=True)
    sns.boxplot(data=df, x="movie_revenue_log", y='label', ax=ax0, palette=['#6a737b', '#8B0000'], fliersize=0)

    ax0.set(xlabel="", ylabel="")
    ax0.set(xticklabels=[], yticklabels=[])
    ax0.set(xticks=[], yticks=[])
    ax0.spines["top"].set_visible(False)
    ax0.spines["bottom"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.spines["left"].set_visible(False)

    ax1.set(xlabel="Box Office Revenue", ylabel="Number of Books Adaptations")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.get_legend().set_title("")
    ax1.get_legend()._legend_box.align = "left"

    plt.savefig(f"assets/{col}_histogram.pdf", bbox_inches="tight")
    plt.show()
    
def make_movie_rating_histplot_no_stacking(df: pd.DataFrame, col: str, labels: list):
    df = df.assign(label= lambda x: x[col].map({1: labels[0], 0: labels[1]}))

    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 6], hspace=0.05)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    
    sns.histplot(data=df, x='imdb_rating', hue='label', ax=ax1, bins=20, palette=['#6a737b', '#8B0000'], log_scale=False)
    sns.boxplot(data=df, x="imdb_rating", y='label', ax=ax0, palette=['#6a737b', '#8B0000'], fliersize=0)

    ax0.set(xlabel="", ylabel="")
    ax0.set(xticklabels=[], yticklabels=[])
    ax0.set(xticks=[], yticks=[])
    ax0.spines["top"].set_visible(False)
    ax0.spines["bottom"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.spines["left"].set_visible(False)

    ax1.set(xlabel="Movie rating", ylabel="Number of Books Adaptations")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.get_legend().set_title("")
    ax1.get_legend()._legend_box.align = "left"

    plt.savefig(f"assets/{col}_histogram.pdf", bbox_inches="tight")
    plt.show()