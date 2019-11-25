"""
This file provides functions required to replace the functions that are needed
in the various homework excersises for the statistics course of 2019-201800421-1B.

All plotting functions accept a `plt` parameter, which describes the target
canvas for rendering. It can be matplotlib.pyplot (which is the default).
"""

from typing import Iterable, List

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as pyplot

def summary(measurements, dataset_names: List[str], measures: Iterable[str]=[
    'N', 'Mean', 'Standard Deviation', 'Variance', 'Kurtosis', 'Skewness',
    'Minimum', '25th percentile', 'Median', '75th percentile', 'Maximum'
], rounding: int=2, plt=pyplot, show: bool=True):
    """
    Draws a summary table for the given dataset.

    Measures:
    * 'N': Sample size
    * 'Mean'
    * 'Standard Deviation'
    * 'Variance'
    * 'Kurtosis'
    * 'Skewness'
    * 'Minimum'
    * '25th percentile'
    * 'Median'
    * '75th percentile'
    * 'Maximum'

    Example:
        summary([np.array([0.1,0.2,0.3,0.4]), np.array([0.5, 0.6, 0.7, 0.8, 0.9])], ["A", "B"])

    :param measurements: The measurements to plot a summary table for (2d array).
    :param dataset_names: The names of the datasets (shown in column header). (|dataset_names| = |measurements|)
    :param measures: An ordered list of measures to show in the table (see the measures list).
    :param rounding: The amount of decimals to round the numbers in the table to.
    :param plt: The matplotlib instance to use (either pyplot or an Axes instance).
    :param show: Whether to call the show method on plt (if it exists).
    :returns: None
    """

    measure_funs = {
        'N': len,
        'Mean': np.mean,
        'Standard Deviation': lambda m: np.std(m, ddof=1),
        'Variance': lambda m: np.var(m, ddof=1),
        'Kurtosis': lambda m: stats.kurtosis(m, bias=False),
        'Skewness': lambda m: stats.skew(m, bias=False),
        'Minimum': lambda m: np.percentile(m, 0.),
        '25th percentile': lambda m: np.percentile(m, 25.),
        'Median': lambda m: np.percentile(m, 50.),
        '75th percentile': lambda m: np.percentile(m, 75.),
        'Maximum': lambda m: np.percentile(m, 100.),
    }

    rows = measures
    columns = dataset_names

    data = np.array([
        [measure_funs[measure](measurement) for measurement in measurements] for measure in measures
    ])

    data = np.round(data, decimals=rounding)

    plt.table(cellText=data,
              rowLabels=rows,
              colLabels=columns,
              colWidths=[0.3] * len(measurements),
              loc='center')
    plt.axis('off')
    if hasattr(plt, 'show') and show:
        plt.show()

def histogram(measurements, dataset_name: str, plt=pyplot, show: bool=True):
    """
    Shows a histogram with a fitted normal distribution.

    Example:
        histogram(np.array([1, 2, 3, 4, 4, 5, 5, 6, 7]), "X")

    :param measurements: The measurements to create a histogram for.
    :param dataset_name: The of the dataset to show in the header.
    :param plt: The matplotlib instance to use (Either pyplot or an Axes instance)
    :param show: Whether to call the show method on plt (if it exists).
    """

    mu, std = stats.norm.fit(measurements)
    plt.hist(measurements, bins='auto', density=True)
    xmin, xmax = 0, 0
    if hasattr(plt, 'get_xlim'):
        xmin, xmax = plt.get_xlim()
    elif hasattr(plt, 'xlim'):
        xmin, xmax = plt.xlim()

    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)

    plt.plot(x, p, 'k', linewidth=2)

    if hasattr(plt, 'set_title'):
        plt.set_title("Histogram of {}".format(dataset_name))
    elif hasattr(plt, 'title'):
        plt.title("Histogram of {}".format(dataset_name))

    if hasattr(plt, 'show') and show:
        plt.show()

def boxplot(measurements, dataset_names: List[str], plt=pyplot, show: bool=True):
    """
    Shows a boxplot.

    Example:
        boxplot([np.array([0.1,0.2,0.3,0.4]), np.array([0.5, 0.6, 0.7, 0.8, 0.9])], ["A", "B"])

    :param measurements: The measurements to create a boxplot for.
    :param dataset_names: The names of the datasets to show on the bottom axis.
    :param plt: The matplotlib instance to use (Either pyplot or an Axes instance)
    :param show: Whether to call the show method on plt (if it exists).
    """

    plt.boxplot(measurements)

    if hasattr(plt, 'xticks'):
        plt.xticks(np.arange(len(dataset_names) + 2), [""] + dataset_names + [""])
    elif hasattr(plt, 'set_xticklabels'):
        plt.set_xticklabels(dataset_names)
        plt.set_xticks(np.arange(len(dataset_names)))

    if hasattr(plt, 'set_title'):
        plt.set_title("Boxplot")
    elif hasattr(plt, 'title'):
        plt.title("Boxplot")

    if hasattr(plt, 'show') and show:
        plt.show()

def qq_norm(measurements, dataset_name: str, plt=pyplot, show: bool=True):
    """
    Shows a Q-Q Plot for a normal distribution.

    Example:
        qq_norm(np.array([1, 2, 3, 4, 4, 5, 5, 6, 7]), "X")

    :param measurements: The measurements to create a Q-Q Plot for.
    :param dataset_name: The of the dataset to show in the header.
    :param plt: The matplotlib instance to use (Either pyplot or an Axes instance)
    :param show: Whether to call the show method on plt (if it exists).
    """

    stats.probplot(measurements, dist="norm", plot=plt)

    if hasattr(plt, 'set_title'):
        plt.set_title("Q-Q Plot (normal distribution) of {}".format(dataset_name))
    elif hasattr(plt, 'title'):
        plt.title("Q-Q Plot (normal distribution) of {}".format(dataset_name))

    if hasattr(plt, 'show') and show:
        plt.show()

def qq_exp(measurements, dataset_name, plt: str=pyplot, show: bool=True):
    """
    Shows a Q-Q Plot for an exponential distribution.

    Example:
        qq_exp(np.array([1, 2, 3, 4, 4, 5, 5, 6, 7]), "X")

    :param measurements: The measurements to create a Q-Q Plot for.
    :param dataset_name: The of the dataset to show in the header.
    :param plt: The matplotlib instance to use (Either pyplot or an Axes instance)
    :param show: Whether to call the show method on plt (if it exists).
    """

    stats.probplot(measurements, dist="expon", plot=plt)

    if hasattr(plt, 'set_title'):
        plt.set_title("Q-Q Plot (exponential distribution) of {}".format(dataset_name))
    elif hasattr(plt, 'title'):
        plt.title("Q-Q Plot (exponential distribution) of {}".format(dataset_name))

    if hasattr(plt, 'show') and show:
        plt.show()
