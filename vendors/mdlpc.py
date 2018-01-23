"""
Minimum Description Lenght Principle Cut

Code modified from Victor Ruiz, vmr11@pitt.edu
"""

import pandas as pd
import numpy as np

from collections import Counter
from math import log
import sys
import getopt
import re

PRECISION = 1e-4

def entropy(data_classes):
    """
    Computes the entropy of a set of labels (class instantiations)
    :param data_classes: Series with labels of examples in a dataset
    :return: value of entropy
    """
    arr = np.array(data_classes)
    classes, counts = np.unique(arr, return_counts=True)
    # classes = Counter(data_classes)
    proportions = counts/len(arr)
    return -sum(proportions * np.log2(proportions))


def cut_point_information_gain(dataset, cut_point, feature_label, class_label):
    """
    Return de information gain obtained by splitting a numeric attribute in two according to cut_point
    :param dataset: pandas dataframe with a column for attribute values and a column for class
    :param cut_point: threshold at which to partition the numeric attribute
    :param feature_label: column label of the numeric attribute values in data
    :param class_label: column label of the array of instance classes
    :return: information gain of partition obtained by threshold cut_point
    """
    if not isinstance(dataset, pd.DataFrame):
        raise AttributeError('input dataset should be a pandas data frame')

    entropy_full = entropy(dataset[class_label])  # compute entropy of full dataset (w/o split)

    #split data at cut_point
    data_left = dataset[dataset[feature_label] <= cut_point]
    data_right = dataset[dataset[feature_label] > cut_point]
    N, N_left, N_right = len(dataset), len(data_left), len(data_right)

    gain = entropy_full - (N_left / N) * entropy(data_left[class_label]) - \
        (N_right / N) * entropy(data_right[class_label])

    return gain


def cut2labels(cut_points):
    labels = ["<"+str(cut_points[0])]
    for i in range(len(cut_points)-1):
        labels.append("{:s}-{:s}".format(cut_points[i], cut_points[i+1]))
    labels.append(">"+str(cut_points[1]))


class MDLPDiscretizer(object):
    def __init__(self, dataset, class_label, features=None):
        """
        initializes discretizer object:
            saves raw copy of data and creates self._data with only features to discretize and class
            computes initial entropy (before any splitting)
            self._features = features to be discretized
            self._classes = unique classes in raw_data
            self._class_name = label of class in pandas dataframe
            self._data = partition of data with only features of interest and class
            self._cuts = dictionary with cut points for each feature
        :param dataset: pandas dataframe with data to discretize
        :param class_label: name of the column containing class in input dataframe
        :param features: if !None, features that the user wants to discretize specifically
        :return:
        """

        if not isinstance(dataset, pd.DataFrame):  # class needs a pandas dataframe
            raise AttributeError('input dataset should be a pandas data frame')

        self._data_raw = dataset # copy or original input data

        self._class_name = class_label

        self._classes = self._data_raw[self._class_name].unique()

        # if user specifies which attributes to discretize
        if features is not None:
            self._features = [f for f in features if f in self._data_raw.columns]  # check if features in dataframe
            missing = set(features) - set(self._features)  # specified columns not in dataframe
            if missing:
                print('WARNING: user-specified features %s not in input dataframe' % str(missing))
        else:  # then we need to recognize which features are numeric
            numeric_cols = self._data_raw._data.get_numeric_data().items
            self._features = [f for f in numeric_cols if f != class_label]
        # other features that won't be discretized
        self._ignored_features = set(self._data_raw.columns) - set(self._features)

        # create copy of data only including features to discretize and class
        self._data = self._data_raw.loc[:, self._features + [class_label]]
        self.discrete_labels = None
        # pre-compute all boundary points in dataset
        self._boundaries = self.compute_boundary_points_all_features()
        # initialize feature bins with empty arrays
        self._cuts = {f: [] for f in self._features}
        # get cuts for all features
        self.all_features_accepted_cutpoints()
        # discretize self._data
        self.apply_cutpoints()

    def save(self, out_data_path, description_path):
        """
        Save the discretized data to out_data_path and the description to description_path
        :param out_data_path: str
        :param description_path: str
        :return: None
        """
        # save data as csv
        self._data.to_csv(out_data_path)
        # save bins description
        print('Description of bins in file: %s' % out_data_path)
        with open(description_path, 'w') as f:
            for feature in self._features:
                f.write('attr: %s\n\t%s\n' % (feature, ', '.join([label for label in self.discrete_labels[feature]])))

    def mdlpc_criterion(self, data, feature, cut_point):
        """
        Determines whether a partition is accepted according to the MDLPC criterion
        :param feature: feature of interest
        :param cut_point: proposed cut_point
        :return: True/False, whether to accept the partition
        """
        #get dataframe only with desired attribute and class columns, and split by cut_point
        data_partition = data.copy(deep=True)
        data_left = data_partition[data_partition[feature] <= cut_point]
        data_right = data_partition[data_partition[feature] > cut_point]

        # compute information gain obtained when splitting data at cut_point
        cut_point_gain = cut_point_information_gain(dataset=data_partition, cut_point=cut_point,
                                                    feature_label=feature, class_label=self._class_name)
        # compute delta term in MDLPC criterion
        N = len(data_partition) # number of examples in current partition
        partition_entropy = entropy(data_partition[self._class_name])
        k = len(data_partition[self._class_name].unique())
        k_left = len(data_left[self._class_name].unique())
        k_right = len(data_right[self._class_name].unique())
        entropy_left = entropy(data_left[self._class_name])  # entropy of partition
        entropy_right = entropy(data_right[self._class_name])
        delta = log(3 ** k, 2) - (k * partition_entropy) + (k_left * entropy_left) + (k_right * entropy_right)

        # to split or not to split
        gain_threshold = (log(N - 1, 2) + delta) / N

        return cut_point_gain > gain_threshold

    def feature_boundary_points(self, data, feature):
        """
        Given an attribute, find all potential cut_points (boundary points)
        :param feature: feature of interest
        :return: array with potential cut_points
        """
        # get dataframe with only rows of interest, and feature and class columns
        data_partition = data.copy(deep=True)
        data_partition.sort_values(feature, ascending=True, inplace=True)

        boundary_points = []

        # add temporary columns
        data_partition['class_offset'] = data_partition[self._class_name].shift(1)
        # column where first value is now second, and so forth
        data_partition['feature_offset'] = data_partition[feature].shift(1)
        # column where first value is now second, and so forth
        data_partition['feature_change'] = (data_partition[feature] != data_partition['feature_offset'])
        data_partition['mid_points'] = data_partition.loc[:, [feature, 'feature_offset']].mean(axis=1)

        potential_cuts = data_partition[data_partition['feature_change'] == True].index[1:]
        sorted_index = data_partition.index.tolist()

        for row in potential_cuts:
            old_value = data_partition.loc[sorted_index[sorted_index.index(row) - 1]][feature]
            new_value = data_partition.loc[row][feature]
            old_classes = data_partition[data_partition[feature] == old_value][self._class_name].unique()
            new_classes = data_partition[data_partition[feature] == new_value][self._class_name].unique()
            if len(set.union(set(old_classes), set(new_classes))) > 1:
                boundary_point = data_partition.loc[row]['mid_points']
                boundary_point = int(boundary_point/PRECISION) * PRECISION
                boundary_points += [boundary_point]

        return set(boundary_points)

    def compute_boundary_points_all_features(self):
        """
        Computes all possible boundary points for each attribute in self._features (features to discretize)
        :return:
        """
        boundaries = {}
        for attr in self._features:
            data_partition = self._data.loc[:, [attr, self._class_name]]
            boundaries[attr] = self.feature_boundary_points(data=data_partition, feature=attr)
        return boundaries

    def boundaries_in_partition(self, data, feature):
        """
        From the collection of all cut points for all features, find cut points that fall within a feature-partition's
        attribute-values' range
        :param data: data partition (pandas dataframe)
        :param feature: attribute of interest
        :return: points within feature's range
        """
        range_min, range_max = data[feature].min(), data[feature].max()
        return set([x for x in self._boundaries[feature] if (x > range_min) and (x < range_max)])

    def best_cut_point(self, data, feature):
        """
        Selects the best cut point for a feature in a data partition based on information gain
        :param data: data partition (pandas dataframe)
        :param feature: target attribute
        :return: value of cut point with highest information gain (if many, picks first). None if no candidates
        """
        candidates = self.boundaries_in_partition(data=data, feature=feature)
        # candidates = self.feature_boundary_points(data=data, feature=feature)
        if not candidates:
            return None
        gains = [(cut, cut_point_information_gain(dataset=data, cut_point=cut, feature_label=feature,
                                                  class_label=self._class_name)) for cut in candidates]
        gains = sorted(gains, key=lambda x: x[1], reverse=True)

        return gains[0][0] #return cut point

    def single_feature_accepted_cutpoints(self, feature, partition_index=pd.DataFrame().index):
        """
        Computes the cuts for binning a feature according to the MDLP criterion
        :param feature: attribute of interest
        :param partition_index: index of examples in data partition for which cuts are required
        :return: list of cuts for binning feature in partition covered by partition_index
        """
        if partition_index.size == 0:
            partition_index = self._data.index  # if not specified, full sample to be considered for partition

        data_partition = self._data.loc[partition_index, [feature, self._class_name]]

        #exclude missing data:
        if data_partition[feature].isnull().values.any:
            data_partition = data_partition[~data_partition[feature].isnull()]

        #stop if constant or null feature values
        if len(data_partition[feature].unique()) < 2:
            return
        #determine whether to cut and where
        cut_candidate = self.best_cut_point(data=data_partition, feature=feature)
        if cut_candidate == None:
            return
        decision = self.mdlpc_criterion(data=data_partition, feature=feature, cut_point=cut_candidate)

        #apply decision
        if not decision:
            return  # if partition wasn't accepted, there's nothing else to do
        if decision:
            # try:
            #now we have two new partitions that need to be examined
            left_partition = data_partition[data_partition[feature] <= cut_candidate]
            right_partition = data_partition[data_partition[feature] > cut_candidate]
            if left_partition.empty or right_partition.empty:
                return #extreme point selected, don't partition
            self._cuts[feature] += [cut_candidate]  # accept partition
            self.single_feature_accepted_cutpoints(feature=feature, partition_index=left_partition.index)
            self.single_feature_accepted_cutpoints(feature=feature, partition_index=right_partition.index)
            #order cutpoints in ascending order
            self._cuts[feature] = sorted(self._cuts[feature])
            return

    def all_features_accepted_cutpoints(self):
        """
        Computes cut points for all numeric features (the ones in self._features)
        :return:
        """
        for feature in self._features:
            self.single_feature_accepted_cutpoints(feature=feature)
        return

    def apply_cutpoints(self, out_data_path=None, out_bins_path=None):
        """
        Discretizes data by applying bins according to self._cuts. Saves a new, discretized file, and a description of
        the bins
        :param out_data_path: path to save discretized data
        :param out_bins_path: path to save bins description
        :return:
        """
        self.discrete_labels = {}
        for feature in self._features:
            if len(self._cuts[feature]) == 0:
                self._data[feature] = 'Any'
                self.discrete_labels[feature] = ['Any']
            else:
                cuts = [-np.inf] + self._cuts[feature] + [np.inf]
                cut_data = pd.cut(x=self._data[feature].values, bins=cuts,
                                  right=False, precision=4, include_lowest=True)
                self._data[feature] = cut_data
                bin_labels = [str(category) for category in cut_data.categories]
                self.discrete_labels[feature] = bin_labels

        # reconstitute full data, now discretized
        if self._ignored_features:
            self._data = pd.concat([self._data, self._data_raw[list(self._ignored_features)]], axis=1)
            self._data = self._data[self._data_raw.columns] #sort columns so they have the original order


def main(argv):
    out_path_data, out_path_bins, return_bins, class_label, features = None, None, False, None, None

    #read command line arguments
    try:
        parameters, _ = getopt.getopt(argv, shortopts='',
                                      longopts=['in_path=', 'out_path=', 'features=', 'class_label=', 'return_bins'])
    except:
        print('Correct usage: python MDLP.py --in_path=path --out_path=path --features=f1,f2,f3... '
              '--class_label=weather --return_bins')
        sys.exit(2)

    for opt, value in parameters:
        if opt == '--in_path':
            data_path = value
            if not data_path.endswith('csv') or data_path.endswith('CSV'):
                print('Input data must be in csv file format')
                sys.exit(2)
            print('Input file: %s' % data_path)
        elif opt == '--out_path':
            out_path_data = value
            if not data_path.endswith('csv') or data_path.endswith('CSV'):
                out_path_data = '%s.csv' % out_path_data
            print('Output file to be saved at: %s' % out_path_data)
        elif opt == '--features':
            features = re.split(r',', value)
            features = [f for f in features if f]
        elif opt == '--return_bins':
            return_bins = True
        elif opt == '--class_label':
            class_label = value

    if return_bins:
        bins_name = ''.join(re.split(r'\.', out_path_data)[:-1])
        out_path_bins = '%s_bins.txt' % bins_name
        print('Bins information will be saved at: %s' % out_path_bins)

    if not class_label:
        print('A class label must be specified with the --class_label= option')
        sys.exit(2)

    #read input data
    data = pd.read_csv(data_path)
    discretizer = MDLPDiscretizer(dataset=data, class_label=class_label, features=features,
                                   out_path_data=out_path_data, out_path_bins=out_path_bins)

if __name__ == '__main__':
    main(sys.argv[1:])