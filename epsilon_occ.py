#!/usr/bin/env python3

import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.metrics import accuracy_score


class RPEOCC(BaseEstimator, ClassifierMixin):
    def __init__(self, D, epsilon, agg_thr=1, box_thr=0, agg_abs=None):
        self.D = D
        self.epsilon = epsilon
        self.agg_thr = agg_thr
        self.agg_abs = agg_abs
        self.box_thr = box_thr

    def fit(self, x):
        # generate D unit vectors
        # self.feature_len = len(x[0])
        self.feature_len = x[0].shape[1]
        # print("self.feature_len is ", self.feature_len)
        rng = np.random.RandomState(seed=42)
        planes = rng.uniform(low=-1, size=(self.D, self.feature_len))
        mags = np.sqrt((planes * planes).sum(axis=-1)).reshape(-1, 1)
        planes = planes / mags
        self.planes = planes
        self.box_thr = x.shape[0] * self.box_thr

        # dot them with each dimention
        projections = x.dot(planes.T)  # shape should be NxD

        # take epsilon separated intervals in each dimension

        ## sort each dimension
        projections = np.sort(projections, axis=0)

        # list of length D. each entry is a list of tuples representing interval start and end
        self.intervals = [
            get_intervals(projections[:, d], self.epsilon, self.box_thr)
            for d in range(self.D)
        ]
        # print(x.shape, self.planes.shape)
        # print(x.todense())
        # print(self.planes)
        # print("sparsity ", (1.0 - np.count_nonzero(self.planes) / self.planes.size))
        # print(self.intervals)
        # print(lol)

        # self.intervals = filter_intervals(projections, self.intervals, self.box_thr)

        return self

    def decision_function(self, x):
        # get the projections
        projections = x.dot(self.planes.T)

        scores = []
        for v in projections:
            pred_sum = 0
            for d in range(self.D):
                if in_interval(self.intervals[d], v[d]):
                    pred_sum += 1
            scores.append(pred_sum / self.D)
        return np.array(scores)

    def predict(self, x):
        # get the projections
        projections = x.dot(self.planes.T)

        predictions = []
        for v in projections:
            pred_sum = 0
            for d in range(self.D):
                if in_interval(self.intervals[d], v[d]):
                    pred_sum += 1
            if not self.agg_abs:
                if pred_sum > self.D * self.agg_thr:
                    predictions.append(1)
                else:
                    predictions.append(-1)
            elif (self.D - pred_sum) < self.agg_abs:
                predictions.append(1)
            else:
                predictions.append(-1)
        return np.array(predictions)

    def score(self, x, y):
        preds = self.predict(x)
        return accuracy_score(preds, y)

    def get_params(self, deep=None):
        return {
            "D": self.D,
            "epsilon": self.epsilon,
            "agg_thr": self.agg_thr,
            "agg_abs": self.agg_abs,
            "box_thr": self.box_thr,
        }

    def set_params(self, **params):
        self.D = params["D"]
        self.epsilon = params["epsilon"]
        if "agg_thr" in params.keys():
            self.agg_thr = params["agg_thr"]
        if "agg_abs" in params.keys():
            self.agg_abs = params["agg_abs"]
        if "box_thr" in params.keys():
            self.box_thr = params["box_thr"]

    def get_size(self):
        plane_size = self.planes.shape[0] * self.planes.shape[1]
        interval_size = len(self.intervals) * len(self.intervals[0])
        return (plane_size + interval_size) * 64


def get_intervals(l, epsilon, box_thr):
    """ Returns a list of (start, end) tuples of intervals"""
    intervals = []
    start = l[0]
    end = l[0]

    count = 0
    epsilon = (np.max(l) - np.min(l)) * epsilon

    for e in l[1:]:
        if e < end + epsilon:
            end = e
            count += 1
        else:
            if count > box_thr:
                intervals.append((start - np.finfo(np.float32).eps, end + np.finfo(np.float32).eps))
            count = 0
            start = e
            end = e
    else:
        if count > box_thr:
            intervals.append((start - np.finfo(np.float32).eps, end + np.finfo(np.float32).eps))
        count = 0

    return intervals


def in_interval(intervals, val):
    for start, end in intervals:
        if start < val < end:
            return True
    else:
        return False
