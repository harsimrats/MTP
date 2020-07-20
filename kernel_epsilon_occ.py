#!/usr/bin/env python3

import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.metrics import accuracy_score
import my_kernels as k


class RPEOCC(BaseEstimator, ClassifierMixin):
    def __init__(self, D, epsilon, agg_thr=1, box_thr=0, agg_abs=None, kernel="g"):
        self.D = D
        self.epsilon = epsilon
        self.agg_thr = agg_thr
        self.agg_abs = agg_abs
        self.box_thr = box_thr
        self.k = kernel

    def fit(self, x):
        # generate D unit vectors
        self.feature_len = len(x[0])
        rng = np.random.RandomState(seed=42)
        planes = rng.uniform(low=-1, size=(self.D, self.feature_len))
        mags = np.sqrt((planes * planes).sum(axis=-1)).reshape(-1, 1)
        planes = planes / mags
        self.planes = planes

        self.box_thr = len(x) * self.box_thr

        # dot them with each dimention
        if self.k == "g":
            projections = k.gaussian(x, self.planes)  # shape should be NxD
        elif self.k == "p":
            projections = k.poly(x, self.planes)
        elif self.k == "our":
            projections = k.my_kernel(np.array([3, -3]), np.array([-3, 3]))(
                x, self.planes
            )

        # take epsilon separated intervals in each dimension

        ## sort each dimension
        projections = np.sort(projections, axis=0)

        # list of length D. each entry is a list of tuples representing interval start and end
        self.intervals = [
            get_intervals(projections[:, d], self.epsilon, self.box_thr)
            for d in range(self.D)
        ]

        # self.intervals = filter_intervals(projections, self.intervals, self.box_thr)

        return self

    def decision_function(self, x):
        # get the projections
        if self.k == "g":
            projections = k.gaussian(x, self.planes)  # shape should be NxD
        elif self.k == "p":
            projections = k.poly(x, self.planes)
        elif self.k == "our":
            projections = k.my_kernel(np.array([3, -3]), np.array([-3, 3]))(
                x, self.planes
            )

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
        if self.k == "g":
            projections = k.gaussian(x, self.planes)  # shape should be NxD
        elif self.k == "p":
            projections = k.poly(x, self.planes)
        elif self.k == "our":
            projections = k.my_kernel(np.array([3, -3]), np.array([-3, 3]))(
                x, self.planes
            )

        predictions = []
        for v in projections:
            pred_sum = 0
            for d in range(self.D):
                if in_interval(self.intervals[d], v[d]):
                    pred_sum += 1
            if not self.agg_abs:
                if pred_sum >= self.D * self.agg_thr:
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
        return {"D": self.D}


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
            if count >= box_thr:
                intervals.append((start, end))
            count = 0
            start = e
            end = e
    else:
        if count >= box_thr:
            intervals.append((start, end))
        count = 0

    return intervals


def in_interval(intervals, val):
    for start, end in intervals:
        if start <= val <= end:
            return True
    else:
        return False
