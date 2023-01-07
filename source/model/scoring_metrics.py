import nltk
import numpy as np
import sklearn
from nltk.metrics.segmentation import pk, windowdiff


def get_windiff(ref: np.ndarray, pred: np.ndarray, k = None):
    """Receives input for the predicted and the true values as ordered np arrays. Transforms them
    to work with existing documentation, and then returns the relevant windiff score.

    Method assumes that entries are in an nd array, where 0 means no topic change, and 1 marks the
    start of a new topic. 1 is an integer, not a string!

    :param ref: The target segmentation to be compared
    :param pred: Estimated segmentation to use
    :param k: The window width"""

    if k is None:
        # Need to estimate it manually in case it is None. Because some other pieces of code use this
        k = int(round(len(ref) / (np.count_nonzero(ref) * 2.0)))

    return windowdiff(ref.tolist(), pred.tolist(), k, boundary=1)


def get_pk(ref: np.ndarray, pred: np.ndarray, k: None):
    """Method to get the Pk metric for a pair of segmentations. Uses NLTK library, and mainly focuses on
    transforming the input to work for nltk.

    Method assumes that entries are in an nd array, where 0 means no topic change, and 1 marks the
    start of a new topic.

    :param ref: The target segmentation to be compared
    :param pred: Estimated segmentation to use
    :param k: The window width. Because of how pk code works, value is None by default"""

    return pk(ref.tolist(), pred.tolist(), k, boundary=1)


def get_k_kappa(ref: np.ndarray, pred: np.ndarray, k=None):
    """Modified version of Pk that corrects for chance agreement as defined in
    doi:10.1109/SLT.2010.5700820

    For this method, the results mean:

    :param ref: The target segmentation to be compared
    :param pred: Estimated segmentation to use
    :param k: The window width. Because of how pk code works, value is None by default"""
    if k is None:
        # Need to estimate it manually in case it is None. Because some other pieces of code use this
        k = int(round(len(ref) / (np.count_nonzero(ref) * 2.0)))

    # First, getting the K values
    pk_val = get_pk(ref, pred, k)

    # I need a way of getting the boolean boundary presence indicator function
    # TODO: It should just be a check from i to k, on whether any value is 1
    # Which does sound easy enough

    # Because if it's a 1 it is equal to true for numpy, using np any is a perfect replacement
    # Making this quite easy to implement!
    pk_pred_est = get_pk_tilde_estimate(pred, k)
    pk_ref_est = get_pk_tilde_estimate(ref, k)

    chance_agreement = pk_ref_est * pk_pred_est + (1 - pk_ref_est) * (1 - pk_pred_est)

    k_k = (1 - pk_val - chance_agreement) / (1 - chance_agreement)

    return k_k


def get_pk_tilde_estimate(seg: np.ndarray, k):
    """Helper method to implement k_kappa as defined in doi:10.1109/SLT.2010.5700820

    This method just goes through the segment, and calculates some probability which
    I do not properly understand the name for"""

    # Indicator function can just be if there's any element different than 0
    #  due to how this is handled for np arrays
    bool_indic_func = lambda X, i, k: np.any(X[i:i + k])

    prob = 0.0
    for i in range(0, len(seg)-k+1):
        prob += bool_indic_func(seg, i, k)

    return 1 - prob/(len(seg)-k+1)
