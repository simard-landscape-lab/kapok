# -*- coding: utf-8 -*-
"""Kapok Fusion Module

    Module for estimating canopy height and other parameters using
    fusion of UAVSAR with other data such as lidar.

    Author: Michael Denbina

    Copyright 2016-2022 California Institute of Technology.  All rights reserved.
    United States Government Sponsorship acknowledged.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import collections
import os
import pickle

import numpy as np

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from keras import backend as K
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Activation

from kapok.lib import mb_tr_index, mb_cov_index, mb_bl_index


def aleatoric_loss(y_true, y_pred):
    """Aleatoric loss function which can be used to train a network to predict
        both the output value and an uncertainty estimate.  These uncertainties
        can be used to choose between estimates from different baselines or
        datasets.  y_pred should be an [N, 2] array, where y_pred[:,0] are the
        predicted values and y_pred[:,1] are the predicted uncertainties.

    """
    N = y_true.shape[0]
    se = K.pow(y_pred[:,0] - y_true[:,0], 2) # square error
    inv_std = K.exp(-y_pred[:,1])
    mse = K.mean(inv_std*se) # MSE weighted by uncertainty estimate
    reg = K.mean(y_pred[:,1]) # regularize by mean weight
    return 0.5*(mse+reg)


def aleatoric_cosh_loss(y_true, y_pred):
    """Aleatoric loss function which can be used to train a network to predict
        both the output value and an uncertainty estimate.  These uncertainties
        can be used to choose between estimates from different baselines or
        datasets.  y_pred should be an [N, 2] array, where y_pred[:,0] are the
        predicted values and y_pred[:,1] are the predicted uncertainties.

    """
    N = y_true.shape[0]
    err = y_pred[:,0] - y_true[:,0]
    log_cosh_err = K.log(0.5*(K.exp(err) + K.exp(-err)))
    inv_std = K.exp(-y_pred[:,1])
    mse = K.mean(inv_std * log_cosh_err) # MSE weighted by uncertainty estimate
    reg = K.mean(y_pred[:,1]) # regularize by mean weight
    return 0.5*(mse+reg)


def remove_negatives(ph, kz):
    ind = (ph < 0)
    while np.any(ind):
        ph[ind] += np.abs(2*np.pi/kz[ind])
        ind = (ph < 0)
    return ph


def choose_baseline_triplet(scene, baseline_list, num_triplets=None, kz_target=None, limit_tracks=True):
    """Function to choose viable baseline triplets from a list of candidate
        baselines.  Baseline triplets will be chosen such that a range of kz
        values will be represented.  For baselines with similar kz values, the
        baselines with higher mean coherence will be preferred.
    """
    mean_kz = np.zeros((len(baseline_list)), dtype='float32')
    mean_coh = np.zeros((len(baseline_list)), dtype='float32')

    for num, bl in enumerate(baseline_list):
        mean_kz[num] = np.nanmean(np.abs(scene.kz(bl)))
        coh_temp = np.abs(scene.coh('high', bl=bl))
        coh_temp = coh_temp[np.isfinite(coh_temp)]
        mean_coh[num] = np.nanmean(coh_temp)

    ind_sort = np.argsort(mean_kz)
    mean_kz_sorted = mean_kz[ind_sort]
    mean_coh_sorted = mean_coh[ind_sort]
    baseline_list_sorted = baseline_list[ind_sort]

    if kz_target is None:
        ind_small_kz = np.abs(mean_kz_sorted - np.nanmin(mean_kz_sorted)) < 0.04
        ind_large_kz = np.abs(mean_kz_sorted - np.nanmax(mean_kz_sorted)) < 0.04
        ind_middle_kz = (~ind_small_kz) & (~ind_large_kz)
    else:
        target_threshold = 0.1
        ind_small_kz = np.abs(mean_kz_sorted - kz_target[0]) < target_threshold
        ind_middle_kz = np.abs(mean_kz_sorted - kz_target[1]) < target_threshold
        ind_large_kz = np.abs(mean_kz_sorted - kz_target[2]) < target_threshold

    small_kz_group = baseline_list_sorted[ind_small_kz]
    large_kz_group = baseline_list_sorted[ind_large_kz]
    if np.any(ind_middle_kz):
        middle_kz_group = baseline_list_sorted[ind_middle_kz]
    else:
        middle_kz_group = baseline_list_sorted.copy()

    small_kz_coh = mean_coh_sorted[ind_small_kz]
    large_kz_coh = mean_coh_sorted[ind_large_kz]
    middle_kz_coh = mean_coh_sorted[ind_middle_kz]

    # Rank each baseline grouping according to coherence.
    ind_small_coh_sort = np.argsort(small_kz_coh)
    ind_large_coh_sort = np.argsort(large_kz_coh)
    ind_middle_coh_sort = np.argsort(middle_kz_coh)

    small_kz_ranked_group = small_kz_group[ind_small_coh_sort]
    large_kz_ranked_group = large_kz_group[ind_large_coh_sort]
    middle_kz_ranked_group = middle_kz_group[ind_middle_coh_sort]

    if num_triplets is None:
        num_triplets = len(small_kz_ranked_group)

    triplet_list = np.zeros((num_triplets, 3), dtype='int32')
    small_index = 0
    middle_index = 0
    large_index = 0

    for num in range(num_triplets):
        small_index -= 1
        if small_index < -1*len(small_kz_ranked_group):
            small_index = -1
        
        large_index -= 1
        if large_index < -1*len(large_kz_ranked_group):
            large_index = -1

        if limit_tracks:
            tr1, tr2 = mb_tr_index(small_kz_ranked_group[small_index])
            tr3, tr4 = mb_tr_index(large_kz_ranked_group[large_index])
            current_tracks = np.unique(np.array([tr1, tr2, tr3, tr4]))
            while len(current_tracks) > 3:
                large_index -= 1
                if large_index < -1*len(large_kz_ranked_group):
                    large_index = -1
                    small_index -= 1
                    if small_index < -1*len(small_kz_ranked_group):
                        small_index = -1                
                tr1, tr2 = mb_tr_index(small_kz_ranked_group[small_index])
                tr3, tr4 = mb_tr_index(large_kz_ranked_group[large_index])
                current_tracks = np.unique(np.array([tr1, tr2, tr3, tr4]))

        triplet_list[num, 0] = small_kz_ranked_group[small_index]        
        triplet_list[num, 2] = large_kz_ranked_group[large_index]

        # choose middle kz
        middle_index -= 1
        if middle_index < -1*len(middle_kz_ranked_group):
            middle_index = -1
        
        if limit_tracks:
            # only one choice for last baseline
            tr1, tr2 = mb_tr_index(triplet_list[num, 0])
            tr3, tr4 = mb_tr_index(triplet_list[num, 2])
            if tr1 == tr3:
                triplet_list[num, 1] = mb_bl_index(tr2, tr4)
            elif tr1 == tr4:
                triplet_list[num, 1] = mb_bl_index(tr2, tr3)
            elif tr2 == tr3:
                triplet_list[num, 1] = mb_bl_index(tr1, tr4)
            elif tr2 == tr4:
                triplet_list[num, 1] = mb_bl_index(tr1, tr3)
        else:
            triplet_list[num, 1] = middle_kz_ranked_group[middle_index]

        print('Suggested Triplet: ({}, {}, {}) -- mean kz: ({}, {}, {}) -- mean coh: ({}, {}, {})'.format(triplet_list[num, 0], triplet_list[num, 1], triplet_list[num, 2],
            mean_kz[baseline_list == triplet_list[num, 0]],
            mean_kz[baseline_list == triplet_list[num, 1]],
            mean_kz[baseline_list == triplet_list[num, 2]],
            mean_coh[baseline_list == triplet_list[num, 0]],
            mean_coh[baseline_list == triplet_list[num, 1]],
            mean_coh[baseline_list == triplet_list[num, 2]]))

    # Avoid repeating triplets.
    triplet_list = np.unique(triplet_list, axis=0)

    return triplet_list


def choose_baseline_set(scene, baseline_list, num_sets=10, num_baselines_per_set=3, kz_target=None):
    """Function to choose viable baseline sets from a list of candidate
        baselines.  Baseline sets will be chosen such that a range of kz
        values will be represented.  For baselines with similar kz values, the
        baselines with higher mean coherence will be preferred.
    """
    num_sets = int(num_sets)
    num_baselines_per_set = int(num_baselines_per_set)

    if num_baselines_per_set == 3:
        # use specific function for this case
        return choose_baseline_triplet(scene, baseline_list, num_triplets=num_sets, kz_target=kz_target)
    
    mean_kz = np.zeros((len(baseline_list)), dtype='float32')
    mean_coh = np.zeros((len(baseline_list)), dtype='float32')

    for num, bl in enumerate(baseline_list):
        mean_kz[num] = np.nanmean(np.abs(scene.kz(bl)))
        coh_temp = np.abs(scene.coh('high', bl=bl))
        coh_temp = coh_temp[np.isfinite(coh_temp)]
        mean_coh[num] = np.nanmean(coh_temp)

    ind_sort = np.argsort(mean_kz)
    mean_kz_sorted = mean_kz[ind_sort]
    mean_coh_sorted = mean_coh[ind_sort]
    baseline_list_sorted = baseline_list[ind_sort]
    num_times_picked = np.zeros_like(baseline_list_sorted, dtype='int32')

    if kz_target is None:
        # sample linearly through kz range
        kz_bin_edges = np.linspace(np.nanmin(mean_kz_sorted)-0.001, np.nanmax(mean_kz_sorted)+0.001, num=num_baselines_per_set+1)
        kz_bin_centers = kz_bin_edges[0:-1] + kz_bin_edges[1:]
    else:
        # use user-specified kzs for each bin -- this function will try to choose the kz closest to these values, though the cost function also uses coherence and number of times a baseline has been picked previously
        # so as you request larger number of baseline sets, they will tend to be farther from these input kz values, because this function tries to avoid repeating baselines
        kz_bin_centers = np.array(kz_target)

    set_list = np.zeros((num_sets, num_baselines_per_set), dtype='int32')
    temp_kz = np.zeros((num_baselines_per_set), dtype='float32')
    temp_coh = np.zeros((num_baselines_per_set), dtype='float32')

    for num in range(num_sets):
        for num_bl in range(num_baselines_per_set):
            cost = np.abs(mean_kz_sorted - kz_bin_centers[num_bl]) + (1 - mean_coh_sorted) + num_times_picked
            

            best_choice = np.nanargmin(cost)
            set_list[num, num_bl] = baseline_list_sorted[best_choice]
            num_times_picked[best_choice] += 10

            temp_kz[num_bl] = mean_kz_sorted[best_choice]
            temp_coh[num_bl] = mean_coh_sorted[best_choice]

        ind_temp_sort = np.argsort(temp_kz)
        set_list[num] = set_list[num, ind_temp_sort]
        temp_kz = temp_kz[ind_temp_sort]
        temp_coh = temp_coh[ind_temp_sort]
        
        print('Set #{}'.format(num+1))
        print(' -- Baseline IDs: {}'.format(set_list[num]))
        print(' -- Mean kz: {}'.format(temp_kz))
        print(' -- Mean Coh.: {}'.format(temp_coh))
        print('')

    # Avoid repeating triplets.
    set_list = np.unique(set_list, axis=0)

    return set_list


def distilledbl_feature_extractor_v2(scene, baseline_list=None,
    num_to_select=3, blcriteria='prod',
    training_data=None, training_mask=None,
    include_inc=False, include_dem=False):
    """Function to extract a default PolInSAR feature set from a given Kapok
        Scene object.  Uses three baselines distilled from the baselines
        given (on a pixel-basis, choosing baselines that cover a range of
        kz values while maximizing a baseline selection criteria).

        Arguments:
            scene (kapok.Scene): Kapok Scene to extract PolInSAR features from.
            [...]

    """
    if baseline_list is None:
        print('Baseline candidate list not specified!  All baselines will be considered.')
        baseline_list = np.arange(scene.num_baselines)
    elif not isinstance(baseline_list, (collections.abc.Sequence, np.ndarray)):
        baseline_list = np.array([baseline_list])

    if (training_mask is None) and (training_data is not None):
        print('Training data provided but no training mask specified.  All available data will be used for training.')
        training_mask = np.ones(scene.dim, dtype='bool')
    elif (training_mask) is None:
        # no training data, features are being extracted just for prediction, not for training
        # return features for all pixels
        training_mask = np.ones(scene.dim, dtype='bool')

    # Figure out which baselines to use based on the given criteria.
    # Calculate criteria.
    if 'prod' in blcriteria:
        baseline_list = np.sort(baseline_list)
        gamma = scene.pdcoh[baseline_list, :, :, :]
        criteria = np.abs(gamma[:,0] - gamma[:,1]) * np.abs(gamma[:,0] + gamma[:,1])
    else:
        pass # todo: implement other baseline selection criteria

    # Split baseline list into groups by kz.
    kz = np.zeros((len(baseline_list),self.dim[0],self.dim[1]),dtype='float32')
    mean_kz = np.zeros((len(baseline_list)), dtype='float32')
    for m, n in enumerate(baseline_list):
        kz[m] = scene.kz(n)
        mean_kz[m] = np.abs(np.nanmean(kz[m]))

    # split baselines into num_to_select groups

    # Choose baseline within each kz category that maximizes criteria.
    blsel = np.argmax(criteria, axis=0)

    features = None
    training = None

    num_samples = np.sum(training_mask)
    num_features = 19*3
    if include_inc:
        num_features += 1
    if include_dem:
        num_features += 1

    features_temp = np.zeros((num_samples, num_features), dtype='float32')
    foffset = 0

    for bl in baseline_triplet:
        print('Extracting features for baseline {}...'.format(bl))
        tr1, tr2 = mb_tr_index(bl) # tracks corresponding to this baseline

        # backscatter
        features_temp[:, 0 + foffset] = scene.power('hh', tr=tr1)[training_mask]
        features_temp[:, 1 + foffset] = scene.power('hv', tr=tr1)[training_mask]
        features_temp[:, 2 + foffset] = scene.power('vv', tr=tr1)[training_mask]

        # interferometric covariance matrix
        cov_row, cov_col = mb_cov_index(bl)
        features_temp[:, 3 + foffset] = np.real(scene.cov[:, :, cov_row, cov_col])[training_mask]
        features_temp[:, 4 + foffset] = np.real(scene.cov[:, :, cov_row+1, cov_col+1])[training_mask]
        features_temp[:, 5 + foffset] = np.real(scene.cov[:, :, cov_row+2, cov_col+2])[training_mask]
        features_temp[:, 6 + foffset] = np.imag(scene.cov[:, :, cov_row, cov_col])[training_mask]
        features_temp[:, 7 + foffset] = np.imag(scene.cov[:, :, cov_row+1, cov_col+1])[training_mask]
        features_temp[:, 8 + foffset] = np.imag(scene.cov[:, :, cov_row+2, cov_col+2])[training_mask]

        # kz and incidence angle
        kztemp = scene.kz(bl)[training_mask]
        kztemp = np.clip(kztemp, -1, 1)
        features_temp[:, 9 + foffset] = kztemp

        # optimized coherences and coherence region parameters
        high = scene.coh('high',bl=bl)[training_mask]
        low = scene.coh('low',bl=bl)[training_mask]
        ground = scene.coh('ground',bl=bl)[training_mask]

        highpch = remove_negatives(np.angle(high*np.conj(ground))/kztemp, kztemp)
        highpch = np.clip(highpch, 0, 100)
        lowpch = remove_negatives(np.angle(low*np.conj(ground))/kztemp, kztemp)
        lowpch = np.clip(lowpch, 0, 100)

        features_temp[:, 10 + foffset] = np.abs(high)
        features_temp[:, 11 + foffset] = np.abs(low)
        features_temp[:, 12 + foffset] = highpch
        features_temp[:, 13 + foffset] = lowpch
        features_temp[:, 14 + foffset] = np.abs(high - low)
        features_temp[:, 15 + foffset] = np.abs(high - ground)
        coherence_location = (high + low) * np.conj(ground)
        features_temp[:, 16 + foffset] = np.real(coherence_location)
        features_temp[:, 17 + foffset] = np.imag(coherence_location)
        del coherence_location
        features_temp[:, 18 + foffset] = np.angle((high - low) * np.conj(ground))

        foffset += 19

    if include_inc:
        features_temp[:, foffset] = scene.inc[training_mask]
        foffset += 1

    if include_dem:
        features_temp[:, foffset] = scene.dem[training_mask]

    print('Extracted {} features from {} samples.'.format(num_features, num_samples))
    if features is None:
        features = features_temp.copy()
    else:
        features = np.append(features, features_temp, axis=0)

    # masked training data, if specified
    if training_data is not None:
        print('Extracted training data from {} samples.'.format(num_samples))
        if training is None:
            training = training_data[training_mask].flatten()
        else:
            training = np.append(training, training_data[training_mask], axis=0)

    return features, training


def triplebl_feature_extractor_v2(scene, baseline_triplet=None, training_data=None,
                                  training_mask=None, include_inc=False,
                                  include_dem=False, include_alt_ground=False):
    """Function to extract a default PolInSAR feature set from a given Kapok
        Scene object.  Uses three baselines.

        Updated version which includes parameters which most often made it
        through feature selection in Denbina et al., 2018.

        Feature List (19-21 Total Features):
         * Backscatter (3 pol)
         * Interferometric Covariance (3 pol, real and imag = 6 total)
         * kz (1)
         * High and Low Coherence Magnitude (2)
         * High and Low Coherence Phase Center Height Above Estimated Ground (2)
         * abs(High - Low) -- High and Low Coherence Separation (1)
         * abs(High - Ground) -- High and Ground Coherence Separation (1)
         * Coherence Region Centroid Location w/ Ground Phase Removed -- Real and Imaginary Parts (2)
         * Coherence Region Orientation w/ Ground Phase Removed (1)
         * Optionally: Incidence Angle and DEM (based on input arguments)

        Arguments:
            scene (kapok.Scene): Kapok Scene to extract PolInSAR features from.

    """
    if baseline_triplet is None:
        print('Baseline triplet not specified!  First three baselines will be extracted.')
        baseline_triplet = np.array([0, 1, 2])
    elif not isinstance(baseline_triplet, (collections.abc.Sequence, np.ndarray)):
        baseline_triplet = np.array([baseline_triplet])

    if len(baseline_triplet) != 3:
        print('Baseline triplet has incorrect length!  Aborting!')
        return

    if (training_mask is None) and (training_data is not None):
        print('Training data provided but no training mask specified.  All available data will be used for training.')
        training_mask = np.ones(scene.dim, dtype='bool')
    elif (training_mask) is None:
        # no training data, features are being extracted just for prediction, not for training
        # return features for all pixels
        training_mask = np.ones(scene.dim, dtype='bool')

    features = None
    training = None

    num_samples = np.sum(training_mask)
    num_features = 19*3
    if include_inc:
        num_features += 1
    if include_dem:
        num_features += 1

    features_temp = np.zeros((num_samples, num_features), dtype='float32')
    foffset = 0

    for bl in baseline_triplet:
        print('Extracting features for baseline {}...'.format(bl))
        tr1, tr2 = mb_tr_index(bl) # tracks corresponding to this baseline

        # backscatter
        features_temp[:, 0 + foffset] = scene.power('hh', tr=tr1)[training_mask]
        features_temp[:, 1 + foffset] = scene.power('hv', tr=tr1)[training_mask]
        features_temp[:, 2 + foffset] = scene.power('vv', tr=tr1)[training_mask]

        # interferometric covariance matrix
        cov_row, cov_col = mb_cov_index(bl)
        features_temp[:, 3 + foffset] = np.real(scene.cov[:, :, cov_row, cov_col])[training_mask]
        features_temp[:, 4 + foffset] = np.real(scene.cov[:, :, cov_row+1, cov_col+1])[training_mask]
        features_temp[:, 5 + foffset] = np.real(scene.cov[:, :, cov_row+2, cov_col+2])[training_mask]
        features_temp[:, 6 + foffset] = np.imag(scene.cov[:, :, cov_row, cov_col])[training_mask]
        features_temp[:, 7 + foffset] = np.imag(scene.cov[:, :, cov_row+1, cov_col+1])[training_mask]
        features_temp[:, 8 + foffset] = np.imag(scene.cov[:, :, cov_row+2, cov_col+2])[training_mask]

        # kz and incidence angle
        kztemp = scene.kz(bl)[training_mask]
        kztemp = np.clip(kztemp, -1, 1)
        features_temp[:, 9 + foffset] = kztemp

        # optimized coherences and coherence region parameters
        high = scene.coh('high',bl=bl)[training_mask]
        low = scene.coh('low',bl=bl)[training_mask]
        ground = scene.coh('ground',bl=bl)[training_mask]

        highpch = remove_negatives(np.angle(high*np.conj(ground))/kztemp, kztemp)
        highpch = np.clip(highpch, 0, 100)
        lowpch = remove_negatives(np.angle(low*np.conj(ground))/kztemp, kztemp)
        lowpch = np.clip(lowpch, 0, 100)

        features_temp[:, 10 + foffset] = np.abs(high)
        features_temp[:, 11 + foffset] = np.abs(low)
        features_temp[:, 12 + foffset] = highpch
        features_temp[:, 13 + foffset] = lowpch
        features_temp[:, 14 + foffset] = np.abs(high - low)
        features_temp[:, 15 + foffset] = np.abs(high - ground)
        coherence_location = (high + low) * np.conj(ground)
        features_temp[:, 16 + foffset] = np.real(coherence_location)
        features_temp[:, 17 + foffset] = np.imag(coherence_location)
        del coherence_location
        features_temp[:, 18 + foffset] = np.angle((high - low) * np.conj(ground))

        foffset += 19

    if include_inc:
        features_temp[:, foffset] = scene.inc[training_mask]
        foffset += 1

    if include_dem:
        features_temp[:, foffset] = scene.dem[training_mask]

    print('Extracted {} features from {} samples.'.format(num_features, num_samples))
    if features is None:
        features = features_temp.copy()
    else:
        features = np.append(features, features_temp, axis=0)

    # masked training data, if specified
    if training_data is not None:
        print('Extracted training data from {} samples.'.format(num_samples))
        if training is None:
            training = training_data[training_mask].flatten()
        else:
            training = np.append(training, training_data[training_mask], axis=0)

    return features, training


def default_feature_extractor_v2(scene, baseline_index=None, training_data=None,
                                 training_mask=None, include_inc=False,
                                 include_dem=False):
    """Function to extract a default PolInSAR feature set from a given Kapok
        Scene object.

        Updated version which includes parameters which most often made it
        through feature selection in Denbina et al., 2018.

        Feature List (19-21 Total Features):
         * Backscatter (3 pol)
         * Interferometric Covariance (3 pol, real and imag = 6 total)
         * kz (1)
         * High and Low Coherence Magnitude (2)
         * High and Low Coherence Phase Center Height Above Estimated Ground (2)
         * abs(High - Low) -- High and Low Coherence Separation (1)
         * abs(High - Ground) -- High and Ground Coherence Separation (1)
         * Coherence Region Centroid Location w/ Ground Phase Removed -- Real and Imaginary Parts (2)
         * Coherence Region Orientation w/ Ground Phase Removed (1)
         * Optionally: Incidence Angle and DEM (based on input arguments)

        Arguments:
            scene (kapok.Scene): Kapok Scene to extract PolInSAR features from.

    """
    if baseline_index is None:
        print('Baseline index not specified!  All baselines will be extracted.')
        baseline_index = np.arange(scene.num_baselines)
    elif not isinstance(baseline_index, (collections.abc.Sequence, np.ndarray)):
        baseline_index = np.array([baseline_index])

    if (training_mask is None) and (training_data is not None):
        print('Training data provided but no training mask specified.  All available data will be used for training.')
        training_mask = np.ones(scene.dim, dtype='bool')
    elif (training_mask) is None:
        # no training data, features are being extracted just for prediction, not for training
        # return features for all pixels
        training_mask = np.ones(scene.dim, dtype='bool')

    features = None
    training = None

    num_samples = np.sum(training_mask)
    num_features = 19
    if include_inc:
        num_features += 1
    if include_dem:
        num_features += 1

    for bl in baseline_index:
        print('Extracting features for baseline {}...'.format(bl))
        tr1, tr2 = mb_tr_index(bl) # tracks corresponding to this baseline

        features_temp = np.zeros((num_samples, num_features), dtype='float32')

        # backscatter
        features_temp[:, 0] = scene.power('hh', tr=tr1)[training_mask]
        features_temp[:, 1] = scene.power('hv', tr=tr1)[training_mask]
        features_temp[:, 2] = scene.power('vv', tr=tr1)[training_mask]

        # interferometric covariance matrix
        cov_row, cov_col = mb_cov_index(bl)
        features_temp[:, 3] = np.real(scene.cov[:, :, cov_row, cov_col])[training_mask]
        features_temp[:, 4] = np.real(scene.cov[:, :, cov_row+1, cov_col+1])[training_mask]
        features_temp[:, 5] = np.real(scene.cov[:, :, cov_row+2, cov_col+2])[training_mask]
        features_temp[:, 6] = np.imag(scene.cov[:, :, cov_row, cov_col])[training_mask]
        features_temp[:, 7] = np.imag(scene.cov[:, :, cov_row+1, cov_col+1])[training_mask]
        features_temp[:, 8] = np.imag(scene.cov[:, :, cov_row+2, cov_col+2])[training_mask]

        # kz and incidence angle
        kztemp = scene.kz(bl)[training_mask]
        kztemp = np.clip(kztemp, -1, 1)
        features_temp[:, 9] = kztemp

        # optimized coherences and coherence region parameters
        high = scene.coh('high',bl=bl)[training_mask]
        low = scene.coh('low',bl=bl)[training_mask]
        ground = scene.coh('ground',bl=bl)[training_mask]

        highpch = remove_negatives(np.angle(high*np.conj(ground))/kztemp, kztemp)
        highpch = np.clip(highpch, 0, 100)
        lowpch = remove_negatives(np.angle(low*np.conj(ground))/kztemp, kztemp)
        lowpch = np.clip(lowpch, 0, 100)

        features_temp[:, 10] = np.abs(high)
        features_temp[:, 11] = np.abs(low)
        features_temp[:, 12] = highpch
        features_temp[:, 13] = lowpch
        features_temp[:, 14] = np.abs(high - low)
        features_temp[:, 15] = np.abs(high - ground)
        coherence_location = (high + low) * np.conj(ground)
        features_temp[:, 16] = np.real(coherence_location)
        features_temp[:, 17] = np.imag(coherence_location)
        del coherence_location
        features_temp[:, 18] = np.angle((high - low) * np.conj(ground))

        feature_index = 19
        if include_inc:
            features_temp[:, feature_index] = scene.inc[training_mask]
            feature_index += 1

        if include_dem:
            features_temp[:, feature_index] = scene.dem[training_mask]

        # add features for this baseline to stack of all features
        print('Extracted {} features from {} samples.'.format(num_features, num_samples))
        if features is None:
            features = features_temp.copy()
        else:
            features = np.append(features, features_temp, axis=0)

        # masked training data, if specified
        if training_data is not None:
            print('Extracted training data from {} samples.'.format(num_samples))
            if training is None:
                training = training_data[training_mask].flatten()
            else:
                training = np.append(training, training_data[training_mask], axis=0)

    return features, training


def default_feature_extractor_v1(scene, baseline_index=None, training_data=None,
                                 training_mask=None):
    """Function to extract a default PolInSAR feature set from a given Kapok
        Scene object.

        Updated version which includes parameters which most often made it
        through feature selection in Denbina et al., 2018.

        Feature List (19 Total Features):
         * Backscatter (3 pol)
         * Interferometric Covariance (3 pol, real and imag = 6 total)
         * kz (1)
         * High and Low Coherence Magnitude (2)
         * High and Low Coherence Phase Center Height Above Estimated Ground (2)
         * abs(High - Low) -- High and Low Coherence Separation (1)
         * abs(High - Ground) -- High and Ground Coherence Separation (1)
         * Coherence Region Centroid Location w/ Ground Phase Removed -- Real and Imaginary Parts (2)
         * Coherence Region Orientation w/ Ground Phase Removed (1)

        Arguments:
            scene (kapok.Scene): Kapok Scene to extract PolInSAR features from.

    """
    if baseline_index is None:
        print('Baseline index not specified!  All baselines will be extracted.')
        baseline_index = np.arange(scene.num_baselines)
    elif not isinstance(baseline_index, (collections.abc.Sequence, np.ndarray)):
        baseline_index = np.array([baseline_index])

    if (training_mask is None) and (training_data is not None):
        print('Training data provided but no training mask specified.  All available data will be used for training.')
        training_mask = np.ones(scene.dim, dtype='bool')
    elif (training_mask) is None:
        # no training data, features are being extracted just for prediction, not for training
        # return features for all pixels
        training_mask = np.ones(scene.dim, dtype='bool')

    features = None
    training = None

    num_samples = np.sum(training_mask)
    num_features = 19

    for bl in baseline_index:
        print('Extracting features for baseline {}...'.format(bl))
        tr1, tr2 = mb_tr_index(bl) # tracks corresponding to this baseline

        features_temp = np.zeros((num_samples, num_features), dtype='float32')

        # backscatter
        features_temp[:, 0] = scene.power('hh', tr=tr1)[training_mask]
        features_temp[:, 1] = scene.power('hv', tr=tr1)[training_mask]
        features_temp[:, 2] = scene.power('vv', tr=tr1)[training_mask]

        # interferometric covariance matrix
        cov_row, cov_col = mb_cov_index(bl)
        features_temp[:, 3] = np.real(scene.cov[:, :, cov_row, cov_col])[training_mask]
        features_temp[:, 4] = np.real(scene.cov[:, :, cov_row+1, cov_col+1])[training_mask]
        features_temp[:, 5] = np.real(scene.cov[:, :, cov_row+2, cov_col+2])[training_mask]
        features_temp[:, 6] = np.imag(scene.cov[:, :, cov_row, cov_col])[training_mask]
        features_temp[:, 7] = np.imag(scene.cov[:, :, cov_row+1, cov_col+1])[training_mask]
        features_temp[:, 8] = np.imag(scene.cov[:, :, cov_row+2, cov_col+2])[training_mask]

        # kz and incidence angle
        kztemp = scene.kz(bl)[training_mask]
        kztemp = np.clip(kztemp, -1, 1)
        features_temp[:, 9] = kztemp

        # optimized coherences and coherence region parameters
        high = scene.coh('high',bl=bl)[training_mask]
        low = scene.coh('low',bl=bl)[training_mask]
        ground = scene.coh('ground',bl=bl)[training_mask]

        highpch = remove_negatives(np.angle(high*np.conj(ground))/kztemp, kztemp)
        highpch = np.clip(highpch, 0, 100)
        lowpch = remove_negatives(np.angle(low*np.conj(ground))/kztemp, kztemp)
        lowpch = np.clip(lowpch, 0, 100)

        features_temp[:, 10] = np.abs(high)
        features_temp[:, 11] = np.abs(low)
        features_temp[:, 12] = highpch
        features_temp[:, 13] = lowpch
        features_temp[:, 14] = np.abs(high - low)
        features_temp[:, 15] = np.abs(high - ground)
        coherence_location = (high + low) * np.conj(ground)
        features_temp[:, 16] = np.real(coherence_location)
        features_temp[:, 17] = np.imag(coherence_location)
        del coherence_location
        features_temp[:, 18] = np.angle((high - low) * np.conj(ground))

        # add features for this baseline to stack of all features
        print('Extracted {} features from {} samples.'.format(num_features, num_samples))
        if features is None:
            features = features_temp.copy()
        else:
            features = np.append(features, features_temp, axis=0)

        # masked training data, if specified
        if training_data is not None:
            print('Extracted training data from {} samples.'.format(num_samples))
            if training is None:
                training = training_data[training_mask].flatten()
            else:
                training = np.append(training, training_data[training_mask], axis=0)

    return features, training


def default_feature_extractor_v0_noinc(scene, baseline_index=None, training_data=None,
                                 training_mask=None):
    """Function to extract a default PolInSAR feature set from a given Kapok
        Scene object.

        Original version excluding incidence angle.

        Feature List (14 Total Features):
         * backscatter (3 pol)
         * interferometric covariance (3 pol, real and imag = 6 total)
         * kz
         * high and low coherence magnitude
         * high and low coherence phase center height above estimated ground

        Arguments:
            scene (kapok.Scene): Kapok Scene to extract PolInSAR features from.

    """
    if baseline_index is None:
        print('Baseline index not specified!  All baselines will be extracted.')
        baseline_index = np.arange(scene.num_baselines)
    elif not isinstance(baseline_index, (collections.abc.Sequence, np.ndarray)):
        baseline_index = np.array([baseline_index])

    if (training_mask is None) and (training_data is not None):
        print('Training data provided but no training mask specified.  All available data will be used for training.')
        training_mask = np.ones(scene.dim, dtype='bool')
    elif (training_mask) is None:
        # no training data, features are being extracted just for prediction, not for training
        # return features for all pixels
        training_mask = np.ones(scene.dim, dtype='bool')

    features = None
    training = None

    num_samples = np.sum(training_mask)
    num_features = 14

    for bl in baseline_index:
        print('Extracting features for baseline {}...'.format(bl))
        tr1, tr2 = mb_tr_index(bl) # tracks corresponding to this baseline

        features_temp = np.zeros((num_samples, num_features), dtype='float32')

        # backscatter
        features_temp[:, 0] = scene.power('hh', tr=tr1)[training_mask]
        features_temp[:, 1] = scene.power('hv', tr=tr1)[training_mask]
        features_temp[:, 2] = scene.power('vv', tr=tr1)[training_mask]

        # interferometric covariance matrix
        cov_row, cov_col = mb_cov_index(bl)
        features_temp[:, 3] = np.real(scene.cov[:, :, cov_row, cov_col])[training_mask]
        features_temp[:, 4] = np.real(scene.cov[:, :, cov_row+1, cov_col+1])[training_mask]
        features_temp[:, 5] = np.real(scene.cov[:, :, cov_row+2, cov_col+2])[training_mask]
        features_temp[:, 6] = np.imag(scene.cov[:, :, cov_row, cov_col])[training_mask]
        features_temp[:, 7] = np.imag(scene.cov[:, :, cov_row+1, cov_col+1])[training_mask]
        features_temp[:, 8] = np.imag(scene.cov[:, :, cov_row+2, cov_col+2])[training_mask]

        # kz and incidence angle
        kztemp = scene.kz(bl)
        kztemp = np.clip(kztemp, -1, 1)
        features_temp[:, 9] = kztemp[training_mask]

        # high and low optimized coherences
        high = scene.coh('high',bl=bl)
        low = scene.coh('low',bl=bl)
        ground = scene.coh('ground',bl=bl)

        highpch = remove_negatives(np.angle(high*np.conj(ground))/kztemp,kztemp)
        highpch = np.clip(highpch, 0, 100)
        lowpch = remove_negatives(np.angle(low*np.conj(ground))/kztemp, kztemp)
        lowpch = np.clip(lowpch, 0, 100)

        features_temp[:, 10] = np.abs(high)[training_mask]
        features_temp[:, 11] = np.abs(low)[training_mask]
        features_temp[:, 12] = highpch[training_mask]
        features_temp[:, 13] = lowpch[training_mask]

        # add features for this baseline to stack of all features
        print('Extracted {} features from {} samples.'.format(num_features, num_samples))
        if features is None:
            features = features_temp.copy()
        else:
            features = np.append(features, features_temp, axis=0)

        # masked training data, if specified
        if training_data is not None:
            print('Extracted training data from {} samples.'.format(num_samples))
            if training is None:
                training = training_data[training_mask].flatten()
            else:
                training = np.append(training, training_data[training_mask], axis=0)

    return features, training


def default_feature_extractor_v0(scene, baseline_index=None, training_data=None,
                                 training_mask=None):
    """Function to extract a default PolInSAR feature set from a given Kapok
        Scene object.

        Original version which includes incidence angle, but does not include
        coherence region parameters.

        Feature List (15 Total Features):
         * backscatter (3 pol)
         * interferometric covariance (3 pol, real and imag = 6 total)
         * kz
         * inc
         * high and low coherence magnitude
         * high and low coherence phase center height above estimated ground

        Arguments:
            scene (kapok.Scene): Kapok Scene to extract PolInSAR features from.

    """
    if baseline_index is None:
        print('Baseline index not specified!  All baselines will be extracted.')
        baseline_index = np.arange(scene.num_baselines)
    elif not isinstance(baseline_index, (collections.abc.Sequence, np.ndarray)):
        baseline_index = np.array([baseline_index])

    if (training_mask is None) and (training_data is not None):
        print('Training data provided but no training mask specified.  All available data will be used for training.')
        training_mask = np.ones(scene.dim, dtype='bool')
    elif (training_mask) is None:
        # no training data, features are being extracted just for prediction, not for training
        # return features for all pixels
        training_mask = np.ones(scene.dim, dtype='bool')

    features = None
    training = None

    num_samples = np.sum(training_mask)
    num_features = 15

    for bl in baseline_index:
        print('Extracting features for baseline {}...'.format(bl))
        tr1, tr2 = mb_tr_index(bl) # tracks corresponding to this baseline

        features_temp = np.zeros((num_samples, num_features), dtype='float32')

        # backscatter
        features_temp[:, 0] = scene.power('hh', tr=tr1)[training_mask]
        features_temp[:, 1] = scene.power('hv', tr=tr1)[training_mask]
        features_temp[:, 2] = scene.power('vv', tr=tr1)[training_mask]

        # interferometric covariance matrix
        cov_row, cov_col = mb_cov_index(bl)
        features_temp[:, 3] = np.real(scene.cov[:, :, cov_row, cov_col])[training_mask]
        features_temp[:, 4] = np.real(scene.cov[:, :, cov_row+1, cov_col+1])[training_mask]
        features_temp[:, 5] = np.real(scene.cov[:, :, cov_row+2, cov_col+2])[training_mask]
        features_temp[:, 6] = np.imag(scene.cov[:, :, cov_row, cov_col])[training_mask]
        features_temp[:, 7] = np.imag(scene.cov[:, :, cov_row+1, cov_col+1])[training_mask]
        features_temp[:, 8] = np.imag(scene.cov[:, :, cov_row+2, cov_col+2])[training_mask]

        # kz and incidence angle
        kztemp = scene.kz(bl)
        kztemp = np.clip(kztemp, -1, 1)
        features_temp[:, 9] = kztemp[training_mask]

        features_temp[:, 10] = scene.inc[training_mask]

        # high and low optimized coherences
        high = scene.coh('high',bl=bl)
        low = scene.coh('low',bl=bl)
        ground = scene.coh('ground',bl=bl)

        highpch = remove_negatives(np.angle(high*np.conj(ground))/kztemp,kztemp)
        highpch = np.clip(highpch, 0, 100)
        lowpch = remove_negatives(np.angle(low*np.conj(ground))/kztemp, kztemp)
        lowpch = np.clip(lowpch, 0, 100)

        features_temp[:, 11] = np.abs(high)[training_mask]
        features_temp[:, 12] = np.abs(low)[training_mask]
        features_temp[:, 13] = highpch[training_mask]
        features_temp[:, 14] = lowpch[training_mask]

        # add features for this baseline to stack of all features
        print('Extracted {} features from {} samples.'.format(num_features, num_samples))
        if features is None:
            features = features_temp.copy()
        else:
            features = np.append(features, features_temp, axis=0)

        # masked training data, if specified
        if training_data is not None:
            print('Extracted training data from {} samples.'.format(num_samples))
            if training is None:
                training = training_data[training_mask].flatten()
            else:
                training = np.append(training, training_data[training_mask], axis=0)

    return features, training


class FusionRegressor(object):
    """Object for performing fusion with other remote sensing data. """

    def __init__(self, feature_extractor=None, estimate_uncertainty=False,
                 loss_function=None, include_inc=False, include_dem=False,
                 batch_size=1024, epochs=200, learning_rate=0.0005,
                 epsilon=1e-2):
        """FusionRegressor initialization method.

        Arguments:
            feature_extractor (func): The function that will be used
                to extract the features from each scene.  If not specified,
                a default feature extractor from this module will be used.
                Note: If you wish to use your own feature extractor, look at
                kapok.fusion.default_feature_extractor_v* to see the forms of
                those functions.
            estimate_uncertainty (bool): Set to True to have the FusionRegressor
                output both a prediction and an uncertainty.  Set to False to
                output a prediction only.  If enabled, this will use a modified
                loss function, kapok.fusion.aleatoric_loss().  In this case,
                the loss will be a weighted mean square error, with weights
                derived from the network's uncertainty estimates, regularized
                by the mean uncertainty of the network.  If set to False, the
                mean square error loss function will be used.
            loss_function (func): Set to use a custom loss function.  If not
                set, a default loss function will be used, either mean square
                error if estimate_uncertainty is False, and
                kapok.fusion.aleatoric_loss() if estimate_uncertainty is True.
            include_inc (bool): Set to True to include incidence angle in
                features.  Only works for feature_extractor functions which
                support the same argument.
            include_dem (bool): Set to True to include DEM in features.  Only
                works for feature_extractor functions which support the same
                argument.

        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.estimate_uncertainty = estimate_uncertainty

        self.all_training_features = None
        self.all_training_values = None

        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = default_feature_extractor_v1

        if loss_function is not None:
            self.loss_function = loss_function
        elif estimate_uncertainty:
            self.loss_function = aleatoric_loss
        else:
            self.loss_function = 'mean_squared_error'

        self.include_inc = include_inc
        self.include_dem = include_dem

    def add(self, scene, **kwargs):
        """Add training data to the network.

            Arguments:
                scene (kapok.Scene): Kapok Scene object to add to the set of
                    data used to train the network.
                kwargs: Additional keyword arguments to be passed to the
                    feature_extractor() method of this FusionRegressor
                    object.

        """
        training_features, training_values = self.feature_extractor(scene, include_inc=self.include_inc, include_dem=self.include_dem, **kwargs)

        if self.all_training_features is None:
            self.all_training_features = training_features.copy()
            self.all_training_values = training_values.copy()
        else:
            self.all_training_features = np.append(self.all_training_features, training_features, axis=0)
            self.all_training_values = np.append(self.all_training_values, training_values, axis=0)

    def fit(self):
        """ Fit the network using the previously loaded training data.

            Arguments:
                None

        """
        self.num_training_samples = self.all_training_features.shape[0]
        self.num_features = self.all_training_features.shape[1]

        self.scaler = StandardScaler()
        self.scaler.fit(self.all_training_features)  # Don't cheat - fit only on training data
        x_train = self.scaler.transform(self.all_training_features)

        # dense_layer_sizes = np.array([30, 30, 30, 15])
        # dense_layer_sizes = np.array([20, 20, 20, 10])
        dense_layer_sizes = np.array([10, 10, 10, 5])
        dropout_factors = np.zeros(dense_layer_sizes.shape, dtype='float32')
        activation_function = 'relu'

        self.model = Sequential()
        self.model.add(Input((self.num_features), name='in'))
        self.model.add(Dense(self.num_features, activation=activation_function))

        for n, layer_size in enumerate(dense_layer_sizes):
            if dropout_factors[n] > 0:
                self.model.add(Dropout(dropout_factors[n]))
            self.model.add(Dense(int(self.num_features*layer_size), activation=activation_function))

        if self.estimate_uncertainty:
            self.model.add(Dense(2, kernel_initializer='normal')) # two outputs, an estimate and an uncertainty
        else:
            self.model.add(Dense(1, kernel_initializer='normal')) # one output, just the estimate

        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=self.epsilon)
        self.model.compile(loss=self.loss_function,
                           optimizer=opt,
                           metrics=['mae'])

        self.results = self.model.fit(x_train, self.all_training_values, epochs=self.epochs, batch_size=self.batch_size)

    def predict(self, scene, **kwargs):
        """Predict using the network.

            Arguments:
                scene (kapok.Scene): Kapok Scene object to use as input to the
                    prediction.
                kwargs: Additional keyword arguments to be passed to the
                    Regressor's feature_extractor method.  Usually, this
                    should be the baseline_index.

        """
        print('Predicting...')
        features, _ = self.feature_extractor(scene, include_inc=self.include_inc, include_dem=self.include_dem, **kwargs)
        ind_void = ~np.isfinite(features)
        if np.any(ind_void):
            features[ind_void] = np.nan # replace inf features with NaN
        features = self.scaler.transform(features)
        predicted = self.model.predict(features, batch_size=self.batch_size)

        if self.estimate_uncertainty:
            prediction = np.reshape(predicted[:, 0], scene.dim)
            uncertainty = np.reshape(predicted[:, 1], scene.dim)
            uncertainty = np.exp(uncertainty) # convert log uncertainty to actual variance
            return prediction, uncertainty
        else:
            prediction = np.reshape(predicted, scene.dim)
            return prediction

    def save(self, path):
        """Save this regressor and the associated Keras model to disk.

            Arguments:
                path (str): File path where the model will be saved.

        """
        self.model.save(path)
        scaler_file = os.path.join(path, 'scaler.pkl')
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)

    def load(self, path):
        """Load a regressor and the associated Keras model from disk.

            Arguments:
                path (str): File path where the model will be loaded from.

        """
        self.model = keras.models.load_model(path)
        scaler_file = os.path.join(path, 'scaler.pkl')
        with open(scaler_file, 'rb') as f:
            self.scaler = pickle.load(f)

    # def plot_network(self):
    #     plot_model(self.model, to_file=outpath+'keras_model.png', show_shapes=True, show_layer_names=False)
