#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import itertools
import copy
import joblib
import numpy
import scipy.sparse
import segment
import collections
import skimage.io
import features
import color_space
import sys
sys.path.insert(0, '../../')
import src.localization.region_category_map as rcm
from sklearn.decomposition import PCA
from skimage.transform import rotate
import math

def _calc_adjacency_matrix(label_img, n_region):
    r = numpy.vstack([label_img[:, :-1].ravel(), label_img[:, 1:].ravel()])
    b = numpy.vstack([label_img[:-1, :].ravel(), label_img[1:, :].ravel()])
    t = numpy.hstack([r, b])
    A = scipy.sparse.coo_matrix((numpy.ones(t.shape[1]), (t[0], t[1])), shape=(n_region, n_region), dtype=bool).todense().getA()
    A = A | A.transpose()

    for i in range(n_region):
        A[i, i] = True

    dic = {i : {i} ^ set(numpy.flatnonzero(A[i])) for i in range(n_region)}

    Adjacency = collections.namedtuple('Adjacency', ['matrix', 'dictionary'])
    return Adjacency(matrix = A, dictionary = dic)

def _new_adjacency_dict(A, i, j, t):
    Ak = copy.deepcopy(A)
    Ak[t] = (Ak[i] | Ak[j]) - {i, j}
    del Ak[i], Ak[j]
    for (p, Q) in Ak.items():
        if i in Q or j in Q:
            Q -= {i, j}
            Q.add(t)

    return Ak

def _new_label_image(F, i, j, t):
    Fk = numpy.copy(F)
    Fk[Fk == i] = Fk[Fk == j] = t
    return Fk

def _build_initial_similarity_set(A0, feature_extractor):
    S = list()
    for (i, J) in A0.items():
        S += [(feature_extractor.similarity(i, j), (i, j)) for j in J if i < j]

    return sorted(S)

def _merge_similarity_set(feature_extractor, Ak, S, i, j, t):
    # remove entries which have i or j
    S = list(filter(lambda x: not(i in x[1] or j in x[1]), S))

    # calculate similarity between region t and its adjacencies
    St = [(feature_extractor.similarity(t, x), (t, x)) for x in Ak[t] if t < x] +\
         [(feature_extractor.similarity(x, t), (x, t)) for x in Ak[t] if x < t]

    return sorted(S + St)

def hierarchical_segmentation(I, k = 100, feature_mask = features.SimilarityMask(1, 1, 1, 1), eraseMap=None):
    # F0, n_region = segment.segment_label(I, 0.8, k, 100)
    F0, n_region = segment.segment_label(I, 0.5, k, 500)
    n_region = F0.max() + 1
    # ++ calculate outside region labels
    if eraseMap is not None:
        eraseLabels = set(list(F0[numpy.where(eraseMap == 1)].flatten()))
    else:
        eraseLabels = []
    adj_mat, A0 = _calc_adjacency_matrix(F0, n_region)
    feature_extractor = features.Features(I, F0, n_region)

    # stores list of regions sorted by their similarity
    S = _build_initial_similarity_set(A0, feature_extractor)
    # ++ pop out regions outside circle
    if eraseMap is not None:
        ii = len(S) - 1
        while ii >= 0:
            if (S[ii][1][0] in eraseLabels) or (S[ii][1][1] in eraseLabels):
                S.pop(ii)
            ii -= 1
    # stores region label and its parent (empty if initial).
    R = {i : () for i in range(n_region)}

    A = [A0]    # stores adjacency relation for each step
    F = [F0]    # stores label image for each step

    # greedy hierarchical grouping loop
    while len(S):
        (s, (i, j)) = S.pop()
        t = feature_extractor.merge(i, j)

        # record merged region (larger region should come first)
        R[t] = (i, j) if feature_extractor.size[j] < feature_extractor.size[i] else (j, i)

        Ak = _new_adjacency_dict(A[-1], i, j, t)
        A.append(Ak)

        S = _merge_similarity_set(feature_extractor, Ak, S, i, j, t)
        # ++ pop out regions outside circle
        if eraseMap is not None:
            ii = len(S) - 1
            while ii >= 0:
                if (S[ii][1][0] in eraseLabels) or (S[ii][1][1] in eraseLabels):
                    S.pop(ii)
                ii -= 1

        F.append(_new_label_image(F[-1], i, j, t))

    # bounding boxes for each hierarchy
    L = feature_extractor.bbox

    L_regions = {}
    for r, l in R.iteritems():
        if r < n_region:
            L_regions[r] = [r]
        else:
            ll = []
            if l[0] >= n_region:
                ll = ll + L_regions[l[0]]
            else:
                ll.append(l[0])

            if l[1] >= n_region:
                ll = ll + L_regions[l[1]]
            else:
                ll.append(l[1])
            L_regions[r] = ll
    return (R, F, L, L_regions, eraseLabels)

def hierarchical_segmentation_M(paras, feature_mask = features.SimilarityMask(1, 1, 1, 1)):
    # F0, n_region = segment.segment_label(I, 0.8, k, 100)
    # F0, n_region = segment.segment_label(I, 0.5, k, 500)
    # ++ calculate outside region labels
    train = paras['train']
    is_rotate = paras['is_rotate']
    if train:
        eraseRegionLabels = paras['eraseRegionLabels']
        F0, region_labels, eraseLabels = rcm.region_special_map(paras)
        # regions_special = []
        # regions_rest = []
        # regions_common = []
        # for _, v in region_labels.iteritems():
        #     regions_special = regions_special + v[0]
        #     regions_rest = regions_rest + v[1]
        #     regions_common = regions_common + v[2]
        coordinates_special = numpy.zeros((0, 2))
        print region_labels
        for ls in region_labels.values()[0][0]:
            # print 'ls: ' + str(ls)
            coordinates_special = numpy.vstack([coordinates_special, numpy.argwhere(F0==ls)])
        pca = PCA()
        pca.fit(coordinates_special)

        components = pca.components_
        main_ax = components[0]
        angle = math.atan(main_ax[0] / main_ax[1]) * (180.0 / math.pi)
        # print 'F0 before: '
        # print F0.dtype
        # eraseLabels = list(eraseLabels)
        # el = eraseLabels[0]
        # for ell in eraseLabels:
        #     F0[numpy.where(F0==ell)] = el
        # F0 = numpy.array(rotate(F0, angle).round(), dtype='i')
        # print 'F0 after: '
        # print F0.dtype
        n_region = len(set(list(F0.flatten())))

        I = paras['im']
        # print 'I before: '
        # print I.dtype
        # I = numpy.array(rotate(I, angle), dtype='i')
        # print 'I after: '
        # print I.dtype

        eraseRegions_fb = {'fg': [], 'bg': []}
        for rl in range(3):
            if rl in eraseRegionLabels:
                eraseRegions_fb['fg'] = eraseRegions_fb['fg'] + region_labels.values()[0][rl]
            else:
                eraseRegions_fb['bg'] = eraseRegions_fb['bg'] + region_labels.values()[0][rl]
        Rd = {}
        Fd = {}
        Ld = {}
        L_regionsd = {}
        for fb_k, fb_v in eraseRegions_fb.iteritems():
            print 'n_region', n_region
            print fb_k, fb_v
            eraseLabels_k = set(fb_v + list(eraseLabels))

            adj_mat, A0 = _calc_adjacency_matrix(F0, n_region)
            feature_extractor = features.Features(I, F0, n_region)

            # stores list of regions sorted by their similarity
            S = _build_initial_similarity_set(A0, feature_extractor)
            # ++ pop out regions outside circle and selected common
            ii = len(S) - 1
            while ii >= 0:
                if (S[ii][1][0] in eraseLabels_k) or (S[ii][1][1] in eraseLabels_k):
                    S.pop(ii)
                ii -= 1
            # stores region label and its parent (empty if initial).
            R = {i : () for i in range(n_region)}

            A = [A0]    # stores adjacency relation for each step
            F = [F0]    # stores label image for each step

            # greedy hierarchical grouping loop
            while len(S):
                (s, (i, j)) = S.pop()
                t = feature_extractor.merge(i, j)

                # record merged region (larger region should come first)
                R[t] = (i, j) if feature_extractor.size[j] < feature_extractor.size[i] else (j, i)

                Ak = _new_adjacency_dict(A[-1], i, j, t)
                A.append(Ak)

                S = _merge_similarity_set(feature_extractor, Ak, S, i, j, t)
                # ++ pop out regions outside circle and selected common
                ii = len(S) - 1
                while ii >= 0:
                    if (S[ii][1][0] in eraseLabels_k) or (S[ii][1][1] in eraseLabels_k):
                        S.pop(ii)
                    ii -= 1

                F.append(_new_label_image(F[-1], i, j, t))

            # bounding boxes for each hierarchy
            L = feature_extractor.bbox

            L_regions = {}
            for r, l in R.iteritems():
                if r < n_region:
                    L_regions[r] = [r]
                else:
                    ll = []
                    if l[0] >= n_region:
                        ll = ll + L_regions[l[0]]
                    else:
                        ll.append(l[0])

                    if l[1] >= n_region:
                        ll = ll + L_regions[l[1]]
                    else:
                        ll.append(l[1])
                    L_regions[r] = ll
            Rd[fb_k] = R
            Fd[fb_k] = F
            Ld[fb_k] = L
            L_regionsd[fb_k] = L_regions
        return (Rd, Fd, Ld, L_regionsd, eraseRegions_fb)
    else:
        # if is_rotate:
        #     F0, region_labels, eraseLabels, special_labels = rcm.region_special_map(paras)
        #     coordinates_special = numpy.zeros((0, 2))
        #     for ls in special_labels:
        #         print 'ls: ' + str(ls)
        #         coordinates_special = numpy.vstack([coordinates_special, numpy.argwhere(F0 == ls)])
        #     print 'coordinates_special', coordinates_special
        #     pca = PCA()
        #     pca.fit(coordinates_special)
        #
        #     components = pca.components_
        #     main_ax = components[0]
        #     angle = math.atan(main_ax[0] / main_ax[1]) * (180.0 / math.pi)
        # else:
        #     F0, region_labels, eraseLabels = rcm.region_special_map(paras)

        F0, region_labels, eraseLabels, _ = rcm.region_special_map(paras)
        eraseLabels = set(region_labels + list(eraseLabels))
        n_region = len(set(list(F0.flatten())))
        I = paras['im']

        special_labels = [x for x in range(n_region) if x not in eraseLabels]
        # print special_labels
        if len(special_labels) == 0:
            angle = 0
        else:
            coordinates_special = numpy.zeros((0, 2))
            for ls in special_labels:
                # print 'ls: ' + str(ls)
                coordinates_special = numpy.vstack([coordinates_special, numpy.argwhere(F0 == ls)])
            # print 'coordinates_special', coordinates_special
            pca = PCA()
            pca.fit(coordinates_special)

            components = pca.components_
            main_ax = components[0]
            angle = math.atan(main_ax[0] / main_ax[1]) * (180.0 / math.pi)
        # if eraseMap is not None:
        #     eraseLabels = set(list(F0[numpy.where(eraseMap == 1)].flatten()))
        adj_mat, A0 = _calc_adjacency_matrix(F0, n_region)
        feature_extractor = features.Features(I, F0, n_region)

        # stores list of regions sorted by their similarity
        S = _build_initial_similarity_set(A0, feature_extractor)
        # ++ pop out regions outside circle and selected common
        ii = len(S) - 1
        while ii >= 0:
            if (S[ii][1][0] in eraseLabels) or (S[ii][1][1] in eraseLabels):
                S.pop(ii)
            ii -= 1
        # stores region label and its parent (empty if initial).
        R = {i : () for i in range(n_region)}

        A = [A0]    # stores adjacency relation for each step
        F = [F0]    # stores label image for each step

        # greedy hierarchical grouping loop
        while len(S):
            (s, (i, j)) = S.pop()
            t = feature_extractor.merge(i, j)

            # record merged region (larger region should come first)
            R[t] = (i, j) if feature_extractor.size[j] < feature_extractor.size[i] else (j, i)

            Ak = _new_adjacency_dict(A[-1], i, j, t)
            A.append(Ak)

            S = _merge_similarity_set(feature_extractor, Ak, S, i, j, t)
            # ++ pop out regions outside circle and selected common
            ii = len(S) - 1
            while ii >= 0:
                if (S[ii][1][0] in eraseLabels) or (S[ii][1][1] in eraseLabels):
                    S.pop(ii)
                ii -= 1

            F.append(_new_label_image(F[-1], i, j, t))

        # bounding boxes for each hierarchy
        L = feature_extractor.bbox

        L_regions = {}
        for r, l in R.iteritems():
            if r < n_region:
                L_regions[r] = [r]
            else:
                ll = []
                if l[0] >= n_region:
                    ll = ll + L_regions[l[0]]
                else:
                    ll.append(l[0])

                if l[1] >= n_region:
                    ll = ll + L_regions[l[1]]
                else:
                    ll.append(l[1])
                L_regions[r] = ll
        if is_rotate:
            return (R, F, L, L_regions, eraseLabels, angle)
        else:
            return (R, F, L, L_regions, eraseLabels)

def _generate_regions(R_fb, L_fb, L_regions_fb, eraseLables_fb=None):
    # print R
    # print L
    # print L_regions
    if len(R_fb) == 2:
        regions_fb = {}
        for k in R_fb.keys():
            R = R_fb[k]
            L = L_fb[k]
            L_regions = L_regions_fb[k]
            eraseLables = eraseLables_fb[k]
            n_ini = sum(not parent for parent in R.values())
            n_all = len(R)

            regions = list()
            regions_l = list()
            for label in R.keys():
                i = min(n_all - n_ini + 1, n_all - label)
                vi = numpy.random.rand() * i
                # regions.append((vi, L[i], L_regions[i]))
                if eraseLables is not None:
                    if label not in eraseLables:
                        regions.append((vi, L[label], L_regions[label]))
                else:
                    regions.append((vi, L[label], L_regions[label]))
            regions_fb[k] = sorted(regions)
        return regions_fb
    else:
        R = R_fb
        L = L_fb
        L_regions = L_regions_fb
        eraseLables = eraseLables_fb
        n_ini = sum(not parent for parent in R.values())
        n_all = len(R)

        regions = list()
        for label in R.keys():
            i = min(n_all - n_ini + 1, n_all - label)
            vi = numpy.random.rand() * i
            # regions.append((vi, L[i], L_regions[i]))
            if eraseLables is not None:
                if label not in eraseLables:
                    regions.append((vi, L[label], L_regions[label]))
            else:
                regions.append((vi, L[label], L_regions[label]))
        return sorted(regions)

def _selective_search_one(I, color, k, mask, eraseMap):
    I_color = color_space.convert_color(I, color)
    (R, F, L, L_regions, eraseLabels) = hierarchical_segmentation(I_color, k, mask, eraseMap)
    return _generate_regions(R, L, L_regions, eraseLabels), F[0]

def _selective_search_one_M(paras, color, k, mask):
    paras['k'] = k
    I = paras['im']
    I_color = color_space.convert_color(I, color)
    paras['im'] = I_color
    train = paras['train']
    is_rotate = paras['is_rotate']
    if is_rotate:
        (R, F, L, L_regions, eraseLabels, angle) = hierarchical_segmentation_M(paras, mask)
    else:
        (R, F, L, L_regions, eraseLabels) = hierarchical_segmentation_M(paras, mask)
    if train:
        return _generate_regions(R, L, L_regions, eraseLabels), F.values()[0][0]
    else:
        if is_rotate:
            return _generate_regions(R, L, L_regions, eraseLabels), F[0], angle
        else:
            return _generate_regions(R, L, L_regions, eraseLabels), F[0]

def selective_search(I, color_spaces = ['rgb'], ks = [100], feature_masks = [features.SimilarityMask(1, 1, 1, 1)], eraseMap=None, n_jobs = -1):
    parameters = itertools.product(color_spaces, ks, feature_masks)
    region_set = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_selective_search_one)(I, color, k, mask, eraseMap) for (color, k, mask) in parameters)

    # region_set = region_set[0:len(region_set):4]

    #flatten list of list of tuple to list of tuple
    # regions = sum(region_set, [])
    return region_set  #  sorted(regions), region_labels

def selective_search_M(paras, n_jobs = -1):
    color_spaces = paras['color_spaces']
    ks = paras['ks']
    feature_masks = paras['feature_masks']
    parameters = itertools.product(color_spaces, ks, feature_masks)
    region_set = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_selective_search_one_M)(paras, color, k, mask) for (color, k, mask) in parameters)

    # region_set = region_set[0:len(region_set):4]

    #flatten list of list of tuple to list of tuple
    # regions = sum(region_set, [])
    return region_set  #  sorted(regions), region_labels