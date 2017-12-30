#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mock import patch

import numpy
import features

class TestFeaturesColorHistogram:
    def setup_method(self, method = None, w = 10, h = 10):
        self.h, self.w = h, w
        image = numpy.zeros((self.h, self.w, 3), dtype=numpy.uint8)
        label = numpy.zeros((self.h, self.w), dtype=int)
        self.f = features.Features(image, label, 1)

    def test_1region_1color(self):
        self.setup_method()
        hist = self.f._Features__init_color(1)
        assert len(hist) == 1
        assert hist[0].shape == (75,)
        r_expected = [0.333333333] + [0] * 24
        g_expected = [0.333333333] + [0] * 24
        b_expected = [0.333333333] + [0] * 24
        numpy.testing.assert_array_almost_equal(hist[0].ravel(), r_expected + g_expected + b_expected)

    def test_1region_255color(self):
        self.setup_method(w = 1, h = 256)
        for y in range(self.h):
            self.f.image[y, :, :] = y

        hist = self.f._Features__init_color(1)
        assert len(hist) == 1
        assert hist[0].shape == (75,)
        r_expected = [11] * 23 + [3, 0] # because bin width equals 11
        g_expected = [11] * 23 + [3, 0]
        b_expected = [11] * 23 + [3, 0]
        expected = numpy.array(r_expected + g_expected + b_expected)
        numpy.testing.assert_array_almost_equal(hist[0].ravel(), expected / float(numpy.sum(expected)))

    def test_2region_1color(self):
        self.setup_method(w = 1, h = 2)
        for y in range(self.h):
            self.f.label[y, :] = y

        hist = self.f._Features__init_color(2)
        assert len(hist) == 2
        assert hist[0].shape == (75,)
        r1_expected = ([1.0/3] + [0] * 24) + ([1.0/3] + [0] * 24) + ([1.0/3] + [0] * 24)
        r2_expected = ([1.0/3] + [0] * 24) + ([1.0/3] + [0] * 24) + ([1.0/3] + [0] * 24)
        numpy.testing.assert_array_almost_equal(hist[0].ravel(), r1_expected)
        numpy.testing.assert_array_almost_equal(hist[1].ravel(), r2_expected)


class TestFeaturesSize:
    def setup_method(self, method=None):
        image = numpy.zeros((10, 10, 3), dtype=numpy.uint8)
        label = numpy.zeros((10, 10), dtype=int)
        self.f = features.Features(image, label, 1)

    def test_1region(self):
        self.setup_method()
        sizes = self.f._Features__init_size(1)
        assert len(sizes) == 1
        assert sizes[0] == 100

    def test_2region(self):
        self.setup_method()
        self.f.label[:5, :] = 1
        sizes = self.f._Features__init_size(2)
        assert len(sizes) == 2
        assert sizes[0] == 50
        assert sizes[1] == 50

class TestFeaturesBoundingBox:
    def setup_method(self, method=None):
        image = numpy.zeros((10, 10, 3), dtype=numpy.uint8)
        label = numpy.zeros((10, 10), dtype=int)
        self.f = features.Features(image, label, 1)

    def test_1region(self):
        self.setup_method()
        bb = self.f._Features__init_bounding_box(1)
        assert len(bb) == 1
        assert bb[0] == (0, 0, 9, 9)

    def test_4region(self):
        self.setup_method()
        self.f.label[:5, :5] = 0
        self.f.label[:5, 5:] = 1
        self.f.label[5:, :5] = 2
        self.f.label[5:, 5:] = 3
        bb = self.f._Features__init_bounding_box(4)
        assert len(bb) == 4
        assert bb[0] == (0, 0, 4, 4)
        assert bb[1] == (0, 5, 4, 9)
        assert bb[2] == (5, 0, 9, 4)
        assert bb[3] == (5, 5, 9, 9)


class TestSimilarity:
    def setup_method(self, method=None):
        self.dummy_image = numpy.zeros((10, 10, 3), dtype=numpy.uint8)
        self.dummy_label = numpy.zeros((10, 10), dtype=int)
        self.f = features.Features(self.dummy_image, self.dummy_label, 1)

    def test_similarity_size(self):
        self.setup_method()
        self.f.size = {0 : 10, 1 : 20}

        s = self.f._Features__sim_size(0, 1)
        assert s == 0.7

    def test_similarity_color_simple(self):
        self.setup_method()
        self.f.color[0] = numpy.array([1] * 75)
        self.f.color[1] = numpy.array([2] * 75)
        s = self.f._Features__sim_color(0, 1)
        assert s == 75

    def test_similarity_color_complex(self):
        self.setup_method()
        # build 75-dimensional arrays as color histogram
        self.f.color[0] = numpy.array([1, 2, 1, 2, 1] * 15)
        self.f.color[1] = numpy.array([2, 1, 2, 1, 2] * 15)
        s = self.f._Features__sim_color(0, 1)
        assert s == 75

    def test_similarity_texture(self):
        self.setup_method()
        # build 240-dimensional arrays as texture histogram
        self.f.texture[0] = numpy.array([1, 2, 1, 2, 1, 2] * 40)
        self.f.texture[1] = numpy.array([2, 1, 2, 1, 2, 1] * 40)
        s = self.f._Features__sim_texture(0, 1)
        assert s == 240

    def test_similarity_fill(self):
        self.setup_method()
        self.f.bbox[0] = numpy.array([10, 10, 20, 20])
        self.f.size[0] = 100
        self.f.bbox[1] = numpy.array([20, 20, 30, 30])
        self.f.size[1] = 100
        s = self.f._Features__sim_fill(0, 1)
        assert s == 1. - float(400 - 200) / 100

    @patch.object(features.Features, '_Features__sim_size', lambda self, i, j: 1)
    @patch.object(features.Features, '_Features__sim_texture', lambda self, i, j: 1)
    @patch.object(features.Features, '_Features__sim_color', lambda self, i, j: 1)
    @patch.object(features.Features, '_Features__sim_fill', lambda self, i, j: 1)
    def test_similarity_user_all(self):
        self.setup_method()
        w = features.SimilarityMask(1, 1, 1, 1)
        f = features.Features(self.dummy_image, self.dummy_label, 1, w)
        assert f.similarity(0, 1) == 4


class TestMerge:
    def setup_method(self, method=None):
        dummy_image = numpy.zeros((10, 10, 3), dtype=numpy.uint8)
        dummy_label = numpy.zeros((10, 10), dtype=int)
        self.f = features.Features(dummy_image, dummy_label, 1)

    def test_merge_size(self):
        self.setup_method()
        self.f.size = {0: 10, 1: 20}
        self.f._Features__merge_size(0, 1, 2)
        assert self.f.size[2] == 30

    def test_merge_color(self):
        self.setup_method()
        self.f.color[0] = numpy.array([1.] * 75)
        self.f.size[0]  = 100
        self.f.color[1] = numpy.array([2.] * 75)
        self.f.size[1]  = 50
        self.f._Features__merge_color(0, 1, 2)

        expected = (100 * 1. + 50 * 2.) / (100 + 50)
        assert numpy.array_equal(self.f.color[2], [expected] * 75)

    def test_merge_texture(self):
        self.setup_method()
        self.f.texture[0] = numpy.array([1.] * 240)
        self.f.size[0]    = 100
        self.f.texture[1] = numpy.array([2.] * 240)
        self.f.size[1]    = 50
        self.f._Features__merge_texture(0, 1, 2)

        expected = (100 * 1. + 50 * 2.) / (100 + 50)
        assert numpy.array_equal(self.f.texture[2], [expected] * 240)

    def test_merge_bbox(self):
        self.setup_method()
        self.f.bbox[0] = numpy.array([10, 10, 20, 20])
        self.f.size[0] = 100
        self.f.bbox[1] = numpy.array([20, 20, 30, 30])
        self.f.size[1] = 50
        self.f.imsize  = 1000
        self.f._Features__merge_bbox(0, 1, 2)

        assert numpy.array_equal(self.f.bbox[2], [10, 10, 30, 30])

    def test_merge(self):
        self.setup_method()
        self.f.imsize  = 1000
        self.f.size    = {0: 10, 1: 20}
        self.f.color   = {0: numpy.array([1.] * 75), 1: numpy.array([2.] * 75)}
        self.f.texture = {0: numpy.array([1.] * 240), 1: numpy.array([2.] * 240)}
        self.f.bbox    = {0: numpy.array([10, 10, 20, 20]), 1: numpy.array([20, 20, 30, 30])}
        assert self.f.merge(0, 1) == 2
        assert len(self.f.size) == 3
        assert len(self.f.color) == 3
        assert len(self.f.texture) == 3
        assert len(self.f.bbox) == 3

