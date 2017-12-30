# Overview

This is a python implementation of Selective Search [[1]](#selective_search_ijcv)[[2]](#selective_search_iccv). It is forked from [belltailjp/selective_search.py](https://github.com/belltailjp/selective_search_py) to backport it from Python 3 to Python 2.7.

The Selective Search is used as a preprocess of object detection/recognition pipeline.<br/>
It finds regions likely to contain any objects from an input image regardless of its scale and location,
that allows detectors to concentrate only for such 'prospective' regions.<br/>
Therefore you can configure more computationally efficient detector,
or use more rich feature representation and classification method [[3]](#deeplearning)
compared to the conventional exhaustive search scheme.

For more details about the method, please refer the original paper.

This implementation is based on the journal edition of the original paper, and giving similar parameter variations.

![segmentation example](doc/segmentation_example.png)
![selective search example](doc/ss_sample.png)


# Requirements

* CMake (>= 3.3.2)
* GCC (>= 4.8.2)
* Python 2.7
    * For required packages, see `requirements.txt`
* Boost (>= 1.58.0) built with python support
    * If you get errors building the C++ for selective_search on Mac OS X you should install boost and boost-python via [brew](http://brew.sh/), compile them from source, and have them generate universal binaries:
      * `brew install --universal --build-from-source -vd boost`
      * `brew install --universal --build-from-source -vd boost-python`
* [Boost.NumPy](https://github.com/ndarray/Boost.NumPy)
    * If you got an error building on Linux, see [belltailjp/Boost.NumPy](https://github.com/belltailjp/Boost.NumPy)); if you got an error building on Mac OS X, you probably need to generate a universal binary, see [https://github.com/BradNeuberg/Boost.NumPy])(https://github.com/BradNeuberg/Boost.NumPy) for a Mac OS X-specific fork of Boost.Numpy based on cmake.


# Preparation

This implementation contains some C++ code which wraps the Efficient Graph-Based Image Segmentation [[4]](#segmentation) tool used for generating an initial value.
It works as a python module, so build it first.

```sh
% git clone https://github.com/belltailjp/selective_search_py.git
% cd selective_search_py
% wget http://cs.brown.edu/~pff/segment/segment.zip; unzip segment.zip; rm segment.zip
% cmake .
% make
```

Then you will see a shared object `segment.so` in the directory. If you are on Mac OS X you will see `segment.dylib` -- you must manually move this over to be `segment.so` to work correctly.
Keep it on the same directory of main Python script, or referrable location described in `LD_LIBRARY_PATH` on Linux or `DYLD_FALLBACK_LIBRARY_PATH` on Mac OS X.


# Demo

## Interactively show regions likely to contain objects

*showcandidate* demo allows you to interactively see the result of selective search.

```sh
% ./demo_showcandidates.py --image image.jpg
```

![showcandidate GUI example](doc/showcandidates_scr.png)

You can choose any combination of parameters on the left side of the screen.
Then click the "Run" button and wait for a while. You will see the generated regions on the right side.

By changing the slider on the bottom, you can increase/decrease number of region candidates.
The more slider goes to left, the more confident regions are shown like this:

![showcandidate GUI example more region](doc/showcandidates_scr_more.png)


## Show image segmentation hierarchy

*showhierarchy* demo visualizes colored region images for each step in iteration.

```sh
% ./demo_showhierarchy.py image.jpg --k 500 --feature color texture --color rgb
```

![image segmentation hierarchy visualization](doc/hierarchy_example.gif)

If you want to see labels composited with the input image, give a particular alpha-value.

```sh
% ./demo_showhierarchy.py image.jpg --k 500 --feature color texture --color rgb --alpha 0.6
```

![image hierarchy with original image](doc/hierarchy_example_composited.png)


# Implementation

## Overview

Algorithm of the method is described in Journal edition of the original paper in detail ([[1]](#selective_search_ijcv)).
For diversification strategy, this implementation supports to vary the following parameter as the original paper proposed.

* Color space
    * *RGB, Lab, rgI, HSV, normalized RGB* and *Hue*
    * *C* of Color invariance [[5]](#color_invariance) is currently not supported.
* Similarity measure
    * Texture, Color, Fill and Size
* Initial segmentation parameter *k*
    * As the initial (fine-grained) segmentation, this implementation uses [[4]](#segmentation). *k* is one of the parameters of the method.

You can give any combinations for each strategy.


## How to integrate into your code

If you just want to use this implementation as a black box, only the `selective_search` module is necessary to import.

```python
from selective_search import *

img = skimage.io.imread('image.png')
regions = selective_search(img)
for v, (i0, j0, i1, j1) in regions:
    ...
```

Then you can get a list regions sorted by score in ascending order.
Regions with larger score (latter elements of the list) are considered as 'non-prospective' regions, so they can be filtered out as you need.

To change parameters, just give a list of values for each diversification strategy. Note that they must be given as a list.
`selective_search` returns a single list of generated regions which contains every combination of selective search result.
This result is also sorted.

```python
regions = selective_search(img, \
                           color_spaces = ['rgb', 'hsv'],\  #color space. should be lower case.
                           ks = [50, 150, 300],\            #k.
                           feature_masks = [(0, 0, 1, 1)])  #indicates whether S/C/T/F similarity is used, respectively.
```


## Test

This implementation contains automated unit tests using [Nose](https://nose.readthedocs.org/en/latest/).

To execute the full tests, type:

```sh
% nosetests
```


# License

This implementation is publicly available under the MIT license. See LICENSE.txt for more details.

However regarding the selective search method itself, authors of the original paper have not mention anything so far.
Please ask the original authors if you have any concens.


# References

\[1\] <a name="selective_search_ijcv"> [J. R. R. Uijlings et al., Selective Search for Object Recognition, IJCV, 2013](https://ivi.fnwi.uva.nl/isis/publications/bibtexbrowser.php?key=UijlingsIJCV2013&bib=all.bib) <br/>
\[2\] <a name="selective_search_iccv"> [Koen van de Sande et al., Segmentation As Selective Search for Object Recognition, ICCV, 2011](https://ivi.fnwi.uva.nl/isis/publications/bibtexbrowser.php?key=UijlingsIJCV2013&bib=all.bib) <br/>
\[3\] <a name="deeplearning"> [R. Girshick et al., Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation, CVPR, 2014](http://www.cs.berkeley.edu/~rbg/papers/r-cnn-cvpr.pdf) <br/>
\[4\] <a name="segmentation"> [P. Felzenszwalb et al., Efficient Graph-Based Image Segmentation, IJCV, 2004](http://cs.brown.edu/~pff/segment/) <br/>
\[5\] <a name="color_invariance"> J. M. Geusebroek et al., Color invariance, TPAMI, 2001
