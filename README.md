# Planar Affine Rectification from Local Change of Scale and Orientation

Here you can find an implementation of the hybrid RANSAC framework proposed in paper:
Yuval Nissan, Prof. Marc Pollefeys, and Dr. Daniel Barath; Planar Affine Rectification from Local Change of Scale and Orientation,
Internation Conference on Computer Vision (ICCV), 2025.

The implementation is based on the Graph-Cut RANSAC (GCRANSAC) algorithm proposed in paper: [Daniel Barath and Jiri Matas; Graph-Cut RANSAC, Conference on Computer Vision and Pattern Recognition, 2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Barath_Graph-Cut_RANSAC_CVPR_2018_paper.pdf).

This repository is a fork of the original [GCRANSAC repository](https://github.com/danini/graph-cut-ransac).
Adapting it to allow estimation of a single model using multiple feature classes, in a hybrid fashion, broke many of the features available in the original GCRANSAC repository, such as most of the scoring functions, samplers, or any feature that is predicated on the existence of a single feature class.

The current implementation of the hybrid RANSAC framework works only with the MSAC scoring function (adapted to work with multiple inlier sets from the different feature classes), and with the uniform sampler.

# Installation C++

To build and install C++ only `GraphCutRANSAC`, clone or download this repository and then build the project by CMAKE. 
```shell
$ git clone https://github.com/ynissan/graph-cut-ransac
$ cd build
$ cmake ..
$ make
```

# Install Python package and compile C++

```bash
python3 ./setup.py install
```

or

```bash
pip3 install -e .
```

# Example project

To build the sample project showing examples of fundamental matrix, homography and essential matrix fitting, set variable `CREATE_SAMPLE_PROJECT = ON` when creating the project in CMAKE. 
Then 
```shell
$ cd build
$ ./SampleProject
```

# Requirements

- Eigen 3.0 or higher
- CMake 2.8.12 or higher
- OpenCV 3.0 or higher
- A modern compiler with C++17 support

# Acknowledgements

When using the method for planar affine rectification using local scales and orientations, please cite
```
@inproceedings{
    HybridRansac2025,
    author = {Nissan, Yuval and Pollefeys, Marc and Barath, Daniel},
    title = {Planar Affine Rectification from Local Change of Scale and Orientation},
    booktitle = {International Conference on Computer Vision},
    year = {2025}
}
```

If the GCRANSAC-based implementation of the method is used, please cite
```
@inproceedings{GCRansac2018,
	author = {Barath, Daniel and Matas, Jiri},
	title = {Graph-cut {RANSAC}},
	booktitle = {Conference on Computer Vision and Pattern Recognition},
	year = {2018},
}
```

The Python wrapper part is based on the great [Benjamin Jack `python_cpp_example`](https://github.com/benjaminjack/python_cpp_example).
