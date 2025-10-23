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

## Quick Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Fix pylsd for Python 3 compatibility
python scripts/fix_pylsd.py

# Install the package
pip install -e .
```

## Alternative Installation Methods

### Development Mode (Recommended)
Install in editable mode for development - changes to Python files take effect immediately:
```bash
pip install -e .
```

### Production Mode
For a standard non-editable installation:
```bash
pip install .
```

**Note:** The older `python setup.py install` method is deprecated. Use pip instead.

## Note on pylsd Installation

The `pylsd` package on PyPI has Python 2 compatibility issues. After installing dependencies with `pip install -r requirements.txt`, you must run the fix script:

```bash
python scripts/fix_pylsd.py
```

This script automatically patches the installed `pylsd` package to work with Python 3 by:
- Converting absolute imports to relative imports
- Fixing Python 2 exception syntax to Python 3 syntax
- Fixing tempfile handling to use the system temp directory (prevents file write errors)

If you encounter any issues with `pylsd`, you can verify it's working by running:
```bash
python -c "import pylsd; from pylsd import lsd; print('pylsd is working correctly')"
```

# Example Usage

See the Jupyter notebook in the `examples/` directory for a demonstration of how to use the PyGCRANSAC package:
- `example_planar_affine_rectification.ipynb` - Demonstrates planar affine rectification using hybrid RANSAC

# Requirements

- Eigen 3.0 or higher
- CMake 2.8.12 or higher
- OpenCV 3.0 or higher
- A modern compiler with C++17 support

# Acknowledgements

When using the method for planar affine rectification using local scales and orientations, please cite
```
@inproceedings{
    PlanarAffRect2025,
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
