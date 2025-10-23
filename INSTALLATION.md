# Installation Guide

This guide provides step-by-step instructions for installing the Graph-Cut RANSAC library and its dependencies.

## System Requirements

- Linux, macOS, or Windows
- Python 3.8 or higher
- CMake 2.8.12 or higher
- A modern C++ compiler with C++17 support
- Eigen 3.0 or higher
- OpenCV 3.0 or higher

## Python Installation

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/ynissan/graph-cut-ransac
cd graph-cut-ransac

# 2. Create and activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Fix pylsd for Python 3 compatibility
python scripts/fix_pylsd.py

# 5. Build and install the package
pip install -e .
```

### Detailed Steps

#### 1. Install System Dependencies

On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential libeigen3-dev libopencv-dev
```

On macOS:
```bash
brew install cmake eigen opencv
```

#### 2. Install Python Dependencies

The project requires several Python packages listed in `requirements.txt`:
- `numpy>=2.2.0` - Numerical computing
- `opencv-contrib-python>=4.11.0` - Computer vision functions
- `matplotlib>=3.10.0` - Plotting and visualization
- `jupyter>=1.1.0` - Interactive notebooks
- `pylsd>=0.0.1` - Line segment detection

Install them with:
```bash
pip install -r requirements.txt
```

#### 3. Fix pylsd Package

**Important:** The `pylsd` package on PyPI was designed for Python 2 and has compatibility issues with Python 3. After installing the requirements, you must run the fix script:

```bash
python scripts/fix_pylsd.py
```

This script automatically applies the following fixes to the installed `pylsd` package:
- Converts Python 2 absolute imports to Python 3 relative imports
- Updates Python 2 exception syntax (`except Exception, e:`) to Python 3 syntax (`except Exception as e:`)
- Fixes tempfile handling to use the system's temp directory instead of the current working directory (resolves "cannot open file" errors)

**Verification:**
```bash
python -c "import pylsd; from pylsd import lsd; print('pylsd is working correctly')"
```

If you see the success message, pylsd is ready to use.

#### 4. Build and Install Graph-Cut RANSAC

**Recommended: Editable/Development Mode**
```bash
pip install -e .
```

This command builds the C++ extensions and installs the Python package in editable mode, which means:
- Changes to Python source files take effect immediately without reinstalling
- Perfect for development, debugging, and testing
- C++ extensions are still properly compiled
- Easy to uninstall with `pip uninstall pygcransac`

**Alternative: Production Mode**
```bash
pip install .
```

Use this for a standard installation when you don't need editable mode.

**Important:** Do not use `python setup.py install` - this method is deprecated and not recommended by the Python packaging community. Always use pip instead.

## C++ Only Installation

If you only need the C++ library without Python bindings:

```bash
git clone https://github.com/ynissan/graph-cut-ransac
cd graph-cut-ransac
mkdir build
cd build
cmake ..
make
```

## Troubleshooting

### pylsd Import Error

**Problem:** `ModuleNotFoundError: No module named 'lsd'` when trying to import pylsd.

**Solution:**
1. Ensure you've run the fix script:
   ```bash
   python scripts/fix_pylsd.py
   ```

2. If the problem persists, reinstall pylsd and reapply the fix:
   ```bash
   pip uninstall -y pylsd
   pip install 'pylsd @ https://files.pythonhosted.org/packages/8d/ac/128158438742944b170ebf81ec6d913df382387f782f87b0ca4bc51b291c/pylsd-0.0.2.tar.gz'
   python scripts/fix_pylsd.py
   ```

### CMake Can't Find Eigen or OpenCV

**Problem:** CMake configuration fails with "Could not find Eigen3" or "Could not find OpenCV".

**Solution:**
- Make sure the libraries are installed (see step 1 above)
- On Linux, you may need to install the development packages:
  ```bash
  sudo apt-get install libeigen3-dev libopencv-dev
  ```

### Build Errors

**Problem:** Compilation fails with C++ errors.

**Solution:**
- Ensure you have a C++17 compatible compiler
- On older systems, you may need to install a newer compiler:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install g++-9
  export CXX=g++-9
  ```

### Virtual Environment Issues

**Problem:** Package installed but can't be imported.

**Solution:**
- Make sure your virtual environment is activated:
  ```bash
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```
- Verify you're using the correct Python:
  ```bash
  which python  # Should point to your venv
  ```

## Why Is the pylsd Fix Necessary?

The `pylsd` package (version 0.0.2) was published in 2017 and was written for Python 2. It contains several Python 2 specific syntax features and design issues that are incompatible with Python 3:

1. **Absolute imports**: Python 2 allowed `from module import function` for sibling modules, but Python 3 requires relative imports like `from .module import function` within packages.

2. **Exception syntax**: Python 2 used `except Exception, e:` while Python 3 requires `except Exception as e:`.

3. **Tempfile handling**: The original code creates temporary files in the current working directory, which can fail due to:
   - Permission issues in read-only directories
   - Path encoding problems on Windows
   - Cross-platform compatibility issues
   
   The fix uses Python's `tempfile` module to create files in the system's temp directory, which is guaranteed to be writable and properly handles path encoding. This solution was contributed by the community ([GitHub Issue #14](https://github.com/primetang/pylsd/issues/14)).

Since the package hasn't been updated on PyPI, we provide an automated fix script that patches the installed package. This is a user-friendly solution that:
- Doesn't require manual file editing
- Can be easily reapplied if the package is reinstalled
- Is documented in the project README and examples
- Includes all necessary fixes for both import and runtime issues

## Alternative: Using a Fork

If you prefer not to use the fix script, you can use a maintained fork or alternative package. However, for consistency with the examples and to ensure compatibility, we recommend following the installation steps above.

## Verification

After installation, verify everything is working:

```bash
# Activate virtual environment
source venv/bin/activate

# Test Python imports
python -c "import numpy; import cv2; import matplotlib; print('✓ Core dependencies OK')"
python -c "import pylsd; from pylsd import lsd; print('✓ pylsd OK')"
python -c "import pygcransac; print('✓ pygcransac OK')"

# Run example (if you have an image)
jupyter notebook examples/example_affine_rectification.ipynb
```

## Getting Help

If you encounter issues not covered in this guide:
1. Check the [GitHub Issues](https://github.com/ynissan/graph-cut-ransac/issues)
2. Review the [Examples README](examples/README.md)
3. Open a new issue with:
   - Your operating system and version
   - Python version (`python --version`)
   - Error messages and logs
   - Steps to reproduce the problem

