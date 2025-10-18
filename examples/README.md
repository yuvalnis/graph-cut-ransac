# Graph-Cut RANSAC Examples

This directory contains Jupyter notebook demos for the PyGCRANSAC library.

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

From the project root directory:

```bash
pip install -r requirements.txt
```

### 3. Fix pylsd for Python 3

The `pylsd` package requires a compatibility fix. Run the provided script:

```bash
python scripts/fix_pylsd.py
```

This automatically patches the installed `pylsd` package to work with Python 3.

### 4. Install PyGCRANSAC

Build and install the library from source:

```bash
pip install .
```

### 5. Launch Jupyter

```bash
jupyter notebook examples/
```

## Available Notebooks

### `example_affine_rectification.ipynb`

Demonstrates affine rectification using:
- Scale features from SIFT keypoints
- Orientation features from SIFT and Line Segment Detector (LSD)
- Graph-Cut RANSAC for robust homography estimation

**Requirements:** An image with visible texture patterns (e.g., tiled floor, brick wall, checkerboard).

## Troubleshooting

### Missing Image Error

If you see a `FileNotFoundError` when running a notebook, make sure to:
1. Update the `IMAGE_PATH` variable in the notebook with your own image path
2. Ensure the image file exists and is readable

### Import Errors

If you get import errors for `pygcransac`:
- Ensure you've run `pip install .` from the project root
- Make sure you're using the correct virtual environment

### OpenCV Contrib Issues

If you need full OpenCV functionality (including SIFT), ensure you have `opencv-contrib-python` installed:
```bash
pip install opencv-contrib-python>=4.11.0
```

### pylsd Import Errors

If you get `ModuleNotFoundError: No module named 'lsd'` when importing `pylsd`:

1. Make sure you ran the fix script after installing dependencies:
   ```bash
   python scripts/fix_pylsd.py
   ```

2. Verify the fix was applied:
   ```bash
   python -c "import pylsd; from pylsd import lsd; print('✓ pylsd working')"
   ```

3. If issues persist, you can manually reinstall with the fix:
   ```bash
   pip uninstall -y pylsd
   pip install 'pylsd @ https://files.pythonhosted.org/packages/8d/ac/128158438742944b170ebf81ec6d913df382387f782f87b0ca4bc51b291c/pylsd-0.0.2.tar.gz'
   python scripts/fix_pylsd.py
   ```

