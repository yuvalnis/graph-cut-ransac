#!/usr/bin/env python3
"""
Post-installation script to fix pylsd Python 2 to Python 3 compatibility issues.
This script patches the installed pylsd package to work with Python 3.

Fixes applied:
1. Convert Python 2 absolute imports to Python 3 relative imports
2. Fix Python 2 exception syntax to Python 3 syntax
3. Fix tempfile handling to use system temp directory (prevents file write errors)
   Solution for #3 based on community contribution: https://github.com/primetang/pylsd/issues/14
"""

import os
import sys
import site


def find_pylsd_path():
    """Find the installation path of pylsd package."""
    for site_packages in site.getsitepackages() + [site.getusersitepackages()]:
        pylsd_path = os.path.join(site_packages, 'pylsd')
        if os.path.exists(pylsd_path):
            return pylsd_path
    
    # Also check in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        venv_site_packages = os.path.join(sys.prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages', 'pylsd')
        if os.path.exists(venv_site_packages):
            return venv_site_packages
    
    return None


def fix_relative_imports(file_path, old_import, new_import):
    """Fix relative import in a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        if old_import in content:
            content = content.replace(old_import, new_import)
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"✓ Fixed import in {file_path}")
            return True
        else:
            print(f"⊘ Import already fixed or not found in {file_path}")
            return False
    except Exception as e:
        print(f"✗ Error fixing {file_path}: {e}")
        return False


def fix_exception_syntax(file_path):
    """Fix Python 2 exception syntax to Python 3."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        if 'except Exception, e:' in content:
            content = content.replace('except Exception, e:', 'except Exception as e:')
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"✓ Fixed exception syntax in {file_path}")
            return True
        else:
            print(f"⊘ Exception syntax already fixed or not found in {file_path}")
            return False
    except Exception as e:
        print(f"✗ Error fixing {file_path}: {e}")
        return False


def fix_lsd_tempfile(file_path):
    """Fix lsd.py to use proper tempfile handling for cross-platform compatibility."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if fix is already applied
        if 'tempfile.NamedTemporaryFile' in content:
            print(f"⊘ Tempfile fix already applied in {file_path}")
            return False
        
        # The improved lsd() function with proper tempfile handling
        new_lsd_function = '''import tempfile


def lsd(src):
    rows, cols = src.shape
    src = src.reshape(1, rows * cols).tolist()[0]

    # Create a temporary file in the system's temp directory
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
        temp = tmp.name

    lens = len(src)
    src = (ctypes.c_double * lens)(*src)
    lsdlib.lsdGet(src, ctypes.c_int(rows), ctypes.c_int(cols), temp.encode('utf-8'))

    fp = open(temp, 'r')
    cnt = fp.read().strip().split(' ')
    fp.close()
    os.remove(temp)

    count = int(cnt[0])
    dim = int(cnt[1])
    lines = np.array([float(each) for each in cnt[2:]])
    lines = lines.reshape(count, dim)

    return lines
'''
        
        # Find and replace the lsd function
        import re
        pattern = r'def lsd\(src\):.*?return lines\n'
        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, new_lsd_function, content, flags=re.DOTALL)
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"✓ Fixed tempfile handling in {file_path}")
            return True
        else:
            print(f"⊘ Could not find lsd function to replace in {file_path}")
            return False
    except Exception as e:
        print(f"✗ Error fixing {file_path}: {e}")
        return False


def main():
    """Main function to fix pylsd installation."""
    print("Fixing pylsd for Python 3 compatibility...")
    print("-" * 60)
    
    pylsd_path = find_pylsd_path()
    
    if not pylsd_path:
        print("✗ pylsd package not found. Please install it first:")
        print("  pip install 'pylsd @ https://files.pythonhosted.org/packages/8d/ac/128158438742944b170ebf81ec6d913df382387f782f87b0ca4bc51b291c/pylsd-0.0.2.tar.gz'")
        sys.exit(1)
    
    print(f"Found pylsd at: {pylsd_path}")
    print()
    
    # Fix __init__.py
    init_file = os.path.join(pylsd_path, '__init__.py')
    fix_relative_imports(init_file, 'from lsd import lsd', 'from .lsd import lsd')
    
    # Fix lsd.py - relative imports
    lsd_file = os.path.join(pylsd_path, 'lsd.py')
    fix_relative_imports(lsd_file, 'from bindings.lsd_ctypes import *', 'from .bindings.lsd_ctypes import *')
    
    # Fix lsd.py - tempfile handling (this must come after the import fix)
    fix_lsd_tempfile(lsd_file)
    
    # Fix bindings/__init__.py
    bindings_init = os.path.join(pylsd_path, 'bindings', '__init__.py')
    fix_relative_imports(bindings_init, 'from lsd_ctypes import *', 'from .lsd_ctypes import *')
    
    # Fix bindings/lsd_ctypes.py
    lsd_ctypes = os.path.join(pylsd_path, 'bindings', 'lsd_ctypes.py')
    fix_exception_syntax(lsd_ctypes)
    
    print()
    print("-" * 60)
    print("Testing pylsd import...")
    
    try:
        import pylsd
        from pylsd import lsd
        print("✓ pylsd imported successfully!")
        print()
        print("pylsd is now ready to use.")
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        print("Please check the errors above and try again.")
        sys.exit(1)


if __name__ == '__main__':
    main()

