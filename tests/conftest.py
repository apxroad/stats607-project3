# Ensure tests can import from the project "src" package without installing.
import os, sys

# Compute the project root as the parent of the current file's directory.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Prepend the project root to sys.path so `import src...` works in tests
# without requiring a pip install or editable install.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
