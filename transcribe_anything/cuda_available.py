# pylint: disable

"""
Returns 0 if cuda is available, 1 otherwise.. This is 
designed to be run in an isolated environment.
"""

import sys
import torch  # type: ignore  # pylint: disable=import-error

def main() -> int:
    """Returns 0 if cuda is available, 1 otherwise."""
    if torch.cuda.is_available():
        print("CUDA is available")
        return 0
    print("CUDA is not available")
    return 1

if __name__ == '__main__':
    sys.exit(main())
