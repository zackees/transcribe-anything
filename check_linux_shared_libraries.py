import ctypes
import sys

libraries = [
    "libcudnn.so.9",
    "libcusparseLt.so.0",
    "libcupti.so.12",
    "libcusparse.so.12",
    "libcufft.so.11",
    "libcurand.so.10",
    "libcublas.so.12",
    "libnccl.so.2",
]

errors: list[str] = []
for lib in libraries:
    try:
        ctypes.CDLL(lib)
        print(f"Successfully loaded {lib}")
    except OSError as e:
        print(f"Failed to load {lib}: {e}")
        errors.append(lib)

if errors:
    print(f"Missing CUDA shared libraries: {errors}")
    print("Ensure nvidia-container-toolkit is installed on the host")
    print("and run with: docker run --gpus all ...")
    sys.exit(1)
else:
    print("All libraries loaded successfully.")
    sys.exit(0)
