
import ctypes

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

# ctypes.CDLL('libcudnn.so.9')
errors: list[str] = []
for lib in libraries:
    try:
        ctypes.CDLL(lib)
        print(f"Successfully loaded {lib}")
    except OSError as e:
        print(f"Failed to load {lib}: {e}")
        errors.append(lib)
        # exit(1)

if errors:
    print(f"Errors: {errors}")
    while True:
        import time
        print("Container is suspended but running, please log in and check the errors.")
        time.sleep(20)
    exit(1)
else:
    print("All libraries loaded successfully.")
    exit(0)
