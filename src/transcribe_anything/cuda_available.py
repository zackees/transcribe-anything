# pylint: skip-file

"""
Queries the system for CUDA devices and returns a json string with the information.
This is meant to be run under a "isolated-environment".
"""

import argparse
import json
import shutil
import sys
import traceback
import warnings
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass
class CudaDevice:
    """Data class to hold CUDA device information."""

    name: str
    vram: int  # VRAM in bytes
    multiprocessors: int  # Number of multiprocessors
    device_id: int

    def __str__(self):
        return f"{self.name} - VRAM: {self.vram / (1024 ** 3):.2f} GB, " f"Multiprocessors: {self.multiprocessors}"

    def to_json(self) -> dict[str, str | int]:
        """Returns a dictionary representation of the object."""
        return asdict(self)

    @staticmethod
    def from_json(json_data: dict[str, str | int]) -> "CudaDevice":
        """Returns a CudaDevice object from a dictionary."""
        return CudaDevice(**json_data)  # type: ignore


@dataclass
class CudaInfo:
    """Cuda info."""

    cuda_available: bool
    num_cuda_devices: int
    cuda_devices: list[CudaDevice]

    def to_json_str(self) -> str:
        """Returns json str."""
        # Convert dataclass to dictionary for serialization
        data = self.to_json()
        return json.dumps(data, indent=4, sort_keys=True)

    def to_json(self) -> dict[str, Any]:
        """Returns a dictionary representation of the object."""
        out = {}
        for field in fields(self):
            out[field.name] = getattr(self, field.name)
            if field.name == "cuda_devices":
                out[field.name] = [device.to_json() for device in out[field.name]]
        return out

    @staticmethod
    def from_json_str(json_str: str) -> "CudaInfo":
        """Loads from json str and returns a CudaInfo object."""
        assert json_str is not None, "Expected json_str to be set, but was instead None"
        data = json.loads(json_str)
        cuda_devices_data = data.get("cuda_devices", [])
        cuda_devices = [CudaDevice(**device) for device in cuda_devices_data]
        return CudaInfo(data["cuda_available"], data["num_cuda_devices"], cuda_devices)

    def __repr__(self) -> str:
        """Write out as json."""
        return self.to_json_str()


def cuda_cards_available() -> CudaInfo:
    """
    Returns a CudaInfo object with information about the CUDA cards,
    ordered by VRAM and multiprocessors.
    """
    # Have to import here, since others will import CudaDevice and CudaInfo.
    if shutil.which("nvidia-smi") is None:
        return CudaInfo(False, 0, [])

    try:
        import torch  # type: ignore
    except ImportError as e:
        stacktrace: str = traceback.format_exc()
        warnings.warn(f"\n{stacktrace}\n\nERROR importing TORCH!: {e}, disabling cuda")
        return CudaInfo(False, 0, [])

    if torch.cuda.is_available():
        devices: list[CudaDevice] = []
        try:
            count = torch.cuda.device_count()
        except Exception as e:
            # print(f"Error getting device count: {e}")
            sys.stderr.write(f"Error getting device count: {e}\n")
            return CudaInfo(False, 0, [])
        for i in range(count):
            try:
                device = torch.cuda.get_device_properties(i)
                devices.append(
                    CudaDevice(
                        name=torch.cuda.get_device_name(i),
                        vram=device.total_memory,
                        multiprocessors=device.multi_processor_count,
                        device_id=i,
                    )
                )
            except Exception as e:
                # print(f"Error getting device {i}: {e}")
                sys.stderr.write(f"Error getting device {i}: {e}\n")
        # Sort devices by VRAM and then by number of multiprocessors in descending order
        devices.sort(key=lambda x: (x.vram, x.multiprocessors), reverse=True)
        return CudaInfo(True, len(devices), devices)
    return CudaInfo(False, 0, [])


def parse_args() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="Check if CUDA is available.")
    # positional argument is the output file
    parser.add_argument(
        "-o",
        "--output",
        help="The output file",
        type=str,
    )
    return parser.parse_args()


def main() -> int:
    """Returns 0 if cuda is available, 1 otherwise."""
    args = parse_args()
    cuda_info = cuda_cards_available()
    json_str = cuda_info.to_json_str()
    assert json_str is not None, "Expected json_str to be set, but was instead None"
    # print(json_str)
    # args.output.write(json_str)
    if args.output:
        with open(args.output, encoding="utf-8", mode="w") as fd:
            fd.write(json_str)
    else:
        print(json_str)
    if cuda_info.cuda_available:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
