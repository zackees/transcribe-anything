# pylint: skip-file

"""
Queries the system for CUDA devices and returns a json string with the information.
This is meant to be run under a "isolated-environment".
"""

import json
import shutil
import sys
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
        return (
            f"{self.name} - VRAM: {self.vram / (1024 ** 3):.2f} GB, "
            f"Multiprocessors: {self.multiprocessors}"
        )

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
        data = json.loads(json_str)
        cuda_devices_data = data.get("cuda_devices", [])
        cuda_devices = [CudaDevice(**device) for device in cuda_devices_data]
        return CudaInfo(data["cuda_available"], data["num_cuda_devices"], cuda_devices)


def cuda_cards_available() -> CudaInfo:
    """
    Returns a CudaInfo object with information about the CUDA cards,
    ordered by VRAM and multiprocessors.
    """
    # Have to import here, since others will import CudaDevice and CudaInfo.
    if shutil.which("nvidia-smi") is None:
        return CudaInfo(False, 0, [])
    import torch  # type: ignore

    if torch.cuda.is_available():
        devices = [
            CudaDevice(
                name=torch.cuda.get_device_name(i),
                vram=torch.cuda.get_device_properties(i).total_memory,
                multiprocessors=torch.cuda.get_device_properties(
                    i
                ).multi_processor_count,
                device_id=i,
            )
            for i in range(torch.cuda.device_count())
        ]
        # Sort devices by VRAM and then by number of multiprocessors in descending order
        devices.sort(key=lambda x: (x.vram, x.multiprocessors), reverse=True)
        return CudaInfo(True, len(devices), devices)
    return CudaInfo(False, 0, [])


def main() -> int:
    """Returns 0 if cuda is available, 1 otherwise."""
    cuda_info = cuda_cards_available()
    print(cuda_info.to_json_str())
    if cuda_info.cuda_available:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
