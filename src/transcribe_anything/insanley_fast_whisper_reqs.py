"""
Requirements for the insanely fast whisper.
"""

import sys
from pathlib import Path

from iso_env import IsoEnv, IsoEnvArgs, PyProjectToml  # type: ignore

from transcribe_anything.util import has_nvidia_smi

HERE = Path(__file__).parent

# Set the versions
TENSOR_VERSION = "2.6.0"
CUDA_VERSION = "cu126"
TENSOR_CUDA_VERSION = f"{TENSOR_VERSION}+{CUDA_VERSION}"
EXTRA_INDEX_URL = f"https://download.pytorch.org/whl/{CUDA_VERSION}"


def get_current_python_version() -> str:
    """Returns the current python version."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


# All deps for CUDA because it's the most finicky.
_WIN_COMPILED_260: str = """
accelerate==1.3.0
aiohappyeyeballs==2.4.4
aiohttp==3.11.11
aiosignal==1.3.2
alembic==1.14.1
ansicon==1.89.0
antlr4-python3-runtime==4.9.3
anyio==4.8.0
asteroid-filterbanks==0.4.0
async-generator==1.10
attrs==22.2.0
beautifulsoup4==4.11.1
blessed==1.19.1
brotli==1.1.0
certifi==2022.12.7
cffi==1.15.1
charset-normalizer==2.1.1
click==8.1.8
colorama==0.4.6
colorlog==6.9.0
contourpy==1.3.1
cycler==0.12.1
datasets==2.17.1
deepl==1.14.0
dill==0.3.8
docopt==0.6.2
einops==0.8.0
exceptiongroup==1.1.0
filelock==3.17.0
fonttools==4.55.8
free-proxy==1.1.0
frozenlist==1.5.0
fsspec==2023.10.0
greenlet==3.1.1
h11==0.14.0
httpcore==1.0.7
httpx==0.28.1
huggingface-hub==0.28.1
hyperpyyaml==1.2.2
idna==3.4
inquirer==3.1.1
insanely-fast-whisper==0.0.15
intel-openmp==2024.0.3
jinja2==3.1.5
jinxed==1.2.0
joblib==1.4.2
julius==0.2.7
kiwisolver==1.4.8
langdetect==1.0.9
lightning==2.5.0.post0
lightning-utilities==0.12.0
llvmlite==0.44.0
lxml==4.9.3
mako==1.3.8
markdown-it-py==3.0.0
markupsafe==3.0.2
matplotlib==3.10.0
mdurl==0.1.2
more-itertools==10.6.0
mpmath==1.3.0
multidict==6.1.0
multiprocess==0.70.16
networkx==3.4.2
numba==0.61.0
numpy==2.1.3
omegaconf==2.3.0
openai-whisper==20240930
optuna==4.2.0
outcome==1.2.0
packaging==24.2
pandas==2.2.3
pillow==11.1.0
primepy==1.3
propcache==0.2.1
protobuf==5.29.3
psutil==6.1.1
pyannote-audio==3.3.2
pyannote-core==5.0.0
pyannote-database==5.1.3
pyannote-metrics==3.2.1
pyannote-pipeline==3.0.1
pyarrow==19.0.0
pyarrow-hotfix==0.6
pycparser==2.21
pydeeplx==1.0.4
pygments==2.19.1
pyparsing==3.2.1
pysocks==1.7.1
python-dateutil==2.9.0.post0
python-editor==1.0.4
pytorch-lightning==2.5.0
pytorch-metric-learning==2.8.1
pytz==2025.1
pyuseragents==1.0.5
pyyaml==6.0.2
readchar==4.0.3
regex==2024.11.6
requests==2.28.1
rich==13.9.4
ruamel-yaml==0.18.10
ruamel-yaml-clib==0.2.12
safeio==1.2
safetensors==0.5.2
scikit-learn==1.6.1
scipy==1.15.1
selenium==4.7.2
semver==3.0.4
sentencepiece==0.2.0
setuptools==75.8.0
shellingham==1.5.4
six==1.16.0
sniffio==1.3.0
socksio==1.0.0
sortedcontainers==2.4.0
soundfile==0.13.1
soupsieve==2.3.2.post1
speechbrain==1.0.2
sqlalchemy==2.0.37
srt==3.5.2
srtranslator==0.3.5
sympy==1.13.1
tabulate==0.9.0
tensorboardx==2.6.2.2
threadpoolctl==3.5.0
tiktoken==0.8.0
tokenizers==0.21.0
torch==2.6.0+cu126
torch-audiomentations==0.12.0
torch-pitch-shift==1.2.5
torchaudio==2.6.0
torchmetrics==1.6.1
tqdm==4.64.1
transformers==4.48.2
translatepy==2.3
trio==0.22.0
trio-websocket==0.9.2
typer==0.15.1
typing-extensions==4.12.2
tzdata==2025.1
urllib3==1.26.13
wcwidth==0.2.5
webdriverdownloader==1.1.0.3
wsproto==1.2.0
xxhash==3.5.0
yarl==1.18.3
"""

_COMPILED: dict[str, str] = {
    "WIN_CUDA_260": _WIN_COMPILED_260,
}


def _get_reqs_generic(has_nvidia: bool) -> list[str]:
    """Generate the requirements for the generic case."""
    deps = [
        "transformers==4.46.3",  # 4.47.X has problems with mac mps driver see fix: https://github.com/huggingface/transformers/pull/35295
        "pyannote.audio==3.3.2",
        "openai-whisper==20240930",
        "insanely-fast-whisper==0.0.15",
        "torchaudio==2.6.0",
        "datasets==2.17.1",
        "pytorch-lightning==2.5.0",
        "torchmetrics==1.6.1",
        "srtranslator==0.3.5",
        # "numpy==2.2.0",
        "safeIO==1.2",
        "llvmlite==0.44.0",
        "numba==0.61.0",
    ]

    content_lines: list[str] = []

    for dep in deps:
        content_lines.append(dep)
    if has_nvidia:
        content_lines.append(f"torch=={TENSOR_CUDA_VERSION}")
    else:
        content_lines.append(f"torch=={TENSOR_VERSION}")
    if sys.platform != "darwin":
        # Add the windows specific dependencies.
        content_lines.append("intel-openmp==2024.0.3")

    return content_lines


def get_environment() -> IsoEnv:
    """Returns the environment."""
    venv_dir = HERE / "venv" / "insanely_fast_whisper"
    has_nvidia = has_nvidia_smi()
    is_windows = sys.platform == "win32"
    if False and has_nvidia and TENSOR_VERSION == "2.6.0" and is_windows:
        dep_lines = _COMPILED["WIN_CUDA_260"].splitlines()
    else:
        dep_lines = _get_reqs_generic(has_nvidia)
    # filter out empty lines
    dep_lines = [line.strip() for line in dep_lines if line.strip()]
    content_lines: list[str] = []

    content_lines.append("[build-system]")
    content_lines.append('requires = ["setuptools", "wheel"]')
    content_lines.append('build-backend = "setuptools.build_meta"')
    content_lines.append("")

    content_lines.append("[project]")
    content_lines.append('name = "project"')
    content_lines.append('version = "0.1.0"')
    content_lines.append('requires-python = "==3.11.*"')
    content_lines.append("dependencies = [")
    for dep in dep_lines:
        content_lines.append(f'  "{dep}",')
    content_lines.append("]")

    if has_nvidia:
        content_lines.append("[tool.uv.sources]")
        content_lines.append("torch = [")
        content_lines.append("  { index = 'pytorch-cu121' },")
        content_lines.append("]")
        content_lines.append("[[tool.uv.index]]")
        content_lines.append('name = "pytorch-cu121"')
        content_lines.append(f'url = "{EXTRA_INDEX_URL}"')
        content_lines.append("explicit = true")

    content = "\n".join(content_lines)
    build_info = PyProjectToml(content)
    args = IsoEnvArgs(venv_path=venv_dir, build_info=build_info)
    env = IsoEnv(args)
    return env
