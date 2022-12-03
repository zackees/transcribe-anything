import sys
import os

from shutil import rmtree
from setuptools import find_packages, setup, Command

# The directory containing this file
HERE = os.path.dirname(__file__)

NAME = "transcribe-anything"
DESCRIPTION = (
    "Uses whisper AI to transcribe speach from video and audio files. "
    "Also accepts urls for youtube, rumble, bitchute, clear file, etc."
)
URL = "https://github.com/zackees/transcribe-anything"
EMAIL = "dont@email.me"
AUTHOR = "Zach Vorhies"
REQUIRES_PYTHON = ">=3.10.0"
VERSION = "2.0.5"

# The text of the README file
with open(os.path.join(HERE, "README.md")) as fd:
    README = fd.read()


def parse_requirements(filename):
    """load requirements from a pip requirements file"""
    with open(filename, encoding="utf-8", mode="rt") as fd:
        lines = [line.strip() for line in fd.readlines() if line.strip()]
    lines = [line.split("#")[0].strip() for line in lines]
    lines = [line for line in lines if line]
    return lines


REQUIREMENTS = parse_requirements(os.path.join(HERE, "requirements.txt"))


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        pass

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(HERE, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(VERSION))
        os.system("git push --tags")

        sys.exit()


setup(
    name=NAME,
    python_requires=REQUIRES_PYTHON,
    version=VERSION,
    description=DESCRIPTION,
    long_description=README,
    long_description_content_type="text/markdown",
    url=URL,
    author="Zach Vorhies",
    author_email="dont@email.me",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: MacOS :: MacOS X",
        "Environment :: Console",
    ],
    install_requires=REQUIREMENTS,
    entry_points={
        "console_scripts": [
            "transcribe_anything = transcribe_anything.cmd:main",
        ],
    },
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    package_data={},
    include_package_data=True,
    cmdclass={
        "upload": UploadCommand,
    },
)
