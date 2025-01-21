


import os

from setuptools import setup

URL = "https://github.com/zackees/transcribe-anything"
KEYWORDS = "isolated environment, uv pip python build tools"
HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    setup(
        keywords=KEYWORDS,
        url=URL,
        package_data={"": ["assets/example.txt"]},
        include_package_data=True)


