"""
  Quick install
  cd <YOUR DIRECTORY>
  Download and install in one line:
    curl -X GET https://raw.githubusercontent.com/zackees/make_venv/main/make_venv.py | python

  To enter the environment run:
    source activate.sh

  Notes:
    This script is tested to work using python2 and python3 from a fresh install. The only side effect
    of running this script is that virtualenv will be globally installed if it isn't already.
"""

import os
import subprocess
import sys


# This activation script adds the ability to run it from any path and also
# aliasing pip3 and python3 to pip/python so that this works across devices.
_ACTIVATE_SH = """
function abs_path {
  (cd "$(dirname '$1')" &>/dev/null && printf "%s/%s" "$PWD" "${1##*/}")
}
. $( dirname $(abs_path ${BASH_SOURCE[0]}))/venv/bin/activate
export PATH=$( dirname $(abs_path ${BASH_SOURCE[0]}))/:$PATH
alias python3=python
alias pip3=pip
export IN_ACTIVATED_ENV="1"
"""

HERE = os.path.dirname(__file__)
os.chdir(os.path.abspath(HERE))


def _exe(cmd):
    print('Executing "%s"' % cmd)
    # os.system(cmd)
    subprocess.check_call(cmd, shell=True)


def is_tool(name):
    """Check whether `name` is on PATH."""
    from distutils.spawn import find_executable

    return find_executable(name) is not None


if not os.path.exists("venv"):
    if not is_tool("virtualenv"):
        _exe("pip install virtualenv")
    # Which one is better? virtualenv or venv? This may switch later.
    _exe("virtualenv -p python310 venv")
    # _exe('python3 -m venv venv')
    # Linux/MacOS uses bin and Windows uses Script, so create
    # a soft link in order to always refer to bin for all
    # platforms.
    if sys.platform == "win32":
        target = os.path.join(HERE, "venv", "Scripts")
        link = os.path.join(HERE, "venv", "bin")
        _exe('mklink /J "%s" "%s"' % (link, target))
    with open("activate.sh", "wt") as fd:
        fd.write(_ACTIVATE_SH)
else:
    print("%s already exists" % os.path.abspath("venv"))

print(
    'Now use ". activate.sh" (at the project root dir) to enter into the environment.'
)
