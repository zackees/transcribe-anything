{ pkgs
, editablePythonSet
, workspace
}:

let
  virtualenv = editablePythonSet.mkVirtualEnv "transcribe-anything-dev-env" workspace.deps.all;
in
pkgs.mkShell {
  packages = [
    virtualenv
    pkgs.ffmpeg
    pkgs.uv
    pkgs.yt-dlp
  ];

  env = {
    UV_NO_SYNC = "1";
    UV_PYTHON = editablePythonSet.python.interpreter;
    UV_PYTHON_DOWNLOADS = "never";
  };

  shellHook = ''
    unset PYTHONPATH
    export REPO_ROOT=$(git rev-parse --show-toplevel)
    ln -sfn ${virtualenv}/bin/activate "$REPO_ROOT/activate"
  '';
}
