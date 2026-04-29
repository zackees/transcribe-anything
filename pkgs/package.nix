{ lib
, pkgs
, pythonSet
, workspace
, projectSrc ? ../.
}:

let
  pyproject = lib.importTOML (projectSrc + "/pyproject.toml");
  version = pyproject.project.version or "0.0.0";

  virtualenv = pythonSet.mkVirtualEnv "transcribe-anything-env" workspace.deps.default;
in
pkgs.symlinkJoin {
  pname = "transcribe-anything";
  name = "transcribe-anything-${version}";
  paths = [ virtualenv ];

  nativeBuildInputs = [
    pkgs.makeWrapper
  ];

  postBuild = ''
    for script in transcribe_anything transcribe-anything transcribe-anything-init-insane; do
      if [ -x "$out/bin/$script" ]; then
        wrapProgram "$out/bin/$script" --prefix PATH : ${lib.makeBinPath [ pkgs.ffmpeg pkgs.uv ]}
      fi
    done
  '';
}
