{ inputs }:
final: prev:
let
  repo = final.callPackage ./pkgs/workspace.nix {
    pkgs = final;
    inherit (inputs) pyproject-build-systems pyproject-nix uv2nix;
  };

  static-ffmpeg = final.callPackage ./pkgs/static-ffmpeg.nix {
    inherit (repo) pythonSet;
  };
  disklru = final.callPackage ./pkgs/disklru.nix {
    inherit (repo) pythonSet;
  };
  webvtt-py = final.callPackage ./pkgs/webvtt-py.nix {
    inherit (repo) pythonSet;
  };
  uv-iso-env = final.callPackage ./pkgs/uv-iso-env.nix {
    inherit (repo) pythonSet;
  };

  transcribe-anything = final.callPackage ./pkgs/package.nix {
    pkgs = final;
    lib = final.lib;
    inherit (repo) pythonSet workspace;
  };
  transcribe-anything-shell = final.callPackage ./pkgs/devshell.nix {
    pkgs = final;
    inherit (repo) editablePythonSet workspace;
  };
in
{
  inherit transcribe-anything transcribe-anything-shell;

  default = transcribe-anything;
}
