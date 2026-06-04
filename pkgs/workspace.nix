{ lib
, pkgs
, pyproject-build-systems
, pyproject-nix
, uv2nix
}:

let
  workspace = uv2nix.lib.workspace.loadWorkspace {
    workspaceRoot = ../.;
  };

  overlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel";
  };

  editableOverlay = workspace.mkEditablePyprojectOverlay {
    root = "$REPO_ROOT";
  };

  python = pkgs.python314;

  pythonBase = pkgs.callPackage pyproject-nix.build.packages {
    inherit python;
  };

  pythonSet = pythonBase.overrideScope (
    lib.composeManyExtensions [
      pyproject-build-systems.overlays.wheel
      overlay
      (final: prev: {
        docopt = prev.docopt.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
            (final.resolveBuildSystem {
              setuptools = [ ];
              wheel = [ ];
            })
          ];
        });
      })
    ]
  );

  editablePythonSet = pythonSet.overrideScope editableOverlay;
in
{
  inherit
    editableOverlay
    editablePythonSet
    overlay
    python
    pythonSet
    workspace;
}
