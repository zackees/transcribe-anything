let
  flake = builtins.getFlake (toString ./.);
  overlay = import ./overlay.nix { inherit (flake) inputs; };
  pkgs = import flake.inputs.nixpkgs {
    system = builtins.currentSystem;
    overlays = [ overlay ];
  };
in
pkgs.transcribe-anything-shell
