{ pkgs ? import <nixpkgs> {} }:
(pkgs.buildFHSUserEnv {
  name = "cody_thesis";
  targetPkgs = pkgs: (with pkgs; [
    python312
    python312Packages.pip
    python312Packages.virtualenv
    zlib
  ]);
  runScript = ''
    #!/usr/bin/env bash
    virtualenv venv
    source venv/bin/activate
    pip install -r requirements.txt
    bash
  '';
}).env
