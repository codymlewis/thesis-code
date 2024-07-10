{ pkgs ? import <nixpkgs> {} }:
(pkgs.buildFHSUserEnv {
  name = "cody_thesis";
  targetPkgs = pkgs: (with pkgs; [
    python312
    python312Packages.pip
    python312Packages.virtualenv
    pciutils
    zlib
  ]);
  runScript = ''
    #!/usr/bin/env bash
    if [ ! -d venv/ ]; then
      virtualenv venv
      source venv/bin/activate
      pip install -r requirements.txt
    else
      source venv/bin/activate
    fi
    if [ ! -d smahfl/data/ ]; then
      cd smahfl/data_processing/
      python acquire.py
      cd ../../
    fi
    bash
  '';
}).env
