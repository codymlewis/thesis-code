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
    virtualenv venv
    source venv/bin/activate
    pip install -r requirements.txt
    if [[ $(lspci | grep 'NVIDIA') ]]; then
      pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    else
      pip install jax
    fi
    bash
  '';
}).env
