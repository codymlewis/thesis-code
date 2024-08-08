{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
  inputs.utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, utils }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python-env = pkgs.python312.withPackages (pp: with pp; [
          numpy
          pandas
        ]);
      in
      {
        devShell = with pkgs; mkShell {
          packages = with pkgs; [
            python312
            python312Packages.pip
            python312Packages.virtualenv
            pciutils
            zlib
            gcc-unwrapped
          ];
          buildInputs = [ python-env ];
          shellHook = ''
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
          '';
        };
      }
    );
}
