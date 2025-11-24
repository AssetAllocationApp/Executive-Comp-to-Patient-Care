{
  pkgs ? import <nixpkgs> { },
}:

with pkgs;

let
  py = python312;
  pypkgs = py.withPackages (
    pythonPkgs: with pythonPkgs; [
      ipython
      pip
    ]
  );
  # https://gist.github.com/cdepillabout/f7dbe65b73e1b5e70b7baa473dafddb3
  lib-path = with pkgs; lib.makeLibraryPath [
    stdenv.cc.cc

    # numpy needs libz.so.1
    zlib
  ];
in

pkgs.mkShell {
  name = "Jupyter env";

  buildInputs = [
    pypkgs
  ];
  shellHook = ''
    export "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${lib-path}"
    python -m venv .env
    source ./.env/bin/activate
    pip install -r requirements.txt
  '';
}
