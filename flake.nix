{
  description = "piv-from-scratch: a PIV processor and synthetic particle-image generator in plain NumPy/SciPy";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312.withPackages (ps: with ps; [
          numpy
          scipy
          matplotlib
          pillow
          pytest
        ]);
        tools = [ python pkgs.ruff ];
      in
      {
        devShells.default = pkgs.mkShell {
          packages = tools;
          shellHook = ''
            export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
            export MPLBACKEND=Agg
          '';
        };

        checks.default = pkgs.runCommand "pivfs-checks" { nativeBuildInputs = tools; } ''
          cp -r ${self} src-root
          chmod -R u+w src-root
          cd src-root
          export HOME=$TMPDIR MPLBACKEND=Agg PYTHONPATH=$PWD/src
          ruff check .
          ruff format --check .
          pytest -q -p no:cacheprovider tests
          touch $out
        '';

        apps.experiments = {
          type = "app";
          program = toString (pkgs.writeShellScript "pivfs-experiments" ''
            export MPLBACKEND=Agg
            export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
            exec ${python}/bin/python scripts/run_all.py "$@"
          '');
        };
      });
}
