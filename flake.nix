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
        # Headless Chromium + Playwright for the WebGPU parity test (Phase 2).
        webPython = pkgs.python312.withPackages (ps: with ps; [ playwright ]);
        chromiumOnly = pkgs.playwright-driver.browsers.override {
          withFirefox = false;
          withWebkit = false;
        };
        webEnv = ''
          export PLAYWRIGHT_BROWSERS_PATH=${chromiumOnly}
          export PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS=true
        '';
      in
      {
        devShells.default = pkgs.mkShell {
          packages = tools;
          shellHook = ''
            export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
            export MPLBACKEND=Agg
          '';
        };

        devShells.web = pkgs.mkShell {
          packages = [ webPython ];
          shellHook = webEnv;
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

        apps.parity = {
          type = "app";
          program = toString (pkgs.writeShellScript "pivfs-webgpu-parity" ''
            ${webEnv}
            exec ${webPython}/bin/python scripts/webgpu_parity.py "$@"
          '');
        };

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
