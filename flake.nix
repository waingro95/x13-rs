{
  description = "Rust development shell with X-13ARIMA-SEATS Build 62";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rustfmt" "clippy" "rust-analyzer" ];
        };
        rustPlatform = pkgs.makeRustPlatform {
          cargo = rustToolchain;
          rustc = rustToolchain;
        };
        x13as = pkgs.stdenv.mkDerivation {
          pname = "x13as-ascii";
          version = "1.1-build-62";

          src = pkgs.fetchurl {
            url = "https://www2.census.gov/software/x-13arima-seats/x13as/unix-linux/program-archives/x13as_asciisrc-v1-1-b62.tar.gz";
            hash = "sha256-gnlshLVIkUdN8ab/sWvnv3vUqZLjZ5FYZFyIjmzd2hI=";
          };

          nativeBuildInputs = [ pkgs.gfortran pkgs.gnumake ];

          postPatch = ''
            substituteInPlace makefile.gf \
              --replace-fail '$(LINKER) -static -o $@' '$(LINKER) -o $@'
          '';

          buildPhase = ''
            runHook preBuild
            make -f makefile.gf FFLAGS="-O2 -std=legacy -fallow-argument-mismatch"
            runHook postBuild
          '';

          installPhase = ''
            runHook preInstall
            install -Dm755 x13as_ascii "$out/bin/x13as_ascii"
            runHook postInstall
          '';

          meta = {
            description = "U.S. Census Bureau X-13ARIMA-SEATS ASCII, version 1.1 build 62";
            homepage = "https://www.census.gov/data/software/x13as.X-13ARIMA-SEATS.html";
            platforms = pkgs.lib.platforms.unix;
            mainProgram = "x13as_ascii";
          };
        };
      in
      {
        packages = {
          inherit x13as;
          default = x13as;
        };

        checks = {
          rust-default = rustPlatform.buildRustPackage {
            pname = "x13-rust-default";
            version = "0.1.0";
            src = self;
            cargoLock.lockFile = ./Cargo.lock;

            buildType = "debug";
            checkType = "debug";
            auditable = false;
            # Keep ordinary Cargo debug semantics instead of the helper's
            # Darwin-only split-debuginfo override.
            env.RUSTFLAGS = "";

            buildPhase = ''
              runHook preBuild
              cargo clippy -p x13 --all-targets --locked --offline --jobs 1 --verbose -- -D warnings
              runHook postBuild
            '';

            checkPhase = ''
              runHook preCheck
              cargo test -p x13 --all-targets --locked --offline --jobs 1 --verbose
              runHook postCheck
            '';

            installPhase = ''
              runHook preInstall
              mkdir -p "$out"
              touch "$out/passed"
              runHook postInstall
            '';
          };

          x13as-smoke = pkgs.runCommand "x13as-smoke" {
            nativeBuildInputs = [ x13as ];
          } ''
            mkdir work
            cp ${./tests/x13as-smoke.spc} work/smoke.spc
            cd work
            x13as_ascii -i smoke -o output -n -q > console.txt
            grep -q "Version Number 1.1 Build 62" console.txt
            test -s output.d11
            mkdir -p "$out"
            cp output.d11 "$out/"
          '';

          x13as-seats-smoke = pkgs.runCommand "x13as-seats-smoke" {
            nativeBuildInputs = [ x13as ];
          } ''
            mkdir work
            cp ${./tests/x13as-semiannual-smoke.spc} work/smoke.spc
            cd work
            x13as_ascii -i smoke -o output -n -q > console.txt
            grep -q "Version Number 1.1 Build 62" console.txt
            test -s output.s11
            mkdir -p "$out"
            cp output.s11 "$out/"
          '';
        };

        devShells.default = pkgs.mkShell {
          packages = [
            rustToolchain
            x13as
          ];
        };
      }
    );
}
