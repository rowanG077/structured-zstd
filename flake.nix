{
  description = "Rust tools for x86-64 zstd optimization";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          # Stable Rust, matching rust-toolchain.toml; versions pinned by flake.lock.
          rustc
          cargo
          rustfmt
          clippy
          rust-analyzer
          cargo-nextest

          # Assembly / LLVM IR, code size, CPU and allocation profiling.
          cargo-show-asm
          cargo-llvm-lines
          cargo-bloat
          cargo-binutils
          llvmPackages.llvm
          llvmPackages.clang
          llvmPackages.lld
          binutils
          gcc
          cargo-flamegraph
          perf
          valgrind
          gdb
          hyperfine
          util-linux # taskset, lscpu

          # C build support for the upstream zstd FFI benches.
          pkg-config
          zstd
          python3
          jq
          git
          ripgrep
        ];

        RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";
        # Keep the workspace's release/bench profiles and portable CPU baseline.
        # Select the existing --profile flamegraph when debug symbols are needed.
      };
    };
}
