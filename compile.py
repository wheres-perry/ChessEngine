"""
Build script for the chess engine C++ extensions.
Handles the complete build process for development and testing.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


class ChessEngineBuild:
    """Handles building the chess engine C++ extensions."""

    def __init__(self) -> None:
        self.project_root = Path(__file__).parent

    def clean(self) -> None:
        """Clean previous build artifacts."""
        print("🧹 Cleaning previous build artifacts...")

        # Remove compiled extensions
        for ext_file in self.project_root.rglob("*.so"):
            ext_file.unlink()
            print(f"   Removed {ext_file}")

        for ext_file in self.project_root.rglob("*.pyd"):
            ext_file.unlink()
            print(f"   Removed {ext_file}")

        # Remove build directories
        for build_dir in self.project_root.rglob("build"):
            if build_dir.is_dir():
                shutil.rmtree(build_dir)
                print(f"   Removed {build_dir}")

        print("✅ Clean complete")

    def install_python_package(self) -> None:
        """Install the Python package with C++ extensions"""
        print("🐍 Installing Python package with C++ extensions...")
        print("   This command will compile C++ extensions via setup.py...")

        try:
            # Reinstall the package to ensure C++ extensions are compiled
            subprocess.run(
                [
                    "sudo",
                    "pip",
                    "install",
                    "-e",
                    ".",
                    "--upgrade",
                    "--no-deps",
                    "--force-reinstall",
                ],
                check=True,
                cwd=self.project_root,
            )
            print("✅ Python package installed (C++ extensions compiled)")
        except subprocess.CalledProcessError as e:
            print(f"❌ Python package installation failed: {e}")
            sys.exit(1)

    def verify_installation(self) -> None:
        """Verify the C++ extensions can be imported."""
        print("🔍 Verifying installation...")

        try:
            result = subprocess.run(
                [  # noqa: S607,S603
                    "python",
                    "-c",
                    """
try:
    from engine._core import chess_engine_core as core
    board = core.Board()
    moves = board.generate_legal_moves()
    print(f'✅ C++ extension working! Found {len(moves)} moves in starting position')
except Exception as e:
    print(f'❌ Verification failed: {e}')
    import sys
    sys.exit(1)
                """,
                ],
                check=True,
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            print(result.stdout.strip())
            print("✅ Verification complete")
        except subprocess.CalledProcessError as e:
            print(f"❌ Verification failed: {e}")
            if e.stdout:
                print(f"stdout: {e.stdout}")
            if e.stderr:
                print(f"stderr: {e.stderr}")
            sys.exit(1)

    def run_tests(self) -> None:
        """Run the test suite."""
        print("🧪 Running tests...")

        try:
            subprocess.run(
                [  # noqa: S607,S603
                    "pytest",
                    "tests/core/core_engine_test.py",
                    "-v",
                ],
                check=True,
                cwd=self.project_root,
            )
            print("✅ All tests passed!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Tests failed: {e}")
            sys.exit(1)

    def full_build(self, clean: bool = True, run_tests: bool = True) -> None:
        """Run the complete build process."""
        print("🚀 Starting full chess engine build...")
        print("=" * 50)

        if clean:
            self.clean()

        # Dependencies are installed via devcontainer postCreateCommand with uv
        self.install_python_package()  # This handles C++ compilation via setup.py
        self.verify_installation()

        if run_tests:
            self.run_tests()

        print("=" * 50)
        print("🎉 Build complete! Your chess engine is ready!")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build the chess engine")
    parser.add_argument(
        "--no-clean", action="store_true", help="Skip cleaning previous build artifacts"
    )
    parser.add_argument(
        "--no-tests", action="store_true", help="Skip running tests after build"
    )
    parser.add_argument(
        "--clean-only", action="store_true", help="Only clean build artifacts"
    )

    args = parser.parse_args()

    builder = ChessEngineBuild()

    if args.clean_only:
        builder.clean()
        return

    builder.full_build(clean=not args.no_clean, run_tests=not args.no_tests)


if __name__ == "__main__":
    main()
