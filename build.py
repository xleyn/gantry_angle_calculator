"""
build.py

A build script for gantry angle calculator, which bundles the project into a distributable directory using pyinstaller.
This script pulls the pysintaller .spec file, which details the specifics of the bundling process.
When the script is run, the user is prompted to enter a version number for the build. The suggested version number convention
is MAJOR.MINOR.REVISION.(BUILD), as is standard. The bundled project will be available in dist/

Written by Nathan Crossley, 2025.
"""

import subprocess
import sys
from pathlib import Path

version = input("Please enter version number for new build: ").strip()

if not version:
    print("No version number entered, aborting build process.")
    sys.exit()

dist_dir = Path(f"dist/gantry_angle_calculator_v{version}")

subprocess.run(
    [
        "pyinstaller",
        "--noconfirm",
        "--clean",
        "--distpath",
        str(dist_dir.resolve()),
        "gantry_angle_calculator.spec",
    ]
)

move_out_of_internal = ("config",)

for name in move_out_of_internal:
    path = dist_dir / "_internal" / name
    path.rename(dist_dir / name)
