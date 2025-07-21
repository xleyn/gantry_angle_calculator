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
