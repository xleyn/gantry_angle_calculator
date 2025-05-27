import sys
import json
from pathlib import Path
import warnings

import time
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


class IOManager:
    """Class for managing I/O operations."""

    if getattr(sys, "frozen", False):
        project_dir = Path(sys.executable).parent
    else:
        project_dir = Path(__file__).parent.parent

    with open(project_dir.joinpath("config/file_structure.json")) as f:
        config = json.load(f)
        paths_from_proj_dir = dict(
            zip(
                config.keys(),
                map(
                    lambda v, project_dir=project_dir: project_dir / v, config.values()
                ),
            )
        )

    log_template = pd.DataFrame(
        columns=[
            "File Name",
            "Processing Timestamp",
            "Top Left",
            "Top Right",
            "Bottom Left",
            "Bottom Right",
        ]
    )

    @classmethod
    def creation_control(cls):
        """Checks if I/O paths exist. Exits programme if not, rectifying the issues."""
        creation_not_required = True
        for path in cls.paths_from_proj_dir.values():
            if not path.exists():
                creation_not_required = False
                if path.suffix == ".xlsx":
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch(exist_ok=True)
                    cls.log_template.to_excel(path, index=False)
                    print(f"Excel file created: {path.absolute()}")
                else:
                    path.mkdir(parents=True, exist_ok=True)
                    print(f"Directory created: {path.absolute()}")

        if creation_not_required:
            print("All I/O paths exist, as expected!")

        else:
            print(
                "Some I/O paths did not exist!\n"
                "Issue rectified. Please re-run the software."
            )
            time.sleep(10)
            sys.exit()

    @classmethod
    def pull_images(cls) -> list[Path]:
        """Pulls images from the image input directory that need to be analysed.

        Returns:
            list[Path]: List of images that need to be analysed.
        """
        file_types = [".jpg", ".jpeg"]
        already_analysed = cls.pull_already_analysed_images()
        images_to_analyse = [
            img
            for f in file_types
            for img in cls.paths_from_proj_dir["image_input_dir"].rglob(f"*{f}")
            if img.name not in already_analysed
        ]
        if images_to_analyse:
            return images_to_analyse
        else:
            print(
                f"Error: all images in image input folder have already been analysed!"
            )
            time.sleep(5)
            sys.exit()

    @classmethod
    def pull_already_analysed_images(cls) -> list[str]:
        """Pulls image names from excel log that have already been analysed.

        Returns:
            list[str]: List of image names from excel log that have already been analysed.
        """
        df = pd.read_excel(cls.paths_from_proj_dir["excel_log"])
        analysed_images = df[df.columns[0]].to_list()
        return analysed_images
