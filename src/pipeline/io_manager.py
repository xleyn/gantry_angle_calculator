"""
io_manager.py

Script to define IOManager class for managing I/O operations within pipeline.
Handles detection and creation of I/O paths, error handling and interacting with gantry angle log.

Written by Nathan Crossley, 2025.
"""

import sys
import json
from pathlib import Path
import warnings

import time
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


################### CARRY ON COMMENTING FROM HERE AND ABOVE FILES
# MAKE SURE REPOS ARE SYNCED
# COPY REPOS TO R DRIVE
# ENSURE THAT ALL REPOS HAVE GOT TEST DATA
# CLONE LANDAUER REPO
# GO THROUGH R DRIVE AND GET OTHER USEFUL PROJECTS ON LOCAL PC


class IOManager:
    """Class for managing I/O operations within gantry angle pipeline.
    Handles detection and creation of I/O paths, error handling and interacting
    with gantry angle log."""

    # get the root project directory, testing whether running as script or frozen exe
    if getattr(sys, "frozen", False):
        project_dir = Path(sys.executable).parent
    else:
        project_dir = Path(__file__).parents[2]

    # load file structure dict from config file
    with open(project_dir.joinpath("config/file_structure.json")) as f:
        config = json.load(f)

        # define absolute paths needed throughout pipeline by combining root dir and absolute paths from config file
        paths_from_proj_dir = dict(
            zip(
                config.keys(),
                map(
                    lambda v, project_dir=project_dir: project_dir / v, config.values()
                ),
            )
        )

    # construct basic template for gantry angle log
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
        """Checks if I/O paths exist and creates them if not,
        exiting pipelinei and asking user to return."""

        # bool storing whether I/O path creation has been required
        creation_not_required = True

        # iterate through each of the absolute paths required throughout pipeline
        for path in cls.paths_from_proj_dir.values():

            # if path doesn't exist, modify bool and create path
            if not path.exists():
                creation_not_required = False

                # if path is an excel file, create it with the log template
                if path.suffix == ".xlsx":
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch(exist_ok=True)
                    cls.log_template.to_excel(path, index=False)
                    print(f"Excel file created: {path.absolute()}")

                # if path is a directory, create it
                else:
                    path.mkdir(parents=True, exist_ok=True)
                    print(f"Directory created: {path.absolute()}")

        # if no paths needed to be created, inform user
        if creation_not_required:
            print("All I/O paths exist, as expected!")

        # if paths were created, inform user and exit
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
        Checks against the excel log to exclude those that have already been analysed.

        Returns:
            list[Path]: List of images that need to be analysed.
        """

        # define valid file types
        file_types = [".jpg", ".jpeg"]

        # get all images that have already been analysed
        already_analysed = cls.pull_already_analysed_images()

        # get all images from input dir and check against list of already analysed images
        images_to_analyse = [
            img
            for f in file_types
            for img in cls.paths_from_proj_dir["image_input_dir"].rglob(f"*{f}")
            if img.name not in already_analysed
        ]

        # if there are any images to analyse, return them
        if images_to_analyse:
            return images_to_analyse

        # else inform the user and exit
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

        # read excel log and get first column as list of analysed images
        df = pd.read_excel(cls.paths_from_proj_dir["excel_log"])
        analysed_images = df[df.columns[0]].to_list()

        return analysed_images

    @classmethod
    def update_log(cls, image: "GantryAngleImage"):
        """Updates excel log with results from analysis of input image.

        Args:
            image (Image): Image to update log with results from.
        """

        # get current state of excel log
        df = pd.read_excel(cls.paths_from_proj_dir["excel_log"])

        # get new row extracting key results values from image object
        new_row = [image.path.name, pd.Timestamp.now()] + [
            image.paired_lines[corner]["intersection_angle"]
            for corner in ["top_left", "top_right", "bottom_left", "bottom_right"]
        ]

        # construct pd.DataFrame for new row
        new_row_df = pd.DataFrame([dict(zip(df.columns, new_row))])

        # concatenate new row to df if df is not empty
        if not (df.empty or df.isna().all().all()):
            df = pd.concat(
                [df.dropna(axis=0, how="all"), new_row_df], ignore_index=True, axis=0
            )

        # otherwise just use new row
        else:
            df = new_row_df

        # instantiate excel writer and write df to excel file
        with pd.ExcelWriter(
            cls.paths_from_proj_dir["excel_log"], engine="xlsxwriter"
        ) as writer:

            # write df to excel file
            df.to_excel(writer, index=False)

            # modify column widths and freeze panes for formatting
            workbook = writer.book
            worksheet = writer.sheets.values().__iter__().__next__()
            worksheet.set_column(0, 0, 30)
            worksheet.set_column(1, 1, 20)
            worksheet.set_column(2, 16, 15)
            worksheet.freeze_panes(1, 0)
