"""
slice_thickness_task.py

Script to define pipeline class for managing gantry angle analysis process.
Ensures that I/O errors will not be raised and selects images to be analysed.
Each image is analysed and the log is updated.

Written by Nathan Crossley, 2025.
"""

import sys
import traceback
import time

from src.pipeline.io_manager import IOManager
from src.image_processing.gantry_angle_image import GantryAngleImage


class GantryAngleTask:
    """Pipeline class for managing gantry angle analysis process.
    Ensures that I/O errors will not be raised and select images to be analysed.
    Each image is analysed and the log is updated."""

    def __init__(self):
        """Initialises GantryAngleTask class. Potential I/O errors and handled early.
        Images to analyse are loaded and GantryAngleImage objects are instantiated for each.
        """

        # ensures that various I/O errors will not be raised at a later date
        IOManager.creation_control()

        # informs user that images are being loaded
        print("Loading images from input folder...")

        # get images to analyse and instantiate GantryAngleImage objects
        self.images = [GantryAngleImage(path) for path in IOManager.pull_images()]

    def run(self):
        """Triggers the rest of the gantry angle analysis pipeline to run.
        Images are analysed, figures are saved and the log is updated."""

        # try to run pipeline, catching errors
        try:

            # inform user that the images are being analysed
            print("Analysing images...")

            # process each image
            for image in self.images:

                # analyse image, internally save results graphics and update log.
                image.analyse()
                IOManager.update_log(image)

            # inform user that all images have been analysed
            print("All images analysed! Quitting programme.")
            time.sleep(5)

        # if any errors are caught, inform user and exit after delay
        except Exception as e:
            traceback.print_exc()
            print(f"Error in runtime: {e}")
            time.sleep(5)
            sys.exit()
