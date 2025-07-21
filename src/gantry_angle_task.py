import sys
import traceback
import time

from src.io_manager import IOManager
from src.image import Image


class GantryAngleTask:
    """Class for running gantry angle analysis task."""

    def __init__(self):
        """Initialises GantryAngleTask class by loading images."""
        IOManager.creation_control()
        print("Loading images from input folder...")
        self.images = [Image(path) for path in IOManager.pull_images()]

    def run(self):
        """Runs gantry angle analysis on loaded images"""
        try:
            print("Analysing images...")
            for image in self.images:
                image.analyse()
                IOManager.update_log(image)
            print("All images analysed! Quitting programme.")
            time.sleep(5)
        except Exception as e:
            traceback.print_exc()
            print(f"Error in runtime: {e}")
            time.sleep(5)
            sys.exit()
