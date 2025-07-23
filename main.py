"""
main.py

Entrypoint for Gantry Angle Calculator application. Imports the GantryAngleTask class
from the pipeline package and runs the task to begin the slice thickness analysis process.

Written by Nathan Crossley, 2025.
"""

from src.pipeline.gantry_angle_task import GantryAngleTask

if __name__ == "__main__":

    # instantiate and run GantryAngleTask
    slice_thickness_task = GantryAngleTask()
    slice_thickness_task.run()
