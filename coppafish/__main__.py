# This file allows the program to be called from the command line without
# entering the Python interpreter.  To call, use:
#
#     python3 -m coppafish inifile.ini

import os
import sys
import textwrap

import numpy as np

from coppafish import Notebook, Viewer, export_to_pciseq, run_pipeline
from coppafish.setup import file_names


def print_usage(message=None):
    if message:
        message = f"\n\n    ERROR: {message}"
    else:
        message = ""
    USAGE = f"""
    === coppaFISH processing software ===

    To run pipeline, pass a single argument containing the name of the config file.  E.g.,

        python3 -m coppafish config.ini
    
    To open the results viewer, pass a second argument -view. E.g.,
    
        python3 -m coppafish config.ini -view
    {message}
    """
    exit(textwrap.dedent(USAGE))


# Ensure there is exactly one argument, and it is an ini file
if len(sys.argv) == 1:
    print_usage("Please pass the config file as an argument")
if len(sys.argv) >= 4:
    print_usage(
        f"Please only pass config file as first argument and optionally -view as the second.\n"
        f"But {len(sys.argv)-1} arguments passed:\n{sys.argv[1:]}"
    )
if sys.argv[1] in ["--help", "-h"]:
    print_usage()
if not os.path.exists(sys.argv[1]):
    print_usage(f"Cannot find path {sys.argv[1]}, please specify a valid path")

if len(sys.argv) == 2:
    run_pipeline(sys.argv[1])
if len(sys.argv) == 3:
    if not np.isin(sys.argv[2], ["view", "-view", "-plot", "plot", "export", "-export"]):
        print_usage(f"To plot results, second argument should be -view or -export but {sys.argv[2]} given")
    else:
        nb = Notebook(sys.argv[1])
        if np.isin(sys.argv[2], ["export", "-export"]):
            export_to_pciseq(nb, file_names.get_file_names(nb.basic_info, sys.argv[1]))
        else:
            Viewer(nb)
