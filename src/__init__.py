"""Helper code for the NASA/JPL close-approach EDA project.

This file only marks the folder `src` as a Python package, so the notebooks can write
`from src import cleaning`. (C++ analogy: it is like a folder of .h/.cpp files that you
#include from main.)

Modules
-------
nasa_api       download data from the NASA/JPL CAD API and save it
cleaning       clean and check the raw data, plus small summary helpers
features       add new columns (unit conversions, categories)
visualization  plotting helpers
"""
