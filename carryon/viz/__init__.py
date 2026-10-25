"""One script per figure ID (fig_<ID>.py, e.g. fig_F2_1.py) and the trajectory viewer.

Each script writes figures/<ID>.png, figures/<ID>.svg and figures/<ID>.inputs.json, a JSON list of
the run ids it read; run cards use that list to link back to their figures.
"""
