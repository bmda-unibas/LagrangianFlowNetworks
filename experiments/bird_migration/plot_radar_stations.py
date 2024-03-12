from datasets.bird_data import BirdDatasetMultipleNightsLeaveOutMiddle

from experiments.bird_migration.plot_util import plot_radar_positions
BirdData = BirdDatasetMultipleNightsLeaveOutMiddle

complete_ds = BirdData(subset="all", start_date="2018-04-05", end_date="2018-04-07")
plot_radar_positions(complete_ds, suffix="middle")
