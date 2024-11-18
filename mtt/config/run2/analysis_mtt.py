# coding: utf-8

"""
Configuration of the m(ttbar) analysis.
"""

import os

import law
import order as od


thisdir = os.path.dirname(os.path.abspath(__file__))

#
# the main analysis object
#

analysis_mtt = ana = od.Analysis(
    name="analysis_mtt",
    id=1,
)

# analysis-global versions
ana.x.versions = {}

# files of sandboxes that might be required by remote tasks
# (used in cf.HTCondorWorkflow)
analysis_mtt.x.bash_sandboxes = [
    "$CF_BASE/sandboxes/cf.sh",
    "$CF_BASE/sandboxes/venv_columnar.sh",
    # "$MTT_BASE/sandboxes/venv_columnar_tf.sh",
    "$CF_BASE/sandboxes/venv_ml_tf.sh",
]

# cmssw sandboxes that should be bundled for remote jobs in case they are needed
analysis_mtt.x.cmssw_sandboxes = [
    # "$CF_BASE/sandboxes/cmssw_default.sh",
]

# clear the list when cmssw bundling is disabled
if not law.util.flag_to_bool(os.getenv("MTT_BUNDLE_CMSSW", "1")):
    del analysis_mtt.x.cmssw_sandboxes[:]

# config groups for conveniently looping over certain configs
# (used in wrapper_factory)
ana.x.config_groups = {}

#
# set up configs
#

from mtt.config.run2.config_mtt import add_config
from cmsdb.campaigns.run2_2017_nano_v9 import campaign_run2_2017_nano_v9 as campaign_run2_2017_nano_v9

# default config
config_2017 = add_config(
    analysis_mtt,
    campaign_run2_2017_nano_v9.copy(),
    config_name="run2_mtt_2017_nano_v9",
    config_id=2_17_1,  # 2: Run2 17: year 1: full stat
)

# config with limited number of files
config_2017_limited = add_config(
    analysis_mtt,
    campaign_run2_2017_nano_v9.copy(),
    config_name="run2_mtt_2017_nano_v9_limited",
    config_id=2_17_2,  # 2: Run2 17: year 2: limited stat
    limit_dataset_files=1,
)
