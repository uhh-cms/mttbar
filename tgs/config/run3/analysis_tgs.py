# coding: utf-8

"""
Configuration of the Run 3 m(ttbar) trigger study analysis.
"""

import os

import law
import order as od


thisdir = os.path.dirname(os.path.abspath(__file__))

#
# the main analysis object
#

analysis_tgs = ana = od.Analysis(
    name="analysis_tgs",
    id=2,
)

# analysis-global versions
ana.x.versions = {}

# files of sandboxes that might be required by remote tasks
# (used in cf.HTCondorWorkflow)
ana.x.bash_sandboxes = [
    "$CF_BASE/sandboxes/cf.sh",
    "$CF_BASE/sandboxes/venv_columnar.sh",
    # "$MTT_BASE/sandboxes/venv_columnar_tf.sh",
    "$CF_BASE/sandboxes/venv_ml_tf.sh",
]

# cmssw sandboxes that should be bundled for remote jobs in case they are needed
ana.x.cmssw_sandboxes = [
    # "$CF_BASE/sandboxes/cmssw_default.sh",
]

# clear the list when cmssw bundling is disabled
if not law.util.flag_to_bool(os.getenv("MTT_BUNDLE_CMSSW", "1")):
    del ana.x.cmssw_sandboxes[:]

# config groups for conveniently looping over certain configs
# (used in wrapper_factory)
ana.x.config_groups = {}

#
# set up configs
#

from tgs.config.run3.config_tgs import add_config

from cmsdb.campaigns.run3_2022_preEE_nano_v12 import campaign_run3_2022_preEE_nano_v12 as campaign_run3_2022_preEE_nano_v12  # noqa
from cmsdb.campaigns.run3_2022_postEE_nano_v12 import campaign_run3_2022_postEE_nano_v12 as campaign_run3_2022_postEE_nano_v12  # noqa
# from cmsdb.campaigns.run3_2023_preBPix_nano_v12 import campaign_run3_2023_preBPix_nano_v12 as campaign_run3_2023_preBPix_nano_v12  # noqa
# from cmsdb.campaigns.run3_2023_postBPix_nano_v12 import campaign_run3_2023_postBPix_nano_v12 as campaign_run3_2023_postBPix_nano_v12  # noqa

campaign_run3_2022_preEE_nano_v12.x.EE = "pre"
campaign_run3_2022_postEE_nano_v12.x.EE = "post"
# campaign_run3_2023_preBPix_nano_v12.x.BPix = "pre"
# campaign_run3_2023_postBPix_nano_v12.x.BPix = "post"

# default config
config_2022_preEE = add_config(
    ana,
    campaign_run3_2022_preEE_nano_v12.copy(),
    config_name="run3_tgs_2022_preEE_nano_v12",
    config_id=3_22_11,  # 3: Run3 22: year 1: full stat 1: pre EE
)

config_2022_postEE = add_config(
    ana,
    campaign_run3_2022_postEE_nano_v12.copy(),
    config_name="run3_tgs_2022_postEE_nano_v12",
    config_id=3_22_12,  # 3: Run3 22: year 1: full stat 2: post EE
)

# config_2023_preBPix = add_config(
#     ana,
#     campaign_run3_2023_preBPix_nano_v12.copy(),
#     config_name="run3_tgs_2023_preBPix_nano_v12",
#     config_id=3_23_11,  # 3: Run3 23: year 1: full stat 1: pre BPix
# )

# config_2023_postBPix = add_config(
#     ana,
#     campaign_run3_2023_postBPix_nano_v12.copy(),
#     config_name="run3_tgs_2023_postBPix_nano_v12",
#     config_id=3_23_12,  # 3: Run3 23: year 1: full stat 2: post BPix
# )

# config with limited number of files
config_2022_preEE_limited = add_config(
    ana,
    campaign_run3_2022_preEE_nano_v12.copy(),
    config_name="run3_tgs_2022_preEE_nano_v12_limited",
    config_id=3_22_21,  # 3: Run3 22: year 2: limited stat 1: pre EE
    limit_dataset_files=1,
)

config_2022_postEE_limited = add_config(
    ana,
    campaign_run3_2022_postEE_nano_v12.copy(),
    config_name="run3_tgs_2022_postEE_nano_v12_limited",
    config_id=3_22_22,  # 3: Run3 22: year 2: limited stat 2: post EE
    limit_dataset_files=1,
)

# config_2023_preBPix_limited = add_config(
#     ana,
#     campaign_run3_2023_preBPix_nano_v12.copy(),
#     config_name="run3_tgs_2023_preBPix_nano_v12_limited",
#     config_id=3_23_21,  # 3: Run3 23: year 2: limited stat 1: pre BPix
#     limit_dataset_files=1,
# )

# config_2023_postBPix_limited = add_config(
#     ana,
#     campaign_run3_2023_postBPix_nano_v12.copy(),
#     config_name="run3_tgs_2023_postBPix_nano_v12_limited",
#     config_id=3_23_22,  # 3: Run3 23: year 2: limited stat 2: post BPix
#     limit_dataset_files=1,
# )
