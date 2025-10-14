# coding: utf-8

"""
Configuration of the Run 3 m(ttbar) analysis.
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

analysis_mtt_new = ana_new = od.Analysis(
    name="new_analysis_mtt",
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
ana_new.x.bash_sandboxes = [
    "$CF_BASE/sandboxes/cf.sh",
    "$CF_BASE/sandboxes/venv_columnar.sh",
    # "$MTT_BASE/sandboxes/venv_columnar_tf.sh",
    "$CF_BASE/sandboxes/venv_ml_tf.sh",
]

# cmssw sandboxes that should be bundled for remote jobs in case they are needed
ana.x.cmssw_sandboxes = [
    # "$CF_BASE/sandboxes/cmssw_default.sh",
]
ana_new.x.cmssw_sandboxes = [
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

from mtt.config.run3.config_mtt import add_config
from mtt.config.run3.new_mtt_config import add_new_config

from cmsdb.campaigns.run3_2022_preEE_nano_v12 import campaign_run3_2022_preEE_nano_v12 as campaign_run3_2022_preEE_nano_v12  # noqa
from cmsdb.campaigns.run3_2022_postEE_nano_v12 import campaign_run3_2022_postEE_nano_v12 as campaign_run3_2022_postEE_nano_v12  # noqa
from cmsdb.campaigns.run3_2023_preBPix_nano_v12 import campaign_run3_2023_preBPix_nano_v12 as campaign_run3_2023_preBPix_nano_v12  # noqa
from cmsdb.campaigns.run3_2023_postBPix_nano_v12 import campaign_run3_2023_postBPix_nano_v12 as campaign_run3_2023_postBPix_nano_v12  # noqa
from cmsdb.campaigns.run3_2024_nano_v15 import campaign_run3_2024_nano_v15 as campaign_run3_2024_nano_v15  # noqa

campaign_run3_2022_preEE_nano_v12.x.EE = "pre"
campaign_run3_2022_postEE_nano_v12.x.EE = "post"
campaign_run3_2023_preBPix_nano_v12.x.BPix = "pre"
campaign_run3_2023_postBPix_nano_v12.x.BPix = "post"

#
# 22preEE
# id: 3_22_x1 with x in
# (1/9: full stats, 2/8: limited stats, 3/7: medium limited stats)
# with old/new configs
#

# configs with full statistics
config_2022_preEE = add_config(
    ana,
    campaign_run3_2022_preEE_nano_v12.copy(),
    config_name="run3_mtt_2022_preEE_nano_v12",
    config_id=3_22_11,  # 3: Run3 22: year 1: full stat 1: pre EE
)
config_2022_preEE_new = add_new_config(
    ana_new,
    campaign_run3_2022_preEE_nano_v12.copy(),
    config_name="run3_mtt_2022_preEE_nano_v12_new",
    config_id=3_22_91,  # 3: Run3 22: year 9: full stat, new config 1: pre EE
)

# configs with limited number of files
config_2022_preEE_limited = add_config(
    ana,
    campaign_run3_2022_preEE_nano_v12.copy(),
    config_name="run3_mtt_2022_preEE_nano_v12_limited",
    config_id=3_22_21,  # 3: Run3 22: year 2: limited stat 1: pre EE
    limit_dataset_files=2,
)

config_2022_preEE_limited_new = add_new_config(
    ana_new,
    campaign_run3_2022_preEE_nano_v12.copy(),
    config_name="run3_mtt_2022_preEE_nano_v12_limited_new",
    config_id=3_22_81,  # 3: Run3 22: year 8: limited stat, new config 1: pre EE
    limit_dataset_files=2,
)

# configs with medium limited number of files
config_2022_preEE_medium_limited = add_config(
    ana,
    campaign_run3_2022_preEE_nano_v12.copy(),
    config_name="run3_mtt_2022_preEE_nano_v12_medium_limited",
    config_id=3_22_31,  # 3: Run3 22: year 3: medium limited stat 1: pre EE
    limit_dataset_files=10,
)

config_2022_preEE_medium_limited_new = add_new_config(
    ana_new,
    campaign_run3_2022_preEE_nano_v12.copy(),
    config_name="run3_mtt_2022_preEE_nano_v12_medium_limited_new",
    config_id=3_22_71,  # 3: Run3 22: year 7: medium limited stat, new config 2: post EE
    limit_dataset_files=10,
)

#
# 22postEE
# id: 3_22_x2 with x in
# (1/9: full stats, 2/8: limited stats, 3/7: medium limited stats)
# with old/new configs
#

# configs with full statistics
config_2022_postEE = add_config(
    ana,
    campaign_run3_2022_postEE_nano_v12.copy(),
    config_name="run3_mtt_2022_postEE_nano_v12",
    config_id=3_22_12,  # 3: Run3 22: year 1: full stat 2: post EE
)
config_2022_postEE_new = add_new_config(
    ana_new,
    campaign_run3_2022_postEE_nano_v12.copy(),
    config_name="run3_mtt_2022_postEE_nano_v12_new",
    config_id=3_22_92,  # 3: Run3 22: year 9: full stat, new config 2: post EE
)

# configs with limited number of files
config_2022_postEE_limited = add_config(
    ana,
    campaign_run3_2022_postEE_nano_v12.copy(),
    config_name="run3_mtt_2022_postEE_nano_v12_limited",
    config_id=3_22_22,  # 3: Run3 22: year 2: limited stat 2: post EE
    limit_dataset_files=2,
)
config_2022_postEE_limited_new = add_new_config(
    ana_new,
    campaign_run3_2022_postEE_nano_v12.copy(),
    config_name="run3_mtt_2022_postEE_nano_v12_limited_new",
    config_id=3_22_82,  # 3: Run3 22: year 2: limited stat, new config 2: post EE
    limit_dataset_files=2,
)

# configs with medium limited number of files
config_2022_postEE_medium_limited = add_config(
    ana,
    campaign_run3_2022_postEE_nano_v12.copy(),
    config_name="run3_mtt_2022_postEE_nano_v12_medium_limited",
    config_id=3_22_32,  # 3: Run3 22: year 3: medium limited stat 2: post EE
    limit_dataset_files=10,
)
config_2022_postEE_medium_limited_new = add_new_config(
    ana_new,
    campaign_run3_2022_postEE_nano_v12.copy(),
    config_name="run3_mtt_2022_postEE_nano_v12_medium_limited_new",
    config_id=3_22_72,  # 3: Run3 22: year 3: medium limited stat, new config 2: post EE
    limit_dataset_files=10,
)

#
# 23prePBix
# id: 3_23_x1 with x in
# (1/9: full stats, 2/8: limited stats, 3/7: medium limited stats)
# with old/new configs
#

# configs with full statistics
config_2023_preBPix_new = add_new_config(
    ana_new,
    campaign_run3_2023_preBPix_nano_v12.copy(),
    config_name="run3_mtt_2023_preBPix_nano_v12_new",
    config_id=3_23_91,  # 3: Run3 23: year 9: full stat, new config 1: pre BPix
)

config_2023_postBPix_new = add_new_config(
    ana_new,
    campaign_run3_2023_postBPix_nano_v12.copy(),
    config_name="run3_mtt_2023_postBPix_nano_v12_new",
    config_id=3_23_92,  # 3: Run3 23: year 9: full stat, new config 2: post BPix
)

# configs with limited number of files
config_2023_preBPix_limited_new = add_new_config(
    ana_new,
    campaign_run3_2023_preBPix_nano_v12.copy(),
    config_name="run3_mtt_2023_preBPix_nano_v12_limited_new",
    config_id=3_23_81,  # 3: Run3 23: year 8: limited stat, new config 1: pre BPix
    limit_dataset_files=2,
)

config_2023_postBPix_limited_new = add_new_config(
    ana_new,
    campaign_run3_2023_postBPix_nano_v12.copy(),
    config_name="run3_mtt_2023_postBPix_nano_v12_limited_new",
    config_id=3_23_82,  # 3: Run3 23: year 8: limited stat, new config 2: post BPix
    limit_dataset_files=2,
)

# configs with medium limited number of files
config_2023_preBPix_medium_limited_new = add_new_config(
    ana_new,
    campaign_run3_2023_preBPix_nano_v12.copy(),
    config_name="run3_mtt_2023_preBPix_nano_v12_medium_limited_new",
    config_id=3_23_71,  # 3: Run3 23: year 7: medium limited stat, new config 1: pre BPix
    limit_dataset_files=10,
)

config_2023_postBPix_medium_limited_new = add_new_config(
    ana_new,
    campaign_run3_2023_postBPix_nano_v12.copy(),
    config_name="run3_mtt_2023_postBPix_nano_v12_medium_limited_new",
    config_id=3_23_72,  # 3: Run3 23: year 7: medium limited stat, new config 2: post BPix
    limit_dataset_files=10,
)

#
# 24
# id: 3_24_x1 with x in
# (1/9: full stats, 2/8: limited stats, 3/7: medium limited stats)
# with old/new configs
#

# configs with full statistics
config_2024_new = add_new_config(
    ana_new,
    campaign_run3_2024_nano_v15.copy(),
    config_name="run3_mtt_2024_nano_v15_new",
    config_id=3_24_11,  # 3: Run3 24: year 1: full stat 1: 24
)

# configs with limited number of files
config_2024_limited_new = add_new_config(
    ana_new,
    campaign_run3_2024_nano_v15.copy(),
    config_name="run3_mtt_2024_nano_v15_limited_new",
    config_id=3_24_21,  # 3: Run3 24
    limit_dataset_files=2,
)

# configs with medium limited number of files
config_2024_medium_limited_new = add_new_config(
    ana_new,
    campaign_run3_2024_nano_v15.copy(),
    config_name="run3_mtt_2024_nano_v15_medium_limited_new",
    config_id=3_24_31,  # 3: Run3 24
    limit_dataset_files=10,
)


# config_2023_preBPix = add_config(
#     ana,
#     campaign_run3_2023_preBPix_nano_v12.copy(),
#     config_name="run3_mtt_2023_preBPix_nano_v12",
#     config_id=3_23_11,  # 3: Run3 23: year 1: full stat 1: pre BPix
# )

# config_2023_postBPix = add_config(
#     ana,
#     campaign_run3_2023_postBPix_nano_v12.copy(),
#     config_name="run3_mtt_2023_postBPix_nano_v12",
#     config_id=3_23_12,  # 3: Run3 23: year 1: full stat 2: post BPix
# )


# # configs with limited number of files
# config_2023_preBPix_limited = add_config(
#     ana,
#     campaign_run3_2023_preBPix_nano_v12.copy(),
#     config_name="run3_mtt_2023_preBPix_nano_v12_limited",
#     config_id=3_23_21,  # 3: Run3 23: year 2: limited stat 1: pre BPix
#     limit_dataset_files=1,
# )

# config_2023_postBPix_limited = add_config(
#     ana,
#     campaign_run3_2023_postBPix_nano_v12.copy(),
#     config_name="run3_mtt_2023_postBPix_nano_v12_limited",
#     config_id=3_23_22,  # 3: Run3 23: year 2: limited stat 2: post BPix
#     limit_dataset_files=1,
# )
