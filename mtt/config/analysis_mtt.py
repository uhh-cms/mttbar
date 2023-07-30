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

analysis_mtt = od.Analysis(
    name="analysis_mtt",
    id=1,
)

# analysis-global versions
analysis_mtt.set_aux("versions", {
})

# files of sandboxes that might be required by remote tasks
# (used in cf.HTCondorWorkflow)
analysis_mtt.x.bash_sandboxes = [
    "$CF_BASE/sandboxes/cf.sh",
    "$CF_BASE/sandboxes/venv_columnar.sh",
    # "$MTT_BASE/sandboxes/venv_columnar_tf.sh",
    "$CF_BASE/sandboxes/venv_ml_tf.sh",
]

# cmssw sandboxes that should be bundled for remote jobs in case they are needed
analysis_mtt.set_aux("cmssw_sandboxes", [
    # "$CF_BASE/sandboxes/cmssw_default.sh",
])


# clear the list when cmssw bundling is disabled
if not law.util.flag_to_bool(os.getenv("MTT_BUNDLE_CMSSW", "1")):
    del analysis_mtt.x.cmssw_sandboxes[:]

# config groups for conveniently looping over certain configs
# (used in wrapper_factory)
analysis_mtt.set_aux("config_groups", {})

# trailing imports for different configs
import mtt.config.config_2017  # noqa
#import mtt.config.config_2022  # noqa
