
"""
Mixins for all the combine methods.
"""

import luigi
from columnflow.tasks.framework.parameters import SettingsParameter


class CommonStatsOptionsMixin:
    # parameters from -h call
    # defaults from combine
    # currently not implemented:
    # prior, significance, lowerLimit, expectSignalMass, unbinned, generateBinnedWorkaround, defineBackgroundOnlyModelParameters

    rMin = luigi.FloatParameter(
        default=0.0,
        significant=True,
        description="Low end of r."
    )

    rMax = luigi.FloatParameter(
        default=20.0,
        significant=True,
        description="High end of r."
    )

    hintStatOnly = luigi.BoolParameter(
        default=False,
        significant=False,
        description="Ignore systematics when computing the hint."
    )

    toysNoSystematics = luigi.BoolParameter(
        default=False,
        significant=False,
        description="Generate all toys with the central value of the nuisance parameters, without fluctuating them."
    )

    toysFrequentist = luigi.BoolParameter(
        default=False,
        significant=False,
        description="Generate all toys in the frequentist way."
    )

    expectSignal = luigi.FloatParameter(
        default=0.0,
        significant=True,
        description="If set to non-zero, generate *signal* toys instead of background ones, with the specified signal strength."
    )

    setParameters = SettingsParameter(
        default={},
        description="Set the values of relevant physics model parameters. Give a comma-separated list of parameter value assignments. Example: CV=1.0,CF=1.0"
    )

    setParameterRanges = SettingsParameter(
        default={},
        description="Set the range of relevant physics model parameters. Give a colon-separated list of parameter ranges. Example: CV=0.0,2.0:CF=0.0,5.0"
    )

    redefineSignalPOIs = luigi.ListParameter(
        default=[],
        description="Redefines the POIs to be this comma-separated list of variables from the workspace."
    )

    freezeParameters = luigi.ListParameter(
        default=[],
        description="Set these parameters as constant. Use --freezeParameters allConstrainedNuisances to freeze all constrained nuisance parameters (i.e doesn't include rateParams etc)"
    )

    freezeNuisanceGroups = luigi.ListParameter(
        default=[],
        description="Set as constant all these groups of nuisance parameters."
    )

    freezeWithAttributes = luigi.ListParameter(
        default=[],
        description="Set as constant all variables carrying one of these attribute strings."
    )

    toys = luigi.IntParameter(
        default=0,
        significant=True,
        description="Number of Toy MC extractions."
    )

    seed = luigi.IntParameter(
        default=123456,
        significant=True,
        description="Random seed for toy MC generation."
    )

    hintMethod = luigi.Parameter(
        default="",
        significant=True,
        description="First run this method to provide a hint on the result."
    )


class CommonIOOptionsMixin:
    # parameters from -h call
    # overrideSnapshotMass

    saveWorkspace = luigi.BoolParameter(
        default=False,
        significant=False,
        description="Save workspace to output root file."
    )

    workspaceName = luigi.Parameter(
        default="w",
        significant=True,
        description="Workspace name, when reading it from or writing it to a ROOT file."
    )

    snapshotName = luigi.Parameter(
        default="",
        significant=True,
        description="Default snapshot name for pre-fit snapshot for reading or writing to workspace."
    )

    modelConfigName = luigi.Parameter(
        default="ModelConfig",
        significant=True,
        description="ModelConfig name, when reading it from or writing it to a ROOT file."
    )

    modelConfigNameB = luigi.Parameter(
        default="%s_bonly",
        significant=True,
        description="Name of the ModelConfig for b-only hypothesis. If not present, it will be made from the signal model taking zero signal strength. A '%s' in the name will be replaced with the modelConfigName."
    )

    bypassFrequentistFit = luigi.BoolParameter(
        default=False,
        significant=False,
        description="Skip actual minimization for constructing frequentist toys (eg because loaded snapshot already corresponds to desired postfit)."
    )

    validateModel = luigi.BoolParameter(
        default=False,
        significant=False,
        description="Perform some sanity checks on the model and abort if they fail."
    )

    saveToys = luigi.BoolParameter(
        default=False,
        significant=False,
        description="Save results of toy MC in output file."
    )

    floatAllNuisances = luigi.IntParameter(
        default=0,
        significant=False,
        description="Make all nuisance parameters floating."
    )

    floatParameters = luigi.ListParameter(
        default=[],
        significant=False,
        description="Set these parameters floating(note freeze will take priority over float), also accepts regexp with syntax 'rgx{<my regexp>}' or 'var{<my regexp>}'."
    )

    freezeAllGlobalObs = luigi.IntParameter(
        default=1,
        significant=False,
        description="Make all global observables constant."
    )

    name = luigi.Parameter(
        default="test",
        significant=True,
        description="Name of the job, affects the name of the output tree."
    )

    mass = luigi.IntParameter(
        default=120,
        significant=True,
        description="Mass to store in the output tree."
    )

    dataset = luigi.Parameter(
        default="data_obs",
        significant=True,
        description="Name of the dataset for observed limit - use this to replace dataset in workspace for example with a toy dataset. Format as file:workspace:object or file:object."
    )

    dataMapName = luigi.Parameter(
        default="data_obs",
        significant=True,
        description="Name of the dataset for observed limit pattern in the datacard."
    )

    toysFile = luigi.Parameter(
        default="",
        significant=True,
        description="Read toy mc or other intermediate results from this file."
    )


class CascadeMinimizerOptionsMixin:
    # parameters from -h call
    # defaults from combine
    # currently not implemented:
    # cminOldRobustMinimize

    cminPoiOnlyFit = luigi.BoolParameter(
        default=False,
        significant=False,
        description="First do a fit floating only the parameter of interest."
    )

    cminPreScan = luigi.BoolParameter(
        default=False,
        significant=False,
        description="Do a scan before first minimization."
    )

    cminPreFit = luigi.FloatParameter(
        default=0.0,
        significant=False,
        description="If set to a value N > 0, a pre-fit with strategy (N-1) will be performed with the constrained nuisance parameters frozen."
    )

    cminApproxPreFitTolerance = luigi.FloatParameter(
        default=0.0,
        significant=False,
        description="If nonzero, first do a pre-fit withcminFallbackAlgo this tolerance (or 10 times the final tolerance, whichever is largest)."
    )

    cminApproxPreFitStrategy = luigi.IntParameter(
        default=0,
        significant=False,
        description="Strategy to use in the pre-fit."
    )

    cminSingleNuisFit = luigi.BoolParameter(
        default=False,
        significant=False,
        description="First perform a minimization of each nuisance parameter individually."
    )

    cminFallbackAlgo = luigi.Parameter(
        default="",
        significant=False,
        description="Fallback algorithms if the default minimizer fails (can use multiple ones). Syntax is algo[,subalgo][,strategy][:tolerance]."
    )

    cminSetZeroPoint = luigi.FloatParameter(
        default=1.0,
        significant=False,
        description="Change the reference point of the NLL to be zero during minimization."
    )

    cminInitialHesse = luigi.IntParameter(
        default=0,
        significant=False,
        description="Call Hesse before the minimization."
    )

    cminFinalHesse = luigi.IntParameter(
        default=0,
        significant=False,
        description="Call Hesse after the minimization."
    )

    cminDefaultMinimizerType = luigi.Parameter(
        default="Minuit2",
        significant=False,
        description="Default minimizer type to use."
    )

    cminDefaultMinimizerAlgo = luigi.Parameter(
        default="Migrad",
        significant=False,
        description="Default minimizer algorithm to use."
    )

    cminDefaultMinimizerTolerance = luigi.FloatParameter(
        default=0.10000000000000001,
        significant=False,
        description="Default minimizer tolerance to use."
    )

    cminDefaultMinimizerPrecision = luigi.FloatParameter(
        default=-1,
        significant=False,
        description="Default minimizer precision to use."
    )

    cminDefaultMinimizerStrategy = luigi.FloatParameter(
        default=1,
        significant=False,
        description="Set the default minimizer (initial) strategy."
    )

    cminRunAllDiscreteCombinations = luigi.BoolParameter(
        default=False,
        significant=False,
        description="Run all combinations for discrete nuisances."
    )

    cminDiscreteMinTol = luigi.FloatParameter(
        default=0.001,
        significant=False,
        description="Tolerance on min NLL for discrete combination iterations."
    )

    cminM2StorageLevel = luigi.IntParameter(
        default=0,
        significant=False,
        description="Storage level for minuit2 (0 = don't store intermediate covariances, 1 = store them)."
    )


class CommonMiscOptionsMixin:
    # parameters from -h call
    # defaults from combine
    # currently not implemented:
    # 

    optimizeSimPdf = luigi.IntParameter(
        default=1,
        significant=False,
        description="Turn on special optimizations of RooSimultaneous. On by default, you can turn it off if it doesn't work for your workspace."
    )

    noMCbonly = luigi.IntParameter(
        default=0,
        significant=False,
        description="Don't create a background-only modelConfig."
    )

    noDefaultPrior = luigi.IntParameter(
        default=0,
        significant=False,
        description="Don't create a default uniform prior."
    )

    rebuildSimPdf = luigi.IntParameter(
        default=0,
        significant=False,
        description="Rebuild the SimPdf from scratch."
    )

    compile = luigi.BoolParameter(
        default=False,
        significant=False,
        description="Compile expressions instead of interpreting them."
    )

    tempDir = luigi.IntParameter(
        default=0,
        significant=False,
        description="Run the program from a temporary directory (automatically on for text datacards or if 'compile' is activated)"
    )

    guessGenMode = luigi.BoolParameter(
        default=False,
        significant=False,
        description="Guess whether to generate binned or unbinned based on dataset."
    )

    genBinnedChannels = luigi.Parameter(
        default="",
        significant=True,
        description="Flag the given channels to be generated binned (irrespectively of how they were flagged at workspace creation)."
    )

    genUnbinnedChannels = luigi.Parameter(
        default="",
        significant=True,
        description="Flag the given channels to be generated unbinned (irrespectively of how they were flagged at workspace creation)."
    )

    text2workspace = luigi.Parameter(
        default="",
        significant=True,
        description="Pass along options to text2workspace (default = none)."
    )

    trackParameters = luigi.ListParameter(
        default=[],
        significant=False,
        description="Keep track of parameters in workspace, also accepts regexp with syntax 'rgx{<my regexp>}' (default = none)."
    )

    trackErrors = luigi.ListParameter(
        default=[],
        significant=False,
        description="Keep track of errors on parameters in workspace, also accepts regexp with syntax 'rgx{<my regexp>}' (default = none)."
    )

    igpMem = luigi.BoolParameter(
        default=False,
        significant=False,
        description="Setup support for memory profiling using IgProf."
    )

    perfCounters = luigi.BoolParameter(
        default=False,
        significant=False,
        description="Dump performance counters at end of job."
    )

    LoadLibrary = luigi.Parameter(
        default="",
        significant=True,
        description="Load library through gSystem->Load(...). Can specify multiple libraries using this option multiple times."
    )

    keyword_value = luigi.ListParameter(
        default=[],
        significant=False,
        description="Set keyword values with 'WORD=VALUE', will replace $WORD with VALUE in datacards. Filename will also be extended with 'WORDVALUE'. Can specify multiple times"
    )

    X_rtd = luigi.Parameter(
        default="",
        significant=True,
        description="Define some constants to be used at runtime (for debugging purposes). The syntax is --X-rtd identifier[=value], where value is an integer and defaults to 1. Can specify multiple times."
    )

    X_fpeMask = luigi.IntParameter(
        default=7,
        significant=True,
        description="Set FPE mask: 1=NaN, 2=Div0, 4=Overfl, 8=Underf, 16=Inexact; 7=default"
    )


class AsymptoticLimitsMixin:
    # parameters from -h call

    rAbsAcc = luigi.FloatParameter(
        default=0.00050000000000000001,
        significant=True,
        description="Absolute accuracy on r to reach to terminate the scan."
    )

    rRelAcc = luigi.FloatParameter(
        default=0.0050000000000000001,
        significant=True,
        description="Relative accuracy on r to reach to terminate the scan."
    )

    run_mode = luigi.ChoiceParameter(
        choices=["blind", "expected", "observed", "both"],
        default="expected",
        significant=True,
        description="What to run: both (default), observed, expected, blind.."
    )

    singlePoint = luigi.BoolParameter(
        default=False,
        significant=True,
        description="Just compute CLs for the given value of r."
    )

    qtilde = luigi.IntParameter(
        default=1,
        significant=True,
        description="Allow only non-negative signal strengths (default is true)."
    )

    rule = luigi.ChoiceParameter(
        choices=["CLs", "Pmu"],
        default="CLs",
        significant=True,
        description="Which rule to use for the test statistic (default is CLs)."
    )

    picky = luigi.BoolParameter(
        default=False,
        significant=True,
        description="Abort on fit failures."
    )

    noFitAsimov = luigi.BoolParameter(
        default=False,
        significant=True,
        description="Use the pre-fit asimov dataset."
    )

    getLimitFromGrid = luigi.BoolParameter(
        default=False,
        significant=True,
        description="Calculates the limit from a grid of r,cls values."
    )

    newExpected = luigi.IntParameter(
        default=1,
        significant=True,
        description="Use the new formula for expected limits (default is true)."
    )

    minosAlgo = luigi.ChoiceParameter(
        choices=["stepping", "minos", "bisection"],
        default="stepping",
        significant=True,
        description="Algorithm to use to get the median expected limit: 'minos' (fastest), 'bisection', 'stepping' (default, most robust)."
    )

    strictBounds = luigi.BoolParameter(
        default=False,
        significant=True,
        description="Take --rMax as a strict upper bound."
    )


class FitterAlgoOptionsMixin:
    # mirrors combine's FitterAlgoBase options group (src/FitterAlgoBase.cc). Unlike the four
    # "Common ..." mixins above, these apply only to MultiDimFit/FitDiagnostics-family methods
    # (AsymptoticLimits does NOT inherit FitterAlgoBase and must not use this mixin) — currently
    # only robustFit is needed (used by Impacts, which runs per-nuisance MultiDimFit scans
    # internally via combineTool.py). The rest of FitterAlgoBase's options (preFitValue, do95,
    # maxFailedSteps, stepSize, setRobustFitAlgo/Strategy/Tolerance, setCrossingTolerance,
    # profilingMode, saveNLL, keepFailures, protectUnbinnedChannels, autoBoundsPOIs, autoMaxPOIs,
    # forceRecreateNLL) aren't declared here — add them if/when actually needed.

    robustFit = luigi.IntParameter(
        default=1,
        significant=False,
        description="Search manually for 1 and 2 sigma bands instead of using Minos (1=on, 0=off)."
    )
