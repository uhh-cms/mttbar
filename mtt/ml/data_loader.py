"""
Collection of classes to load and prepare data for machine learning.
Taken from hbw analysis.
"""

from __future__ import annotations

import law
import order as od

from columnflow.util import maybe_import
from columnflow.columnar_util import remove_ak_column
from columnflow.ml import MLModel
from mtt.ml.helper import predict_numpy_on_batch
from mtt.util import timeit

ak = maybe_import("awkward")
np = maybe_import("numpy")

logger = law.logger.get_logger(__name__)


def get_proc_mask(
    events: ak.Array,
    proc: str | od.Process,
    config_inst: od.Config | None = None,
) -> tuple(np.ndarray, list):
    """
    Creates the mask selecting events belonging to the process *proc* and a list of all ids belonging to this process.

    :param events: Event array
    :param proc: Either string or process instance.
    :param config_inst: An instance of the Config, can be None if Porcess instance is given.
    :return process mask and the corresponding process ids
    """
    # get process instance
    if config_inst:
        proc_inst = config_inst.get_process(proc)
    elif isinstance(proc, od.Process):
        proc_inst = proc

    proc_id = events.process_id
    unique_proc_ids = set(proc_id)

    # get list of Ids that are belonging to the process and are present in the event array
    sub_id = [
        proc_inst.id
        for proc_inst, _, _ in proc_inst.walk_processes(include_self=True)
        if proc_inst.id in unique_proc_ids
    ]

    # Create process mask
    proc_mask = np.isin(proc_id, sub_id)
    return proc_mask, sub_id


def input_features_sanity_checks(ml_model_inst: MLModel, input_features: list[str]):
    """
    Perform sanity checks on the input features.

    :param ml_model_inst: An instance of the MLModel class.
    :param input_features: A list of strings representing the input features.
    :raises Exception: If input features are not ordered in the same way for all datasets.
    """
    # check if input features are ordered in the same way for all datasets
    if getattr(ml_model_inst, "input_features_ordered", None):
        if ml_model_inst.input_features_ordered != input_features:
            raise Exception(
                f"Input features are not ordered in the sme way for all datasets. "
                f"Expected: {ml_model_inst.input_features_ordered}, "
                f"got: {input_features}",
            )
    else:
        # if not already set, bookkeep input features in the ml_model_inst aswell
        ml_model_inst.input_features_ordered = input_features

    # check that the input features contain exactly what was requested by the MLModel
    if set(input_features) != set(ml_model_inst.input_features):
        raise Exception(
            f"Input features do not match the input features requested by the MLModel. "
            f"Expected: {ml_model_inst.input_features}, got: {input_features}",
        )


class MLDatasetLoader:
    """
    Helper class to conveniently load ML training data from an awkward array.

    Depends on following parameters of the ml_model_inst:
    - input_features: A set of strings representing the input features we want to keep.
    - train_val_test_split: A tuple of floats representing the split of the data into training, validation, and testing.
    - processes: A tuple of strings representing the processes. Can be parallelized over.
    """

    # shuffle the data in *load_split_data* method
    shuffle: bool = True

    input_arrays: tuple = ("features", "weights", "train_weights", "equal_weights")
    evaluation_arrays: tuple = ("prediction",)

    def __init__(
        self,
        ml_model_inst: MLModel,
        process: "str",
        events: ak.Array,
        stats: dict | None = None,
        skip_mask=False,
    ):
        """
        Initializes the MLDatasetLoader with the given parameters.

        :param ml_model_inst: An instance of the MLModel class.
        :param process: A string representing the process.
        :param events: An awkward array representing the events.
        :param stats: A dictionary containing merged stats per training process.
        :raises Exception: If input features are not ordered in the same way for all datasets.

        .. note:: The method prepares the weights, bookkeeps the order of input features,
        removes columns that are not used as training features, and transforms events into a numpy array.
        """
        self._ml_model_inst = ml_model_inst
        self._process = process
        self._skip_mask = skip_mask

        proc_mask, _ = get_proc_mask(events, process, ml_model_inst.config_inst)
        self._stats = stats
        # NOTE: the skip_mask is currently used both for skipping process mask and for removing negative weights.
        # we might want to separate the negative weights handling into some other flag.
        if skip_mask:
            self._events = events
        else:
            self._events = events[proc_mask]
            self._events = events[events.event_weight >= 0.0]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.ml_model_inst.cls_name}, {self.process})"

    @property
    def hyperparameter_deps(self) -> set:
        """
        Hyperparameters that are required to be set in the MLModel class. If they are changed,
        then tasks using this class need to be re-run.
        """
        # TODO: store values of hyperparameters as task output
        return {"input_features", "train_val_test_split", "input_features_ordered"}

    @property
    def parameters(self):
        """
        Values of the MLModel parameters that the MLDatasetLoader depends on.
        """
        if hasattr(self, "_parameters"):
            return self._parameters

        self._parameters = {
            param: getattr(self.ml_model_inst, param, None)
            for param in self.hyperparameter_deps
        }
        return self._parameters

    @property
    def skip_mask(self):
        return self._skip_mask

    @property
    def ml_model_inst(self):
        return self._ml_model_inst

    @property
    def process(self):
        return self._process

    @property
    def process_inst(self):
        return self.ml_model_inst.config_inst.get_process(self.process)

    @property
    def input_features(self) -> tuple:
        if not hasattr(self, "_input_features"):
            # input features are initialized with the features propery
            self.features

        return self._input_features

    @property
    def stats(self) -> dict:
        return self._stats

    @property
    def weights(self) -> np.ndarray:
        if not hasattr(self, "_weights"):
            self._weights = ak.to_numpy(self._events.event_weight).astype(np.float32)

        return self._weights

    @property
    def features(self) -> np.ndarray:
        if hasattr(self, "_features"):
            return self._features

        # work with a copy of the events
        features = self._events.MLInput

        # Create a mapping from raw feature names to MLInput column names
        raw_ml_features = set(self.ml_model_inst.input_features)
        available_columns = set(f"MLInput.{var}" for var in features.fields)

        logger.debug(f"Raw ML features requested: {raw_ml_features}")
        logger.debug(f"Available MLInput columns: {available_columns}")

        # remove columns that are not used as training features
        for var in features.fields:
            col = f"MLInput.{var}"
            # Check if the raw feature name (without MLInput. prefix) is in the requested features
            if var not in raw_ml_features:
                features = remove_ak_column(features, var)
                logger.debug(f"Removed column {col} from features for process {self.process}.")

        # bookkeep order of input features (store the raw names without MLInput. prefix)
        self._input_features = tuple(features.fields)

        # Create a mapping for sanity checks - convert raw names to what the ML model expects
        mapped_features_for_check = tuple(f"MLInput.{feat}" if not feat.startswith("MLInput.") else feat for feat in self._input_features)  # noqa

        try:
            # Temporarily modify the ml_model_inst.input_features for the sanity check
            original_features = self.ml_model_inst.input_features
            # Convert the ml_model raw features to MLInput format for comparison
            expected_with_prefix = set(f"MLInput.{feat}" if not feat.startswith("MLInput.") else feat for feat in original_features)  # noqa

            # Create a temporary modified ml_model_inst for the sanity check
            import types
            temp_ml_model = types.SimpleNamespace()
            temp_ml_model.__dict__.update(self.ml_model_inst.__dict__)
            temp_ml_model.input_features = expected_with_prefix

            input_features_sanity_checks(temp_ml_model, mapped_features_for_check)

        except Exception as e:
            logger.error(f"Input features sanity check failed: {e}")
            logger.error(f"Expected (with MLInput prefix): {expected_with_prefix}")
            logger.error(f"Got (with MLInput prefix): {set(mapped_features_for_check)}")

        # transform features into numpy npdarray
        # NOTE: when converting to numpy, the awkward array seems to stay in memory...
        features = ak.to_numpy(features)
        features = features.astype(
            [(name, np.float32) for name in features.dtype.names], copy=False,
        ).view(np.float32).reshape((-1, len(features.dtype)))

        # check for infinite values
        if np.any(~np.isfinite(features)):
            mask = ~np.isfinite(features)
            n_total = mask.sum()
            n_nan = np.sum(np.isnan(features))
            n_posinf = np.sum(features == np.inf)
            n_neginf = np.sum(features == -np.inf)
            logger.error(f"Found {n_total} non-finite values in input features for process {self.process}:")
            logger.error(f" - NaNs: {n_nan}")
            logger.error(f" - +inf: {n_posinf}")
            logger.error(f" - -inf: {n_neginf}")

            # Additional debugging: identify which features contain non-finite values
            logger.error("Debugging non-finite values:")

            # Check each feature column separately
            for i, feature_name in enumerate(self._input_features):
                feature_col = features[:, i]
                if np.any(~np.isfinite(feature_col)):
                    n_nan_col = np.sum(np.isnan(feature_col))
                    n_posinf_col = np.sum(feature_col == np.inf)
                    n_neginf_col = np.sum(feature_col == -np.inf)
                    n_total_col = n_nan_col + n_posinf_col + n_neginf_col

                    logger.error(f"  Feature '{feature_name}' (col {i}): {n_total_col} non-finite values")
                    logger.error(f"    - NaNs: {n_nan_col}, +inf: {n_posinf_col}, -inf: {n_neginf_col}")

                    # Show statistics for this feature
                    finite_mask = np.isfinite(feature_col)
                    if np.any(finite_mask):
                        finite_values = feature_col[finite_mask]
                        logger.error(f"    - Finite values: min={np.min(finite_values):.6f}, max={np.max(finite_values):.6f}, mean={np.mean(finite_values):.6f}")  # noqa
                    else:
                        logger.error("    - No finite values found!")

                    # Show first few problematic indices and values
                    problem_indices = np.where(~np.isfinite(feature_col))[0][:5]  # First 5 problematic indices
                    logger.error(f"    - First problematic indices: {problem_indices}")
                    for idx in problem_indices:
                        val = feature_col[idx]
                        if np.isnan(val):
                            val_type = "NaN"
                        elif val == np.inf:
                            val_type = "+inf"
                        elif val == -np.inf:
                            val_type = "-inf"
                        else:
                            val_type = "other"
                        logger.error(f"      Event {idx}: {val} ({val_type})")

            # Check if specific events have multiple problematic features
            events_with_problems = np.any(mask, axis=1)
            problematic_event_indices = np.where(events_with_problems)[0]
            n_problematic_events = len(problematic_event_indices)

            logger.error(f"Found {n_problematic_events} events with non-finite features")

            if n_problematic_events > 0:
                # Show details for first few problematic events
                logger.error("Details for first few problematic events:")
                for i, event_idx in enumerate(problematic_event_indices[:3]):  # First 3 events
                    event_mask = mask[event_idx]
                    problematic_features = np.where(event_mask)[0]

                    logger.error(f"  Event {event_idx}: {len(problematic_features)} problematic features")
                    for feat_idx in problematic_features:
                        feat_name = self._input_features[feat_idx]
                        feat_val = features[event_idx, feat_idx]
                        logger.error(f"    {feat_name}: {feat_val}")

            # Check if there's a pattern in the original awkward array
            logger.error("Checking original MLInput data before numpy conversion:")
            orig_features = self._events.MLInput
            for feature_name in self._input_features:
                if f"MLInput.{feature_name}" in orig_features.fields:
                    orig_col = ak.to_numpy(orig_features[f"MLInput.{feature_name}"])
                    if np.any(~np.isfinite(orig_col)):
                        logger.error(f"  Non-finite values also present in original '{feature_name}' column")
                    else:
                        logger.error(f"  Original '{feature_name}' column appears clean - issue in conversion?")

            raise Exception(f"Found non-finite values in input features for process {self.process}.")

        self._features = features

        return self._features

    @property
    def n_events(self) -> int:
        if hasattr(self, "_n_events"):
            return self._n_events
        self._n_events = len(self.weights)
        return self._n_events

    @property
    def shuffle_indices(self) -> np.ndarray:
        if hasattr(self, "_shuffle_indices"):
            return self._shuffle_indices

        self._shuffle_indices = np.random.permutation(self.n_events)
        return self._shuffle_indices

    @property
    def num_event_per_process(self) -> str:
        if not self.skip_mask:
            return "num_events_pos_weights_per_process"
        else:
            return "num_events_per_process"

    def get_xsec_train_weights(self) -> np.ndarray:
        """
        Weighting such that each event has roughly the same weight,
        sub processes are weighted accoridng to their cross section
        """
        if hasattr(self, "_xsec_train_weights"):
            return self._xsec_train_weights

        if not self.stats:
            raise Exception("cannot determine train weights without stats")

        _, sub_id = get_proc_mask(self._events, self.process, self.ml_model_inst.config_inst)
        sum_weights = np.sum([self.stats[self.process]["sum_pos_weights_per_process"][str(id)] for id in sub_id])
        num_events = np.sum(
            [self.stats[self.process][self.num_event_per_process][str(id)] for id in sub_id],
        )
        # if not self.skip_mask:
        #     num_events = np.sum(
        #         [self.stats[self.process]["num_events_pos_weights_per_process"][str(id)] for id in sub_id],
        #     )
        # else:
        #     num_events = np.sum(
        #         [self.stats[self.process]["num_events_per_process"][str(id)] for id in sub_id],
        #     )

        xsec_train_weights = self.weights / sum_weights * num_events

        return xsec_train_weights

    def get_equal_train_weights(self) -> np.ndarray:
        """
        Weighting such that events of each sub processes are weighted equally
        """
        if hasattr(self, "_equally_train_weights"):
            return self._equal_train_weights

        if not self.stats:
            raise Exception("cannot determine train weights without stats")

        combined_proc_inst = self.ml_model_inst.config_inst.get_process(self.process)
        _, sub_id_proc = get_proc_mask(self._events, self.process, self.ml_model_inst.config_inst)
        num_events = np.sum(
            [self.stats[self.process][self.num_event_per_process][str(id)] for id in sub_id_proc],
        )
        # if not self.skip_mask:
        #     num_events = np.sum(
        #         [self.stats[self.process]["num_events_pos_weights_per_process"][str(id)] for id in sub_id_proc],
        #     )
        # else:
        #     num_events = np.sum([self.stats[self.process]["num_events_per_process"][str(id)] for id in sub_id_proc])
        targeted_sum_of_weights_per_process = (
            num_events / len(combined_proc_inst.x.ml_config.sub_processes)
        )
        equal_train_weights = ak.full_like(self.weights, 1.)
        sub_class_factors = {}

        for proc in combined_proc_inst.x.ml_config.sub_processes:
            proc_mask, sub_id = get_proc_mask(self._events, proc, self.ml_model_inst.config_inst)
            sum_pos_weights_per_sub_proc = 0.
            sum_pos_weights_per_proc = self.stats[self.process]["sum_pos_weights_per_process"]

            for id in sub_id:
                id = str(id)
                if id in self.stats[self.process]["num_events_per_process"]:
                    sum_pos_weights_per_sub_proc += sum_pos_weights_per_proc[id]

            if sum_pos_weights_per_sub_proc == 0:
                norm_const_per_proc = 1.
                logger.info(
                    f"No weight sum found in stats for sub process {proc}."
                    f"Normalization constant set to 1 but results are probably not correct.")
            else:
                norm_const_per_proc = targeted_sum_of_weights_per_process / sum_pos_weights_per_sub_proc
                logger.info(f"Normalizing constant for {proc} is {norm_const_per_proc}")

            sub_class_factors[proc] = norm_const_per_proc
            equal_train_weights = np.where(proc_mask, self.weights * norm_const_per_proc, equal_train_weights)

        return equal_train_weights

    @property
    def train_weights(self) -> np.ndarray:
        """
        Weighting according to the parameters set in the ML model config
        """
        if hasattr(self, "_train_weights"):
            return self._train_weights

        if not self.stats:
            raise Exception("cannot determine train weights without stats")

        # TODO: hier muss np.float gemacht werden
        proc = self.process
        proc_inst = self.ml_model_inst.config_inst.get_process(proc)
        if proc_inst.x("ml_config", None) and proc_inst.x.ml_config.weighting == "equal":
            train_weights = self.get_equal_train_weights()
        else:
            train_weights = self.get_xsec_train_weights()

        self._train_weights = ak.to_numpy(train_weights).astype(np.float32)

        return self._train_weights

    @property
    def equal_weights(self) -> np.ndarray:
        """
        Weighting such that each process has roughly the same sum of weights
        """
        if hasattr(self, "_validation_weights"):
            return self._validation_weights

        if not self.stats:
            raise Exception("cannot determine val weights without stats")

        # TODO: per process pls [done] and now please tidy up
        processes = self.ml_model_inst.processes
        num_events_per_process = {}
        for proc in processes:
            id_list = list(self.stats[proc]["num_events_per_process"].keys())
            proc_inst = self.ml_model_inst.config_inst.get_process(proc)
            sub_id = [
                p_inst.id
                for p_inst, _, _ in proc_inst.walk_processes(include_self=True)
                if str(p_inst.id) in id_list
            ]
            if proc == self.process:
                sum_abs_weights = np.sum([
                    self.stats[self.process]["sum_abs_weights_per_process"][str(id)] for id in sub_id
                ])
            num_events_per_proc = np.sum([self.stats[proc]["num_events_per_process"][str(id)] for id in sub_id])
            num_events_per_process[proc] = num_events_per_proc

        validation_weights = self.weights / sum_abs_weights * max(num_events_per_process.values())
        self._validation_weights = ak.to_numpy(validation_weights).astype(np.float32)

        return self._validation_weights

    @property
    def get_data_split(self) -> tuple[int, int]:
        """
        Get the data split for training, validation and testing.

        :param data: The data to be split.
        :return: The end indices for the training and validation data.
        """
        logger.debug("Determining data split for training, validation, and testing.")
        if hasattr(self, "_train_end") and hasattr(self, "_val_end"):
            logger.debug(f"Using cached data split: {self._train_end}, {self._val_end}")
            return self._train_end, self._val_end

        data_split = np.array(self.ml_model_inst.train_val_test_split)
        logger.debug(f"Original data split: {data_split}")
        data_split = data_split / np.sum(data_split)
        logger.debug(f"Normalized data split: {data_split}")

        self._train_end = int(data_split[0] * self.n_events)
        logger.debug(f"Training data end index: {self._train_end}")
        self._val_end = int((data_split[0] + data_split[1]) * self.n_events)
        logger.debug(f"Validation data end index: {self._val_end}")

        return self._train_end, self._val_end

    def load_split_data(self, data: np.array | str) -> tuple[np.ndarray]:
        """
        Function to split data into training, validation, and test sets.

        :param data: The data to be split. If a string is provided, it is treated as an attribute name.
        :return: The training, validation, and test data.
        """
        if isinstance(data, str):
            data = getattr(self, data)
        train_end, val_end = self.get_data_split

        if self.shuffle:
            data = data[self.shuffle_indices]
        logger.debug(f"Data split into {train_end} training, {val_end - train_end} validation, {self.n_events - val_end} test samples.")  # noqa

        return data[:train_end], data[train_end:val_end], data[val_end:]

    def cleanup(self):
        """
        Explicitly clean up memory-intensive cached arrays.
        Call this method when you're done with the data to ensure memory is freed.
        """
        import gc

        # List all cached array attributes for MLDatasetLoader
        cached_attrs = [
            "_features", "_weights", "_train_weights", "_equal_weights",
            "_shuffle_indices", "_input_features", "_parameters",
            # "_events"  # Include the awkward array events
        ]

        for attr in cached_attrs:
            if hasattr(self, attr):
                # Get the attribute and explicitly delete it
                arr = getattr(self, attr, None)
                if arr is not None:
                    if hasattr(arr, "__array__"):  # numpy array
                        del arr
                    elif hasattr(arr, "__del__"):  # awkward array or other objects with destructor
                        del arr
                    delattr(self, attr)

        # Force garbage collection
        gc.collect()


class MLProcessData:
    """
    Helper class to conveniently load ML training data from the MLPreTraining task outputs.

    Data is merged for all folds except the evaluation_fold.

    Implements the following parameters of the ml_model_inst:
    - negative_weights: A string representing the handling of negative weights.
    """

    shuffle = False

    input_arrays: tuple = ("features", "weights", "train_weights", "equal_weights", "target", "labels")
    evaluation_arrays: tuple = ("prediction",)

    def __init__(
        self,
        ml_model_inst: MLModel,
        inputs,
        data_split: str,
        processes: str,
        evaluation_fold: int,
        fold_modus: str = "all_except_evaluation_fold",
    ):
        self._ml_model_inst = ml_model_inst

        self._input = inputs
        self._data_split = data_split
        self._processes = law.util.make_list(processes)
        self._evaluation_fold = evaluation_fold

        assert fold_modus in ("all_except_evaluation_fold", "evaluation_only", "all")
        self._fold_modus = fold_modus

        # initialize input features
        self.input_features

    def cleanup(self):
        """
        Explicitly clean up memory-intensive cached arrays.
        Call this method when you're done with the data to ensure memory is freed.
        """
        import gc

        # List all cached array attributes
        cached_attrs = [
            "_features", "_weights", "_train_weights", "_equal_weights",
            "_target", "_labels", "_prediction", "_m_negative_weights",
            "_shuffle_indices", "_input_features", "_n_events", "_folds",
        ]

        for attr in cached_attrs:
            if hasattr(self, attr):
                # Get the attribute and explicitly delete it
                arr = getattr(self, attr, None)
                if arr is not None:
                    if hasattr(arr, "__array__"):  # numpy array
                        # For numpy arrays, explicitly delete the data
                        del arr
                    delattr(self, attr)

        # Force garbage collection
        gc.collect()

    def __del__(self):
        """
        Destructor for the MLDatasetLoader class.
        Note: __del__ is not guaranteed to be called, use cleanup() explicitly.
        """
        try:
            self.cleanup()
        except Exception:
            # Ignore errors in destructor
            pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self._ml_model_inst.cls_name}, {self._data_split}, {self._processes})"

    @property
    def process_insts(self) -> list[od.process]:
        return [self._ml_model_inst.config_inst.get_process(proc) for proc in self._processes]

    @property
    def shuffle_indices(self) -> np.ndarray:
        if hasattr(self, "_shuffle_indices"):
            return self._shuffle_indices

        self._shuffle_indices = np.random.permutation(self.n_events)
        return self._shuffle_indices

    @property
    def input_features(self) -> tuple[str]:
        if hasattr(self, "_input_features"):
            return self._input_features

        # load input features for all folds and check consistency between them and with the ml_model_inst
        for process in self._processes:
            for i in range(self._ml_model_inst.folds):
                self._input_features = self._input["input_features"][process][i].load(formatter="pickle")
                input_features_sanity_checks(self._ml_model_inst, self._input_features)

        return self._input_features

    @property
    def n_events(self) -> int:
        if hasattr(self, "_n_events"):
            return self._n_events

        # NOTE: this requires us to load labels. Might not be the optimal choice
        self._n_events = len(self.labels)
        return self._n_events

    @property
    def folds(self) -> tuple[int]:
        """ Property to set the folds for which to merge the data """
        if hasattr(self, "_folds"):
            return self._folds

        if self._fold_modus == "all_except_evaluation_fold":
            self._folds = list(range(self._ml_model_inst.folds))
            self._folds.remove(self._evaluation_fold)
        elif self._fold_modus == "evaluation_only":
            self._folds = [self._evaluation_fold]
        elif self._fold_modus == "all":
            self._folds = list(range(self._ml_model_inst.folds))
        else:
            raise Exception(f"unknown fold modus {self._fold_modus} for MLProcessData")
        return self._folds

    @timeit
    def load_all(self):
        """
        Convenience function to load all data into memory.
        """
        logger.info(f"Loading all data for processes {self._processes} in {self._data_split} set in memory.")
        self.features
        self.weights
        self.train_weights
        self.equal_weights
        self.target
        self.labels
        # do not load prediction because it can only be loaded after training
        # self.prediction

    def load_file(self, data_str, data_split, process, fold):
        """
        Load a file from the input dictionary.
        """
        return self._input[data_str][data_split][process][fold].load(formatter="numpy")

    def load_labels(self, data_split, process, fold):
        """
        Load the labels for a given process and fold.
        """
        proc_inst = self._ml_model_inst.config_inst.get_process(process)
        if not proc_inst.has_aux("ml_id"):
            logger.warning(
                f"Process {process} does not have an ml_id. Label will be set to -1.",
            )
            ml_id = -1
        else:
            ml_id = proc_inst.x.ml_id

        # load any column to get the array length
        weights = self.load_file("weights", data_split, process, fold)

        labels = np.ones(len(weights), dtype=np.int32) * ml_id
        return labels

    def load_data(self, data_str: str) -> np.ndarray:
        """
        Load data from the input dictionary. Options for data_str are "features", "weights", "train_weights",
        "equal_weights", "labels", and "prediction".
        When the data is loaded, it is concatenated over all processes and folds.
        When the *shuffle* attribute is set to True, the data is shuffled using the *shuffle_indices* attribute.
        """
        if data_str not in ("features", "weights", "train_weights", "equal_weights", "labels", "prediction"):
            logger.warning(f"Unknown data string {data_str} for MLProcessData.")
        data = []
        for process in self._processes:
            for fold in self.folds:
                if data_str == "labels":
                    fold_data = self.load_labels(self._data_split, process, fold)
                else:
                    fold_data = self.load_file(data_str, self._data_split, process, fold)
                if np.any(~np.isfinite(fold_data)):
                    raise Exception(f"Found non-finite values in {data_str} for {process} in fold {fold}.")
                data.append(fold_data)

        data = np.concatenate(data)
        if self.shuffle:
            data = data[self.shuffle_indices]
        return data

    @property
    def features(self) -> np.ndarray:
        if hasattr(self, "_features"):
            return self._features

        self._features = self.load_data("features")
        return self._features

    @property
    def weights(self) -> np.ndarray:
        if hasattr(self, "_weights"):
            return self._weights

        self._weights = self.load_data("weights")
        return self._weights

    @property
    def m_negative_weights(self) -> np.ndarray:
        if hasattr(self, "_m_negative_weights"):
            return self._m_negative_weights

        # if not already done, run the *train_weights* method that also initializes the m_negative_weights
        self.train_weights
        return self._m_negative_weights

    @property
    def train_weights(self) -> np.ndarray:
        if hasattr(self, "_train_weights"):
            return self._train_weights

        train_weights = self.load_data("train_weights")
        self._m_negative_weights = (train_weights < 0)

        # handling of negative weights based on the ml_model_inst.negative_weights parameter
        if self._ml_model_inst.negative_weights == "ignore":
            train_weights[self._m_negative_weights] = 0
        elif self._ml_model_inst.negative_weights == "abs":
            train_weights = np.abs(train_weights)
        elif self._ml_model_inst.negative_weights == "handle":
            train_weights[self._m_negative_weights] = (
                np.abs(train_weights[self._m_negative_weights]) / (len(self._ml_model_inst.processes) - 1)
            )
        elif self._ml_model_inst.negative_weights == "nothing":
            train_weights = train_weights

        self._train_weights = train_weights
        return self._train_weights

    @property
    def equal_weights(self) -> np.ndarray:
        if hasattr(self, "_equal_weights"):
            return self._equal_weights

        self._equal_weights = self.load_data("equal_weights")
        return self._equal_weights

    @property
    def target(self) -> np.ndarray:
        if hasattr(self, "_target"):
            return self._target

        # use the labels to create the target array
        labels = self.labels
        target = np.eye(len(self._ml_model_inst.train_nodes.keys()))[labels]

        # set target to 0 when labels < 0
        target = np.where(labels[:, np.newaxis] >= 0, target, 0)

        # handling of negative weights based on the ml_model_inst.negative_weights parameter
        if self._ml_model_inst.negative_weights == "handle":
            target[self.m_negative_weights] = 1 - target[self.m_negative_weights]

        # NOTE: I think here the targets are somehow 64floats... Maybe check that
        self._target = target
        return self._target

    @property
    def labels(self) -> np.ndarray:
        if hasattr(self, "_labels"):
            return self._labels

        self._labels = self.load_data("labels")
        return self._labels

    @property
    def prediction(self) -> np.ndarray:
        if hasattr(self, "_prediction"):
            return self._prediction

        if "prediction" in self._input.keys():
            # load prediction if possible
            self._prediction = self.load_data("prediction")
        else:
            # calcluate prediction if needed
            if not hasattr(self._ml_model_inst, "best_model"):
                # if not hasattr(self._ml_model_inst, "trained_model"):
                raise Exception("No trained model found in the MLModel instance. Cannot calculate prediction.")
            # self._prediction = predict_numpy_on_batch(self._ml_model_inst.trained_model, self.features)
            self._prediction = predict_numpy_on_batch(self._ml_model_inst.best_model, self.features)

        return self._prediction  # TODO ML best model
