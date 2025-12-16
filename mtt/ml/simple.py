# coding: utf-8

"""
Simple DNN for event classification
"""
from __future__ import annotations

import law
import order as od
import pickle

from typing import Any, Sequence

from columnflow.ml import MLModel
from columnflow.util import maybe_import, dev_sandbox
from columnflow.columnar_util import Route, set_ak_column, remove_ak_column, ak_concatenate_safe
from columnflow.tasks.selection import MergeSelectionStatsWrapper

from mtt.config.categories import add_categories_ml

np = maybe_import("numpy")
ak = maybe_import("awkward")
tf = maybe_import("tensorflow")
keras = maybe_import("tensorflow.keras")

logger = law.logger.get_logger(__name__)


class TTbarSimpleDNN(MLModel):

    input_features_namespace = "MLInput"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # prepend namespace to input features
        self.input_columns = {
            f"{self.input_features_namespace}.{name}"
            for name in self.input_features
        }

    # -- methods related to task setup & environment

    def sandbox(self, task: law.Task):
        return dev_sandbox("bash::$CF_BASE/sandboxes/venv_ml_tf.sh")

    def setup(self):
        """Custom setup after config objects are assigned."""
        #
        # variables
        #

        # per-process output scores
        for proc in self.processes:
            var = f"{self.cls_name}.score_{proc}"
            if var not in self.config_inst.variables:
                self.config_inst.add_variable(
                    name=var,
                    null_value=-1,
                    binning=(40, 0., 1.),
                    x_title=f"DNN output score, {self.config_inst.get_process(proc).label}",
                    y_title="Events",
                )

        # # truth label (TODO: implement)
        # var_truth = f"{self.cls_name}.ml_label"
        # if var_truth not in self.config_inst.variables:
        #     self.config_inst.add_variable(
        #         name=var_truth,
        #         null_value=-1,
        #         binning=(len(self.processes) + 1, -1.5, len(self.processes) - 0.5),
        #         x_title="DNN true label",
        #     )

        #
        # categories
        #

        # dynamically add ml categories (only if production categories have been added)
        if (
            not self.config_inst.x("has_categories_ml", False) and
            self.config_inst.x("has_categories_production", False)
        ):
            add_categories_ml(self.config_inst, ml_model_inst=self)
            self.config_inst.x.has_categories_ml = True

    # -- methods related to law tasks/dependencies

    def datasets(self, config_inst: od.Config) -> set[od.Dataset]:
        """
        Required datasets for a certain *config_inst*.
        """
        return {config_inst.get_dataset(dataset_name) for dataset_name in self.dataset_names}

    def requires(self, task: law.Task) -> Any:
        """Optional law tasks whose outputs are required for training."""
        return MergeSelectionStatsWrapper.req(
            task,
            shifts="nominal",
            configs=self.config_inst.name,
            datasets=self.dataset_names,
        )

    def output(self, task: law.Task) -> law.FileSystemDirectoryTarget:
        return task.target(f"mlmodel_f{task.branch}of{self.folds}", dir=True)

    # -- methods related to input/output columns

    def uses(self, config_inst: od.Config) -> set[Route | str]:
        return {"normalization_weight", "category_ids"} | self.input_columns

    def produces(self, config_inst: od.Config) -> set[Route | str]:
        produced = set()

        # one column per process with corresponding output score
        for proc in self.processes:
            produced.add(f"{self.cls_name}.score_{proc}")

        # ids for the resulting categorization
        produced.add("category_ids")

        return produced

    # -- hooks for specifying which ArrayFunctions to run for training

    def training_calibrators(
        self,
        config_inst: od.Config,
        requested_calibrators: Sequence[str],
    ) -> list[str]:
        """Calibrators to use for training."""
        return ["skip_jecunc"]

    def training_selector(
        self,
        config_inst: od.Config,
        requested_selector: str,
    ) -> str:
        """Selector to use for training."""
        return "default"

    def training_producers(
        self,
        config_inst: od.Config,
        requested_producers: Sequence[str],
    ) -> list[str]:
        """Producers to use for training."""
        return ["ml_inputs"]

    # -- methods related to ML procedure itself

    def open_model(self, target: law.LocalDirectoryTarget) -> tuple[keras.models.Model, Any]:
        """Open a trained model from *target*."""

        # load the Keras model from the output directory
        model = keras.models.load_model(target.path)

        # load the pickled model history
        with open(f"{target.path}_model_history.pickle", "rb") as f:
            history = pickle.load(f)

        return model, history

    def prepare_inputs(
        self,
        task,
        input,
    ) -> tuple[dict[str, np.array]]:
        logger.info("Starting input preparation...")

        # obtain processes from config
        process_insts = [
            self.config_inst.get_process(proc)
            for proc in self.processes
        ]
        logger.info(f"Using {len(process_insts)} processes: {[proc.name for proc in process_insts]}")

        proc_n_events = np.array(len(self.processes) * [0])
        proc_custom_weights = np.array(len(self.processes) * [0])
        proc_sum_weights = np.array(len(self.processes) * [0])
        proc_idx = {}  # bookkeeping which process each dataset belongs to

        logger.info("Analyzing datasets and counting events...")

        #
        # determine process of each dataset and count number of events & sum of eventweights for this process
        #

        for dataset, files in input["events"][self.config_inst.name].items():
            logger.info(f"Processing dataset: {dataset} with {len(files)} files")

            dataset_inst = self.config_inst.get_dataset(dataset)

            # check dataset only belongs to one process
            if len(dataset_inst.processes) != 1:
                raise Exception("only 1 process inst is expected for each dataset")

            logger.info(f"Loading parquet files for dataset {dataset}...")
            # TODO: use stats here instead
            mlevents = [
                ak.from_parquet(inp["mlevents"].parent.path)
                for inp in files
            ]
            logger.info(f"Loaded {len(mlevents)} parquet files")

            n_events = sum(
                len(events)
                for events in mlevents
            )
            sum_weights = sum(
                ak.sum(events.normalization_weight)
                for events in mlevents
            )
            logger.info(f"Dataset {dataset}: {n_events} events, sum_weights: {sum_weights:.2f}")

            #
            for i, proc in enumerate(process_insts):
                proc_custom_weights[i] = self.proc_custom_weights[proc.name]
                leaf_procs = [
                    p for p, _, _ in self.config_inst.get_process(proc).walk_processes(include_self=True)
                ]
                if dataset_inst.processes.get_first() in leaf_procs:
                    logger.info(f"Dataset *{dataset}* matched to process *{proc.name}* (index {i})")
                    proc_idx[dataset] = i
                    proc_n_events[i] += n_events
                    proc_sum_weights[i] += sum_weights
                    continue

            # fail if no process was found for dataset
            if proc_idx.get(dataset) is None:
                raise Exception(f"dataset {dataset} is not matched to any of the given processes")

        logger.info("Dataset-process matching completed")
        for i, proc in enumerate(process_insts):
            logger.info(f"Process {proc.name}: {proc_n_events[i]} events, sum_weights: {proc_sum_weights[i]:.2f}, custom_weight: {proc_custom_weights[i]}")

        #
        # set inputs, weights and targets for each datset and fold
        #

        logger.info("Preparing DNN inputs...")

        # TODO

        DNN_inputs = {
            "weights": None,
            "inputs": None,
            "target": None,
        }

        # scaler for weights such that the largest are of order 1
        weights_scaler = min(proc_n_events / proc_custom_weights)
        logger.info(f"Weights scaler: {weights_scaler:.6f}")

        sum_nnweights_processes = {}

        logger.info("Processing files for neural network input...")
        for dataset, files in input["events"][self.config_inst.name].items():
            this_proc_idx = proc_idx[dataset]
            this_proc_name = self.processes[this_proc_idx]
            this_proc_n_events = proc_n_events[this_proc_idx]
            this_proc_sum_weights = proc_sum_weights[this_proc_idx]

            logger.info(
                f"Processing dataset: {dataset} for process {this_proc_name}\n"
                f"  Process index: {this_proc_idx}\n"
                f"  #Events: {this_proc_n_events}\n"
                f"  Sum Eventweights: {this_proc_sum_weights:.2f}"
            )

            sum_nnweights = 0

            for file_idx, inp in enumerate(files):
                logger.debug(f"  Processing file {file_idx + 1}/{len(files)}: {inp['mlevents'].path}")

                events = ak.from_parquet(inp["mlevents"].path)
                logger.debug(f"  Loaded {len(events)} events from file")

                weights = events.normalization_weight
                if self.eqweight:
                    weights = weights * weights_scaler / this_proc_sum_weights
                    custom_procweight = self.proc_custom_weights[this_proc_name]
                    weights = weights * custom_procweight
                    logger.debug(f"  Applied equal weighting with custom weight {custom_procweight}")

                weights = ak.to_numpy(weights)

                if np.any(~np.isfinite(weights)):
                    raise Exception(f"Non-finite values found in weights from dataset {dataset}, file {file_idx}")

                sum_nnweights += sum(weights)
                sum_nnweights_processes.setdefault(this_proc_name, 0)
                sum_nnweights_processes[this_proc_name] += sum(weights)

                logger.debug(f"  Weights: min={np.min(weights):.6f}, max={np.max(weights):.6f}, sum={np.sum(weights):.2f}")

                # remove columns not used in training
                input_features = events[self.input_features_namespace]
                logger.debug(f"  Available input features: {list(input_features.fields)}")

                for var in input_features.fields:
                    if var not in self.input_features:
                        events = remove_ak_column(events, f"{self.input_features_namespace}.{var}")
                        logger.debug(f"  Removed unused feature: {var}")

                # transform events into numpy ndarray
                # TODO: at this point we should save the order of our input variables
                #       to ensure that they will be loaded in the correct order when
                #       doing the evaluation
                events = events[self.input_features_namespace]
                logger.debug(f"  Using features: {list(events.fields)}")

                events = ak.to_numpy(events)
                events = events.astype(
                    [(name, np.float32) for name in events.dtype.names],
                    copy=False,
                ).view(np.float32).reshape((-1, len(events.dtype)))

                if np.any(~np.isfinite(events)):
                    raise Exception(f"Non-finite values found in inputs from dataset {dataset}, file {file_idx}")

                logger.debug(f"  Input array shape: {events.shape}")

                # create the truth values for the output layer
                target = np.zeros((len(events), len(self.processes)))
                target[:, this_proc_idx] = 1
                logger.debug(f"  Target array shape: {target.shape}, process index: {this_proc_idx}")

                if np.any(~np.isfinite(target)):
                    raise Exception(f"Non-finite values found in target from dataset {dataset}, file {file_idx}")

                if DNN_inputs["weights"] is None:
                    logger.info("  Initializing DNN input arrays")
                    DNN_inputs["weights"] = weights
                    DNN_inputs["inputs"] = events
                    DNN_inputs["target"] = target
                else:
                    logger.debug("  Concatenating to existing DNN input arrays")
                    DNN_inputs["weights"] = np.concatenate([DNN_inputs["weights"], weights])
                    DNN_inputs["inputs"] = np.concatenate([DNN_inputs["inputs"], events])
                    DNN_inputs["target"] = np.concatenate([DNN_inputs["target"], target])

        logger.info("Final NN weights per process:")
        for proc_name, weight_sum in sum_nnweights_processes.items():
            logger.info(f"  {proc_name}: {weight_sum:.2f}")

        #
        # shuffle events and split into train and validation part
        #

        total_events = len(DNN_inputs["weights"])
        logger.info(f"Total events for training: {total_events}")

        inputs_size = sum([arr.size * arr.itemsize for arr in DNN_inputs.values()])
        logger.info(f"Total input size: {inputs_size / 1024**3:.2f} GB")

        logger.info("Shuffling events...")
        shuffle_indices = np.array(range(len(DNN_inputs["weights"])))
        np.random.shuffle(shuffle_indices)

        n_validation_events = int(self.validation_fraction * len(DNN_inputs["weights"]))
        n_training_events = len(DNN_inputs["weights"]) - n_validation_events

        logger.info(f"Splitting into {n_training_events} training and {n_validation_events} validation events")
        logger.info(f"Validation fraction: {self.validation_fraction}")

        train, validation = {}, {}
        for k in DNN_inputs.keys():
            DNN_inputs[k] = DNN_inputs[k][shuffle_indices]

            validation[k] = DNN_inputs[k][:n_validation_events]
            train[k] = DNN_inputs[k][n_validation_events:]

        logger.info("Input preparation completed successfully")
        logger.info(f"Training set shapes: inputs={train['inputs'].shape}, weights={train['weights'].shape}, target={train['target'].shape}")
        logger.info(f"Validation set shapes: inputs={validation['inputs'].shape}, weights={validation['weights'].shape}, target={validation['target'].shape}")

        return train, validation

    def train(
        self,
        task: law.Task,
        input: Any,
        output: law.LocalDirectoryTarget,
    ) -> None:
        """
        Train the model.
        """
        logger.info("=" * 50)
        logger.info("STARTING NEURAL NETWORK TRAINING")
        logger.info("=" * 50)

        #
        # TF settings
        #

        logger.info("Setting up TensorFlow...")

        # run on GPU
        gpus = tf.config.list_physical_devices("GPU")
        logger.info(f"Available GPUs: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            logger.info(f"  GPU {i}: {gpu}")

        # restrict to run only on first GPU
        # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logger.info("GPU memory growth enabled")
        except RuntimeError as e:
            # GPU already initialized -> print warning and continue
            logger.warning(f"GPU memory growth setting failed: {e}")
        except IndexError:
            logger.warning("No GPUs found. Will use CPU.")

        #
        # prepare input
        #

        logger.info("Preparing training inputs...")
        train, validation = self.prepare_inputs(task, input)

        # check for non-finite values (inf, nan)
        logger.info("Checking for non-finite values in training data...")
        for key in train.keys():
            if np.any(~np.isfinite(train[key])):
                raise Exception(f"Non-finite values found in training {key}")
            if np.any(~np.isfinite(validation[key])):
                raise Exception(f"Non-finite values found in validation {key}")
        logger.info("Data validation passed - no non-finite values found")

        #
        # prepare model
        #

        logger.info("Building neural network model...")

        n_inputs = len(self.input_features)
        n_outputs = len(self.processes)

        logger.info("Network architecture:")
        logger.info(f"  Input features: {n_inputs}")
        logger.info(f"  Hidden layers: {self.layers}")
        logger.info(f"  Output nodes: {n_outputs} (processes: {self.processes})")
        logger.info(f"  Dropout rate: {self.dropout}")
        logger.info(f"  Learning rate: {self.learning_rate}")

        # start model definition
        model = keras.Sequential()

        # define input normalization
        model.add(keras.layers.BatchNormalization(input_shape=(n_inputs,)))
        logger.info("  Added batch normalization layer")

        # hidden layers
        for i, n_nodes in enumerate(self.layers):
            model.add(keras.layers.Dense(
                units=n_nodes,
                activation="relu",
            ))
            logger.info(f"  Added dense layer {i+1}: {n_nodes} nodes")

            # optional dropout after each hidden layer
            if self.dropout:
                model.add(keras.layers.Dropout(self.dropout))
                logger.info(f"  Added dropout layer: {self.dropout}")

        # output layer
        model.add(keras.layers.Dense(
            n_outputs,
            activation="softmax",
        ))
        logger.info(f"  Added output layer: {n_outputs} nodes with softmax activation")

        # optimizer
        # settings from https://github.com/jabuschh/ZprimeClassifier/blob/8c3a8eee/Training.py#L93  # noqa
        optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9, beta_2=0.999,
            epsilon=1e-6,
            amsgrad=False,
        )
        logger.info("Created Adam optimizer")

        # compile model
        model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizer,
            weighted_metrics=["categorical_accuracy"],
        )
        logger.info("Model compiled successfully")

        # Print model summary
        logger.info("Model summary:")
        model.summary(print_fn=lambda x: logger.info(x))

        #
        # training
        #

        logger.info("Setting up training callbacks...")

        # early stopping criteria
        early_stopping = keras.callbacks.EarlyStopping(
            # stop when validation loss no longer improves
            monitor="val_loss",
            mode="min",
            # minimum change to consider as improvement
            min_delta=0.005,
            # wait this many epochs w/o improvement before stopping
            patience=max(1, int(self.epochs / 4)),  # 100
            # start monitoring from the beginning
            start_from_epoch=0,
            verbose=0,
            restore_best_weights=True,
        )
        logger.info(f"  Early stopping: patience={max(1, int(self.epochs / 4))}, min_delta=0.005")

        # learning rate reduction on plateau
        lr_reducer = keras.callbacks.ReduceLROnPlateau(
            # reduce LR when validation loss stops improving
            monitor="val_loss",
            mode="min",
            # minimum change to consider as improvement
            min_delta=0.001,
            # factor by which the learning rate will be reduced
            factor=0.5,
            # wait this many epochs w/o improvement before reducing LR
            patience=max(1, int(self.epochs / 8)),  # 100
        )
        logger.info(f"  LR reduction: patience={max(1, int(self.epochs / 8))}, factor=0.5")

        logger.info("Creating TensorFlow datasets...")
        # construct TF datasets
        with tf.device("CPU"):
            # training
            tf_train = tf.data.Dataset.from_tensor_slices(
                (train["inputs"], train["target"], train["weights"]),
            ).batch(self.batchsize)
            logger.info(f"  Training dataset: batch size {self.batchsize}")

            # validation
            tf_validate = tf.data.Dataset.from_tensor_slices(
                (validation["inputs"], validation["target"], validation["weights"]),
            ).batch(self.batchsize)
            logger.info(f"  Validation dataset: batch size {self.batchsize}")

        # do training
        logger.info("=" * 30)
        logger.info("STARTING TRAINING")
        logger.info("=" * 30)
        logger.info(f"Training for {self.epochs} epochs...")

        model.fit(
            tf_train,
            validation_data=tf_validate,
            epochs=self.epochs,
            callbacks=[early_stopping, lr_reducer],
            verbose=2,
        )

        logger.info("Training completed")

        # save trained model and history
        logger.info(f"Saving model to {output.path}")
        output.parent.touch()
        model.save(f"{output.path}.keras")

        logger.info("Saving training history")
        with open(f"{output.path}_model_history.pickle", "wb") as f:
            pickle.dump(model.history.history, f)

        logger.info("Model and history saved successfully")
        logger.info("=" * 50)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 50)

    def evaluate(
        self,
        task: law.Task,
        events: ak.Array,
        models: list[Any],
        fold_indices: ak.Array,
        events_used_in_training: bool = False,
    ) -> ak.Array:
        """
        Evaluate the model on *events* and return them.
        """
        logger.info("=" * 50)
        logger.info("STARTING MODEL EVALUATION")
        logger.info("=" * 50)
        
        logger.info(f"Evaluating on {len(events)} events")
        logger.info(f"Number of models (folds): {len(models)}")
        logger.info(f"Events used in training: {events_used_in_training}")

        # unpack models and history
        models, history = zip(*models)
        logger.info("Models and history unpacked")

        # create a copy of the inputs to use for evaluation
        logger.info("Creating copy of input events...")
        inputs = ak.copy(events)

        # remove columns not used in training
        logger.info("Removing unused input features...")
        input_features = inputs[self.input_features_namespace]
        logger.info(f"Available features: {list(input_features.fields)}")
        
        for var in input_features.fields:
            if var not in self.input_features:
                inputs = remove_ak_column(inputs, f"{self.input_features_namespace}.{var}")
                logger.debug(f"  Removed unused feature: {var}")

        logger.info(f"Using {len(self.input_features)} features: {self.input_features}")

        # transform inputs into numpy ndarray
        logger.info("Converting inputs to numpy array...")
        inputs = inputs[self.input_features_namespace]
        inputs = ak.to_numpy(inputs)
        inputs = inputs.astype(
            [(name, np.float32) for name in inputs.dtype.names],
            copy=False,
        ).view(np.float32).reshape((-1, len(inputs.dtype)))

        logger.info(f"Input array shape: {inputs.shape}")

        # do prediction for all models and all inputs
        logger.info("Running predictions with all models...")
        predictions = []
        for i, model in enumerate(models):
            logger.info(f"  Running prediction with model {i+1}/{len(models)}")
            prediction = ak.from_numpy(model.predict_on_batch(inputs))
            
            if len(prediction[0]) != len(self.processes):
                raise Exception("Number of output nodes should be equal to number of processes")
                
            logger.info(f"  Model {i+1} prediction shape: {prediction.layout.form}")
            predictions.append(prediction)

        logger.info("All model predictions completed")

        # choose prediction from model that has not used the test dataset/fold
        logger.info("Selecting predictions based on fold indices...")
        outputs = ak.ones_like(predictions[0]) * -1
        
        for i in range(self.folds):
            fold_mask = fold_indices == i
            n_events_in_fold = ak.sum(fold_mask)
            logger.info(f"  Fold {i}: {n_events_in_fold} events")
            
            # reshape mask from N*bool to N*k*bool (TODO: simpler way?)
            idx = ak.to_regular(
                ak_concatenate_safe(
                    [
                        ak.singletons(fold_indices == i),
                    ] * len(self.processes),
                    axis=1,
                ),
            )
            outputs = ak.where(idx, predictions[i], outputs)

        # sanity check of output dimensions
        if len(outputs[0]) != len(self.processes):
            raise Exception(
                f"Number of output nodes ({len(outputs[0])}) should be "
                f"equal to number of processes ({len(self.processes)}).",
            )

        logger.info(f"Final output shape: {outputs.layout.form}")

        # write output scores to columns
        logger.info("Writing output scores to event columns...")
        for i, proc in enumerate(self.processes):
            score_values = outputs[:, i]
            logger.info(f"  Process {proc}: score range [{ak.min(score_values):.4f}, {ak.max(score_values):.4f}]")
            events = set_ak_column(
                events, f"{self.cls_name}.score_{proc}", score_values,
            )

        #
        # compute categories
        #

        logger.info("Computing ML-based categories...")
        
        # ML categorization on top of existing categories
        ml_categories = [cat for cat in self.config_inst.categories if "dnn_" in cat.name]
        ml_proc_to_id = {cat.name.replace("dnn_", ""): cat.id for cat in ml_categories}
        
        logger.info(f"Available ML categories: {len(ml_categories)}")
        for cat_name, cat_id in ml_proc_to_id.items():
            logger.info(f"  {cat_name}: category ID {cat_id}")

        # convenience array with all ML scores
        scores = ak.Array({
            f.replace("score_", ""): events[self.cls_name, f]
            for f in events[self.cls_name].fields if f.startswith("score_")
        })
        
        logger.info(f"Score fields: {list(scores.fields)}")

        # compute max score per event and corresponding category
        logger.info("Computing maximum scores and corresponding categories...")
        ml_category_id = ak.Array(np.zeros(len(events)))
        max_score = ak.Array(np.zeros(len(events)))
        
        for proc in scores.fields:
            larger_score = (scores[proc] > max_score)
            n_better = ak.sum(larger_score)
            logger.info(f"  Process {proc}: {n_better} events have this as max score")
            
            ml_category_id = ak.where(larger_score, ml_proc_to_id[proc], ml_category_id)
            max_score = ak.where(larger_score, scores[proc], max_score)

        logger.info(f"Max score range: [{ak.min(max_score):.4f}, {ak.max(max_score):.4f}]")

        # overwrite `category_ids` to include the ML category
        logger.info("Updating category IDs with ML categories...")
        original_categories = ak.unique(events.category_ids)
        logger.info(f"Original category IDs: {original_categories}")
        
        category_ids = ak.where(
            events.category_ids != 0,  # do not split inclusive category into DNN sub-categories
            events.category_ids + ak.values_astype(ml_category_id, np.int32),
            events.category_ids,
        )
        
        new_categories = ak.unique(category_ids)
        logger.info(f"New category IDs: {new_categories}")
        
        events = set_ak_column(events, "category_ids", category_ids)

        # sanity check that the produced category IDs are all valid leaf categories
        logger.info("Validating produced category IDs...")
        leaf_category_ids = {c.id for c in self.config_inst.get_leaf_categories()}
        present_category_ids = set(ak.ravel(events.category_ids))
        invalid_categories = present_category_ids - leaf_category_ids
        
        logger.info(f"Valid leaf category IDs: {sorted(leaf_category_ids)}")
        logger.info(f"Present category IDs: {sorted(present_category_ids)}")
        
        if invalid_categories:
            invalid_categories = ", ".join(sorted(invalid_categories))
            raise RuntimeError(
                f"ML evaluation produced category ids that are not defined as valid "
                f"leaf categories in the config: {invalid_categories}",
            )
            
        logger.info("Category validation passed")
        logger.info("=" * 50)
        logger.info("MODEL EVALUATION COMPLETED")
        logger.info("=" * 50)

        return events


# instantiate ML model
simple_dnn = TTbarSimpleDNN.derive("simple", cls_dict={

    # hyperparameters
    # "batchsize": 32768,
    "batchsize": 4096,  # reasonable for training on CPU?
    "dropout": 0.5,
    "epochs": 20,  # 20/500
    "eqweight": True,  # False?
    "folds": 5,  # 2/5
    "layers": [512, 512],
    "learning_rate": 0.0005,
    "validation_fraction": 0.25,

    # custom weights for processes (if applicable)
    "proc_custom_weights": {
        "tt": 1,
        "st": 1,
        "w_lnu": 1,
        "dy": 1,
    },

    # processes used for training
    "processes": [
        "tt",
        "st",
        "w_lnu",
        "dy",
    ],

    # datasets used for training
    "dataset_names": {
        # # Run 2 datasets
        # # TTbar
        # "tt_sl_powheg",
        # "tt_dl_powheg",
        # "tt_fh_powheg",
        # # SingleTop
        # "st_tchannel_t_4f_powheg",
        # "st_tchannel_tbar_4f_powheg",
        # "st_twchannel_t_powheg",
        # "st_twchannel_tbar_powheg",
        # "st_schannel_lep_4f_amcatnlo",
        # "st_schannel_had_4f_amcatnlo",
        # # WJets
        # "w_lnu_ht70To100_madgraph",
        # "w_lnu_ht100To200_madgraph",
        # "w_lnu_ht200To400_madgraph",
        # "w_lnu_ht400To600_madgraph",
        # "w_lnu_ht600To800_madgraph",
        # "w_lnu_ht800To1200_madgraph",
        # "w_lnu_ht1200To2500_madgraph",
        # "w_lnu_ht2500_madgraph",
        # # DY
        # "dy_lep_m50_ht70to100_madgraph",
        # "dy_lep_m50_ht100to200_madgraph",
        # "dy_lep_m50_ht200to400_madgraph",
        # "dy_lep_m50_ht400to600_madgraph",
        # "dy_lep_m50_ht600to800_madgraph",
        # "dy_lep_m50_ht800to1200_madgraph",
        # "dy_lep_m50_ht1200to2500_madgraph",
        # "dy_lep_m50_ht2500_madgraph",

        # Run 3 datasets
        # TTbar
        "tt_sl_powheg",
        "tt_dl_powheg",
        "tt_fh_powheg",
        # SingleTop
        # "st_tchannel_t_4f_powheg",
        # "st_tchannel_tbar_4f_powheg",
        "st_twchannel_t_sl_powheg",
        "st_twchannel_tbar_sl_powheg",
        "st_twchannel_t_dl_powheg",
        # "st_twchannel_tbar_dl_powheg",
        "st_twchannel_t_fh_powheg",
        "st_twchannel_tbar_fh_powheg",
        # WJets
        "w_lnu_1j_madgraph",
        "w_lnu_2j_madgraph",
        "w_lnu_3j_madgraph",
        "w_lnu_4j_madgraph",
        # DY
        "dy_4j_mumu_m50toinf_madgraph",
        "dy_4j_ee_m50toinf_madgraph",
        # "dy_4j_tautau_m50toinf_madgraph",
        # # VV
        # # to be added?
        # "ww_pythia",
        # "wz_pythia",
        # "zz_pythia",
    },

    # ML inputs
    "input_features": [
        "n_jet",
        "n_fatjet",
    ] + [
        f"jet_{var}_{i + 1}"
        for var in ("energy", "pt", "eta", "phi", "mass", "btagUParTAK4B")
        for i in range(5)
    ] + [
        f"fatjet_{var}_{i + 1}"
        for var in ("energy", "pt", "eta", "phi", "msoftdrop", "tau21", "tau32")
        for i in range(3)
    ] + [
        f"lepton_{var}"
        for var in ("energy", "pt", "eta", "phi")
    ] + [
        f"met_{var}"
        for var in ("pt", "phi")
    ],
})
