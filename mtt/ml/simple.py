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
from columnflow.columnar_util import Route, set_ak_column, remove_ak_column
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

    # -- hooks for sepcifying which ArrayFunctions to run for training

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
        with open(f"{target.path}/model_history.pkl", "rb") as f:
            history = pickle.load(f)

        return model, history

    def prepare_inputs(
        self,
        task,
        input,
    ) -> tuple[dict[str, np.array]]:

        # obtain processes from config
        process_insts = [
            self.config_inst.get_process(proc)
            for proc in self.processes
        ]

        proc_n_events = np.array(len(self.processes) * [0])
        proc_custom_weights = np.array(len(self.processes) * [0])
        proc_sum_weights = np.array(len(self.processes) * [0])
        proc_idx = {}  # bookkeeping which process each dataset belongs to

        #
        # determine process of each dataset and count number of events & sum of eventweights for this process
        #

        for dataset, files in input["events"][self.config_inst.name].items():
            dataset_inst = self.config_inst.get_dataset(dataset)

            # check dataset only belongs to one process
            if len(dataset_inst.processes) != 1:
                raise Exception("only 1 process inst is expected for each dataset")

            # TODO: use stats here instead
            mlevents = [
                ak.from_parquet(inp["mlevents"].fn)
                for inp in files
            ]
            n_events = sum(
                len(events)
                for events in mlevents
            )
            sum_weights = sum(
                ak.sum(events.normalization_weight)
                for events in mlevents
            )

            #
            for i, proc in enumerate(process_insts):
                proc_custom_weights[i] = self.proc_custom_weights[proc.name]
                leaf_procs = [
                    p for p, _, _ in self.config_inst.get_process(proc).walk_processes(include_self=True)
                ]
                if dataset_inst.processes.get_first() in leaf_procs:
                    logger.info(f"the dataset *{dataset}* is used for training the *{proc.name}* output node")
                    proc_idx[dataset] = i
                    proc_n_events[i] += n_events
                    proc_sum_weights[i] += sum_weights
                    continue

            # fail if no process was found for dataset
            if proc_idx.get(dataset) is None:
                raise Exception(f"dataset {dataset} is not matched to any of the given processes")

        #
        # set inputs, weights and targets for each datset and fold
        #

        # TODO

        DNN_inputs = {
            "weights": None,
            "inputs": None,
            "target": None,
        }

        # scaler for weights such that the largest are of order 1
        weights_scaler = min(proc_n_events / proc_custom_weights)

        sum_nnweights_processes = {}
        for dataset, files in input["events"][self.config_inst.name].items():
            this_proc_idx = proc_idx[dataset]

            this_proc_name = self.processes[this_proc_idx]
            this_proc_n_events = proc_n_events[this_proc_idx]
            this_proc_sum_weights = proc_sum_weights[this_proc_idx]

            logger.info(
                f"dataset: {dataset}, \n"
                f"  #Events: {this_proc_n_events}, \n"
                f"  Sum Eventweights: {this_proc_sum_weights}",
            )
            sum_nnweights = 0

            for inp in files:
                events = ak.from_parquet(inp["mlevents"].path)
                weights = events.normalization_weight
                if self.eqweight:
                    weights = weights * weights_scaler / this_proc_sum_weights
                    custom_procweight = self.proc_custom_weights[this_proc_name]
                    weights = weights * custom_procweight

                weights = ak.to_numpy(weights)

                if np.any(~np.isfinite(weights)):
                    raise Exception(f"Non-finite values found in weights from dataset {dataset}")

                sum_nnweights += sum(weights)
                sum_nnweights_processes.setdefault(this_proc_name, 0)
                sum_nnweights_processes[this_proc_name] += sum(weights)

                # remove columns not used in training
                input_features = events[self.input_features_namespace]
                for var in input_features.fields:
                    if var not in self.input_features:
                        events = remove_ak_column(events, f"{self.input_features_namespace}.{var}")

                # transform events into numpy ndarray
                # TODO: at this point we should save the order of our input variables
                #       to ensure that they will be loaded in the correct order when
                #       doing the evaluation
                events = events[self.input_features_namespace]
                events = ak.to_numpy(events)
                events = events.astype(
                    [(name, np.float32) for name in events.dtype.names],
                    copy=False,
                ).view(np.float32).reshape((-1, len(events.dtype)))

                if np.any(~np.isfinite(events)):
                    raise Exception(f"Non-finite values found in inputs from dataset {dataset}")

                # create the truth values for the output layer
                target = np.zeros((len(events), len(self.processes)))
                target[:, this_proc_idx] = 1

                if np.any(~np.isfinite(target)):
                    raise Exception(f"Non-finite values found in target from dataset {dataset}")

                if DNN_inputs["weights"] is None:
                    DNN_inputs["weights"] = weights
                    DNN_inputs["inputs"] = events
                    DNN_inputs["target"] = target
                else:
                    DNN_inputs["weights"] = np.concatenate([DNN_inputs["weights"], weights])
                    DNN_inputs["inputs"] = np.concatenate([DNN_inputs["inputs"], events])
                    DNN_inputs["target"] = np.concatenate([DNN_inputs["target"], target])

        #
        # shuffle events and split into train and validation part
        #

        inputs_size = sum([arr.size * arr.itemsize for arr in DNN_inputs.values()])
        logger.info(f"inputs size is {inputs_size / 1024**3} GB")

        shuffle_indices = np.array(range(len(DNN_inputs["weights"])))
        np.random.shuffle(shuffle_indices)

        n_validation_events = int(self.validation_fraction * len(DNN_inputs["weights"]))

        train, validation = {}, {}
        for k in DNN_inputs.keys():
            DNN_inputs[k] = DNN_inputs[k][shuffle_indices]

            validation[k] = DNN_inputs[k][:n_validation_events]
            train[k] = DNN_inputs[k][n_validation_events:]

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
        #
        # TF settings
        #

        # run on GPU
        gpus = tf.config.list_physical_devices("GPU")

        # restrict to run only on first GPU
        # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            # GPU already initialized -> print warning and continue
            print(e)
        except IndexError:
            print("No GPUs found. Will use CPU.")

        #
        # prepare input
        #

        # TODO: implement
        train, validation = self.prepare_inputs(task, input)

        # check for non-finite values (inf, nan)
        for key in train.keys():
            if np.any(~np.isfinite(train[key])):
                raise Exception(f"Non-finite values found in training {key}")
            if np.any(~np.isfinite(validation[key])):
                raise Exception(f"Non-finite values found in validation {key}")

        #
        # prepare model
        #

        # TODO: implement
        n_inputs = len(self.input_features)
        n_outputs = len(self.processes)

        # start model definition
        model = keras.Sequential()

        # define input normalization
        model.add(keras.layers.BatchNormalization(input_shape=(n_inputs,)))

        # hidden layers
        for n_nodes in self.layers:
            model.add(keras.layers.Dense(
                units=n_nodes,
                activation="ReLU",
            ))

            # optional dropout after each hidden layer
            if self.dropout:
                model.add(keras.layers.Dropout(self.dropout))

        # output layer
        model.add(keras.layers.Dense(
            n_outputs,
            activation="softmax",
        ))

        # optimizer
        # settings from https://github.com/jabuschh/ZprimeClassifier/blob/8c3a8eee/Training.py#L93  # noqa
        optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9, beta_2=0.999,
            epsilon=1e-6,
            amsgrad=False,
        )

        # compile model
        model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizer,
            weighted_metrics=["categorical_accuracy"],
        )

        #
        # training
        #

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

        # construct TF datasets
        with tf.device("CPU"):
            # training
            tf_train = tf.data.Dataset.from_tensor_slices(
                (train["inputs"], train["target"], train["weights"]),
            ).batch(self.batchsize)

            # validation
            tf_validate = tf.data.Dataset.from_tensor_slices(
                (validation["inputs"], validation["target"], validation["weights"]),
            ).batch(self.batchsize)

        # do training
        model.fit(
            tf_train,
            validation_data=tf_validate,
            epochs=self.epochs,
            callbacks=[early_stopping, lr_reducer],
            verbose=2,
        )

        # save trained model and history
        output.parent.touch()
        model.save(output.path)
        with open(f"{output.path}/model_history.pkl", "wb") as f:
            pickle.dump(model.history.history, f)

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

        # unpack models and history
        models, history = zip(*models)

        # create a copy of the inputs to use for evaluation
        inputs = ak.copy(events)

        # remove columns not used in training
        input_features = inputs[self.input_features_namespace]
        for var in input_features.fields:
            if var not in self.input_features:
                inputs = remove_ak_column(inputs, f"{self.input_features_namespace}.{var}")

        # transform inputs into numpy ndarray
        inputs = inputs[self.input_features_namespace]
        inputs = ak.to_numpy(inputs)
        inputs = inputs.astype(
            [(name, np.float32) for name in inputs.dtype.names],
            copy=False,
        ).view(np.float32).reshape((-1, len(inputs.dtype)))

        # do prediction for all models and all inputs
        predictions = []
        for i, model in enumerate(models):
            prediction = ak.from_numpy(model.predict_on_batch(inputs))
            if len(prediction[0]) != len(self.processes):
                raise Exception("Number of output nodes should be equal to number of processes")
            predictions.append(prediction)

        # choose prediction from model that has not used the test dataset/fold
        #outputs = ak.where(ak.ones_like(predictions[0]), -1, -1)  # noqa
        outputs = ak.ones_like(predictions[0]) * -1
        for i in range(self.folds):
            # reshape mask from N*bool to N*k*bool (TODO: simpler way?)
            idx = ak.to_regular(
                ak.concatenate(
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

        # write output scores to columns
        for i, proc in enumerate(self.processes):
            events = set_ak_column(
                events, f"{self.cls_name}.score_{proc}", outputs[:, i],
            )

        #
        # compute categories
        #

        # ML categorization on top of existing categories
        ml_categories = [cat for cat in self.config_inst.categories if "dnn_" in cat.name]
        ml_proc_to_id = {cat.name.replace("dnn_", ""): cat.id for cat in ml_categories}

        # convenience array with all ML scores
        scores = ak.Array({
            f.replace("score_", ""): events[self.cls_name, f]
            for f in events[self.cls_name].fields if f.startswith("score_")
        })

        # compute max score per event and corresponding category
        ml_category_id = ak.Array(np.zeros(len(events)))
        max_score = ak.Array(np.zeros(len(events)))
        for proc in scores.fields:
            larger_score = (scores[proc] > max_score)
            ml_category_id = ak.where(larger_score, ml_proc_to_id[proc], ml_category_id)
            max_score = ak.where(larger_score, scores[proc], max_score)

        # overwrite `category_ids` to include the ML category
        # --
        # NOTE: this simply adds the ML category ID (an integer) to the ones already present
        # in the inputs (except the inclusive category, which is left as-is). For this to
        # produce the expected results, the ML categories need to follow the naming scheme
        # described under `mtt.config.categories`, which ensures that
        #     `cat_id_with_ml == cat_id_without_ml + cat_id_ml`
        category_ids = ak.where(
            events.category_ids != 0,  # do not split inclusive category into DNN sub-categories
            events.category_ids + ak.values_astype(ml_category_id, np.int32),
            events.category_ids,
        )
        events = set_ak_column(events, "category_ids", category_ids)

        # sanity check that the produced category IDs are all valid leaf categories
        leaf_category_ids = {c.id for c in self.config_inst.get_leaf_categories()}
        present_category_ids = set(ak.ravel(events.category_ids))
        invalid_categories = present_category_ids - leaf_category_ids
        if invalid_categories:
            invalid_categories = ", ".join(sorted(invalid_categories))
            raise RuntimeError(
                f"ML evaluation produced category ids that are not defined as valid "
                f"leaf categories in the config: {invalid_categories}",
            )

        return events


# instantiate ML model
simple_dnn = TTbarSimpleDNN.derive("simple", cls_dict={

    # hyperparameters
    "batchsize": 32768,
    "dropout": 0.5,
    "epochs": 20,  # 500
    "eqweight": True,  # False?
    "folds": 2,
    "layers": [512, 512],
    "learning_rate": 0.0005,
    "validation_fraction": 0.25,

    # custom weights for processes (if applicable)
    "proc_custom_weights": {
        "tt": 1,
        "st": 1,
        "w_lnu": 1,
        "dy_lep": 1,
    },

    # processes used for training
    "processes": [
        "tt",
        "st",
        "w_lnu",
        "dy_lep",
    ],

    # datasets used for training
    "dataset_names": {
        # TTbar
        "tt_sl_powheg",
        "tt_dl_powheg",
        "tt_fh_powheg",
        # SingleTop
        "st_tchannel_t_powheg",
        "st_tchannel_tbar_powheg",
        "st_twchannel_t_powheg",
        "st_twchannel_tbar_powheg",
        "st_schannel_lep_amcatnlo",
        "st_schannel_had_amcatnlo",
        # WJets
        "w_lnu_ht70To100_madgraph",
        "w_lnu_ht100To200_madgraph",
        "w_lnu_ht200To400_madgraph",
        "w_lnu_ht400To600_madgraph",
        "w_lnu_ht600To800_madgraph",
        "w_lnu_ht800To1200_madgraph",
        "w_lnu_ht1200To2500_madgraph",
        "w_lnu_ht2500_madgraph",
        # DY
        "dy_lep_m50_ht70to100_madgraph",
        "dy_lep_m50_ht100to200_madgraph",
        "dy_lep_m50_ht200to400_madgraph",
        "dy_lep_m50_ht400to600_madgraph",
        "dy_lep_m50_ht600to800_madgraph",
        "dy_lep_m50_ht800to1200_madgraph",
        "dy_lep_m50_ht1200to2500_madgraph",
        "dy_lep_m50_ht2500_madgraph",
    },

    # ML inputs
    "input_features": [
        "n_jet",
        "n_fatjet",
    ] + [
        f"jet_{var}_{i + 1}"
        for var in ("energy", "pt", "eta", "phi", "mass", "btag")
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
