# coding: utf-8

"""
Mixin classes to build ML models
Taken from hbw analysis.
"""

from __future__ import annotations

import functools
import math

import law
# import order as od

from columnflow.types import Union
from columnflow.util import maybe_import, DotDict
from mtt.util import log_memory, call_func_safe, timeit


np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


def loop_dataset(data, max_count=10000):
    for i, x in enumerate(data):
        if i % int(max_count / 100) == 0:
            print(i)
        if i == max_count:
            break


class DenseModelMixin(object):
    """
    Mixin that provides an implementation for `prepare_ml_model`
    """

    _default__activation: str = "relu"
    _default__layers: tuple[int] = (64, 64, 64)
    _default__dropout: float = 0.50
    _default__learningrate: float = 0.00050

    # TODO: these parameters are currently not part of the MLModel repr
    loss: str = "categorical_crossentropy"
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0

    def cast_ml_param_values(self):
        """
        Cast the values of the parameters to the correct types
        """
        super().cast_ml_param_values()
        self.activation = str(self.activation)
        self.layers = tuple(int(n_nodes) for n_nodes in self.layers)
        self.dropout = float(self.dropout)
        self.learningrate = float(self.learningrate)

        self.loss = str(self.loss)
        self.focal_loss_alpha = float(self.focal_loss_alpha)
        self.focal_loss_gamma = float(self.focal_loss_gamma)

    def prepare_ml_model(
        self,
        task: law.Task,
    ):
        import tensorflow.keras as keras
        from keras.models import Sequential
        from keras.layers import Dense, BatchNormalization
        from mtt.ml.tf_util import cumulated_crossentropy
        # from keras.losses import CategoricalFocalCrossentropy

        n_inputs = len(set(self.input_features))
        # n_outputs = len(self.processes)
        n_outputs = len(self.train_nodes.keys())

        # define the DNN model
        model = Sequential()

        # BatchNormalization layer with input shape
        model.add(BatchNormalization(input_shape=(n_inputs,)))

        activation_settings = DotDict({
            "elu": ("elu", "he_uniform", "Dropout"),
            "relu": ("relu", "he_uniform", "Dropout"),
            # "prelu": ("PReLU", "he_normal", "Dropout"),
            "selu": ("selu", "lecun_normal", "AlphaDropout"),
            "tanh": ("tanh", "glorot_normal", "Dropout"),
            "softmax": ("softmax", "glorot_normal", "Dropout"),
        })
        keras_act_name, init_name, dropout_layer = activation_settings[self.activation]

        # hidden layers
        for n_nodes in self.layers:
            model.add(Dense(
                units=n_nodes,
                activation=keras_act_name,
            ))

            # Potentially add dropout layer after each hidden layer
            if self.dropout:
                Dropout = getattr(keras.layers, dropout_layer)
                model.add(Dropout(self.dropout))

        # output layer
        model.add(Dense(n_outputs, activation="softmax"))

        # compile the network
        optimizer = keras.optimizers.Adam(
            learning_rate=self.learningrate, beta_1=0.9, beta_2=0.999,
            epsilon=1e-6, amsgrad=False,
        )

        model_compile_kwargs = {
            "loss": "categorical_crossentropy" if self.negative_weights == "ignore" else cumulated_crossentropy,
            "optimizer": optimizer,
            "metrics": ["categorical_accuracy"],
            "weighted_metrics": ["categorical_accuracy"],
        }
        if self.loss == "focal_loss":
            from keras.losses import CategoricalFocalCrossentropy
            model_compile_kwargs["loss"] = CategoricalFocalCrossentropy(
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
            )
        model.compile(**model_compile_kwargs)

        return model


class CallbacksBase(object):
    """ Base class that handles parametrization of callbacks """
    _default__callbacks: set[str] = {
        "backup", "checkpoint", "reduce_lr",
        # "early_stopping",
    }
    remove_backup: bool = True

    # NOTE: we could remove these parameters since they can be implemented via reduce_lr_kwargs
    _default__reduce_lr_factor: float = 0.8
    _default__reduce_lr_patience: int = 3
    _default__reduce_lr_min_delta: float = 0.0
    _default__reduce_lr_mode: str = "auto"
    _default__reduce_lr_monitor: str = "val_loss"

    # FIXME: for some reasons, these default parameters are not picked up from the derived class
    #        parameters can be set via early_stopping_kwargs though
    _default__early_stopping_monitor: str = "val_loss"
    _default__early_stopping_min_delta: float = 0.0

    # custom callback kwargs
    checkpoint_kwargs: dict = {}
    backup_kwargs: dict = {}
    early_stopping_kwargs: dict = {}
    reduce_lr_kwargs: dict = {}

    def cast_ml_param_values(self):
        """
        Cast the values of the parameters to the correct types
        """
        super().cast_ml_param_values()
        self.callbacks = set(self.callbacks)
        self.remove_backup = bool(self.remove_backup)
        self.reduce_lr_factor = float(self.reduce_lr_factor)
        self.reduce_lr_patience = int(self.reduce_lr_patience)
        self.reduce_lr_min_delta = float(self.reduce_lr_min_delta)
        self.reduce_lr_mode = str(self.reduce_lr_mode)
        self.reduce_lr_monitor = str(self.reduce_lr_monitor)

        self.early_stopping_monitor = str(self.early_stopping_monitor)
        self.early_stopping_min_delta = float(self.early_stopping_min_delta)

    def get_callbacks(self, output):
        import tensorflow.keras as keras
        # check that only valid options have been requested
        callback_options = {"backup", "checkpoint", "reduce_lr", "early_stopping"}
        if diff := self.callbacks.difference(callback_options):
            logger.warning(f"Callbacks '{diff}' have been requested but are not properly implemented")

        # list of callbacks to be returned at the end
        callbacks = []

        # output used for BackupAndRestore callback (not deleted by --remove-output)
        # NOTE: does that work when running remote?
        # TODO: we should also save the parameters + input_features in the backup to ensure that they
        #       are equivalent (delete backup if not)
        backup_output = output["mlmodel"].sibling(f"backup_{output['mlmodel'].basename}", type="d")
        if self.remove_backup:
            backup_output.remove()

        #
        # for each requested callback, merge default kwargs with custom callback kwargs
        #

        if "backup" in self.callbacks:
            backup_kwargs = dict(
                backup_dir=backup_output.abspath,
            )
            backup_kwargs.update(self.backup_kwargs)
            callbacks.append(keras.callbacks.BackupAndRestore(**backup_kwargs))

        if "checkpoint" in self.callbacks:
            checkpoint_kwargs = dict(
                filepath=output["checkpoint"].abspath,
                save_weights_only=False,
                monitor="val_loss",
                mode="auto",
                save_best_only=True,
            )
            checkpoint_kwargs.update(self.checkpoint_kwargs)
            callbacks.append(keras.callbacks.ModelCheckpoint(**checkpoint_kwargs))

        if "early_stopping" in self.callbacks:
            early_stopping_kwargs = dict(
                monitor=self.early_stopping_monitor,
                min_delta=self.early_stopping_min_delta,
                patience=max(min(50, int(self.epochs / 5)), 10),
                verbose=1,
                restore_best_weights=True,
                start_from_epoch=max(min(50, int(self.epochs / 5)), 10),
            )
            early_stopping_kwargs.update(self.early_stopping_kwargs)
            callbacks.append(keras.callbacks.EarlyStopping(**early_stopping_kwargs))

        if "reduce_lr" in self.callbacks:
            reduce_lr_kwargs = dict(
                monitor=self.reduce_lr_monitor,
                factor=self.reduce_lr_factor,
                patience=self.reduce_lr_patience,
                verbose=1,
                mode=self.reduce_lr_mode,
                min_delta=self.reduce_lr_min_delta,
                min_lr=0,
            )
            reduce_lr_kwargs.update(self.reduce_lr_kwargs)
            callbacks.append(keras.callbacks.ReduceLROnPlateau(**reduce_lr_kwargs))

        if len(callbacks) != len(self.callbacks):
            logger.warning(
                f"{len(self.callbacks)} callbacks have been requested but only {len(callbacks)} are returned",
            )

        return callbacks


class ClassicModelFitMixin(CallbacksBase):
    """
    Mixin to run ML Training with "classic" training loop.
    TODO: this will require a different reweighting
    """

    _default__callbacks: set = {
        "backup", "checkpoint", "reduce_lr",
        # "early_stopping",
    }
    remove_backup: bool = True
    _default__reduce_lr_factor: float = 0.8
    _default__reduce_lr_patience: int = 3
    _default__epochs: int = 200
    _default__batchsize: int = 2 ** 12

    def cast_ml_param_values(self):
        """
        Cast the values of the parameters to the correct types
        """
        super().cast_ml_param_values()
        self.epochs = int(self.epochs)
        self.batchsize = int(self.batchsize)

    def fit_ml_model(
        self,
        task: law.Task,
        model,
        train: DotDict[np.array],
        validation: DotDict[np.array],
        output,
    ) -> None:
        """
        Training loop with normal tf dataset
        """
        import tensorflow as tf

        log_memory("start")

        tf_train = tf.data.Dataset.from_tensor_slices(
            (train["inputs"], train["target"], train["weights"]),
        ).batch(self.batchsize).prefetch(tf.data.AUTOTUNE)
        tf_validation = tf.data.Dataset.from_tensor_slices(
            (validation["inputs"], validation["target"], validation["weights"]),
        ).batch(self.batchsize).prefetch(tf.data.AUTOTUNE)

        log_memory("init")

        # set the kwargs used for training
        model_fit_kwargs = {
            "validation_data": tf_validation,
            "epochs": self.epochs,
            "verbose": 2,
            "callbacks": self.get_callbacks(output),
        }

        logger.info("Starting training...")
        tf.debugging.set_log_device_placement(True)
        model.fit(
            tf_train,
            **model_fit_kwargs,
        )
        log_memory("loop")

        # delete tf datasets to clear memory
        del tf_train
        del tf_validation
        log_memory("del")


class ModelFitMixin(CallbacksBase):
    # parameters related to callbacks
    _default__callbacks: set = {
        "backup", "checkpoint", "reduce_lr",
        # "early_stopping",
    }
    remove_backup: bool = True
    _default__reduce_lr_factor: float = 0.8
    _default__reduce_lr_patience: int = 3

    _default__epochs: int = 200
    _default__batchsize: int = 2 ** 12
    # either set steps directly or use attribute from the MultiDataset
    steps_per_epoch: Union[int, str] = "iter_smallest_process"

    def cast_ml_param_values(self):
        """
        Cast the values of the parameters to the correct types
        """
        super().cast_ml_param_values()
        self.epochs = int(self.epochs)
        self.batchsize = int(self.batchsize)
        if isinstance(self.steps_per_epoch, float):
            self.steps_per_epoch = int(self.steps_per_epoch)
        else:
            self.steps_per_epoch = str(self.steps_per_epoch)

    def resolve_weights_xsec(self, data, max_diff_int: float = 0.3):
        """
        Represents cross-section weighting
        """
        rel_sumw_dict = {proc_inst: {} for proc_inst in data.keys()}
        factor = 1
        smallest_sumw = None
        for proc_inst, arrays in data.items():
            sumw = np.sum(arrays.weights) * proc_inst.x.sub_process_class_factor
            if not smallest_sumw or smallest_sumw >= sumw:
                smallest_sumw = sumw

        for proc_inst, arrays in data.items():
            sumw = np.sum(arrays.weights) * proc_inst.x.sub_process_class_factor
            rel_sumw = sumw / smallest_sumw
            rel_sumw_dict[proc_inst] = rel_sumw

            if (rel_sumw - round(rel_sumw)) / rel_sumw > max_diff_int:
                factor = 2

        rel_sumw_dict = {proc_inst: int(rel_sumw * factor) for proc_inst, rel_sumw in rel_sumw_dict.items()}

        return rel_sumw_dict

    def get_batch_sizes(self, data, round: str = "down"):
        batch_sizes = {}
        rel_sumw_dicts = {}
        for train_node_proc_name, node_config in self.train_nodes.items():
            train_node_proc = self.config_inst.get_process(train_node_proc_name)
            sub_procs = (
                (set(node_config.get("sub_processes", set())) | {train_node_proc_name}) &
                {proc.name for proc in data.keys()}
            )
            if not sub_procs:
                raise ValueError(f"Cannot find any sub-processes for {train_node_proc_name} in the data")
            sub_procs = {self.config_inst.get_process(proc_name) for proc_name in sub_procs}
            class_factor_mode = train_node_proc.x("class_factor_mode", "equal")
            if class_factor_mode == "xsec":
                rel_sumw_dicts[train_node_proc.name] = self.resolve_weights_xsec(
                    {proc_inst: data[proc_inst] for proc_inst in sub_procs},
                )
            elif class_factor_mode == "equal":
                rel_sumw_dicts[train_node_proc.name] = {
                    proc_inst: proc_inst.x("sub_process_class_factor", 1) for proc_inst in sub_procs
                }
        for train_node_proc_name, node_config in self.train_nodes.items():
            train_node_proc = self.config_inst.get_process(train_node_proc_name)
            rel_sumw = rel_sumw_dicts[train_node_proc.name]
            rel_sumw_dicts[train_node_proc.name]["sum"] = sum([rel_sumw[proc_inst] for proc_inst in rel_sumw.keys()])
            rel_sumw_dicts[train_node_proc.name]["min"] = min([rel_sumw[proc_inst] for proc_inst in rel_sumw.keys()])

        lcm_list = lambda numbers: functools.reduce(math.lcm, numbers)
        lcm = lcm_list([rel_sumw["sum"] for rel_sumw in rel_sumw_dicts.values()])

        for train_node_proc_name, node_config in self.train_nodes.items():
            train_node_proc = self.config_inst.get_process(train_node_proc_name)
            class_factor = self.class_factors.get(train_node_proc.name, 1)
            rel_sumw_dict = rel_sumw_dicts[train_node_proc.name]
            batch_factor = class_factor * lcm // rel_sumw_dict["sum"]
            if not isinstance(batch_factor, int):
                raise ValueError(
                    f"Batch factor {batch_factor} is not an integer. "
                    "This is likely due to a non-integer class factor.",
                )

            for proc_inst, rel_sumw in rel_sumw_dict.items():
                if isinstance(proc_inst, str):
                    continue
                batch_sizes[proc_inst] = rel_sumw * batch_factor

        # if we requested a batchsize, scale the batch sizes to the requested batchsize, rounding up or down
        if not self.batchsize:
            return batch_sizes
        elif round == "down":
            batch_scaler = self.batchsize // sum(batch_sizes.values()) or 1
        elif round == "up":
            batch_scaler = math.ceil(self.batchsize / sum(batch_sizes.values()))
        else:
            raise ValueError(f"Unknown round option {round}")

        batch_sizes = {proc_inst: int(batch_size * batch_scaler) for proc_inst, batch_size in batch_sizes.items()}
        return batch_sizes

    # def set_validation_weights(self, validation, batch_sizes, train_steps_per_epoch):
    #     """
    #     FIXME: Update the train weights such that??
    #     taken from hbw analysis but seems odd
    #     """
    #     for proc_inst, arrays in validation.items():
    #         bs = batch_sizes[proc_inst]
    #         arrays.validation_weights = arrays.weights / np.sum(arrays.weights) * bs * train_steps_per_epoch

    def set_validation_weights(self, validation, batch_sizes, train_steps_per_epoch):
        """Use consistent weights between training and validation"""
        for proc_inst, arrays in validation.items():
            # keep original weights - no artificial scaling
            arrays.validation_weights = arrays.train_weights
            logger.debug(f"Validation weights for {proc_inst.name}: sum={np.sum(arrays.validation_weights):.2e}")

    def _check_weights(self, train):
        sum_nodes = np.zeros(len(self.train_nodes), dtype=np.float32)
        for proc, data in train.items():
            sum_nodes += np.bincount(data.labels, weights=data.train_weights, minlength=len(self.train_nodes))
            logger.info(f"Sum of weights for process {proc}: {np.sum(data.train_weights)}")

        for proc, node_config in self.train_nodes.items():
            logger.info(f"Sum of weights for train node process {proc}: {sum_nodes[node_config['ml_id']]}")

    @timeit
    def fit_ml_model(
        self,
        task: law.Task,
        model,
        train: DotDict[np.array],
        validation: DotDict[np.array],
        output,
    ) -> None:
        """
        Training loop but with custom dataset
        """
        # import tensorflow as tf
        from mtt.ml.tf_util import MultiDataset
        from mtt.ml.plotting import plot_history

        log_memory("start")

        batch_sizes = self.get_batch_sizes(data=train)
        logger.debug(f"batch_sizes: {batch_sizes}")

        # Add debugging for batch calculations
        for proc_inst, batch_size in batch_sizes.items():
            proc_events = len(train[proc_inst].train_weights)
            steps_for_this_process = proc_events // batch_size if batch_size > 0 else 0
            logger.debug(f"Process {proc_inst.name}: {proc_events} events ÷ {batch_size} batch_size = {steps_for_this_process} steps")  # noqa: E501

        # Create MultiDataset
        tf_train = MultiDataset(data=train, batch_size=batch_sizes, kind="train", buffersize=0)
        log_memory("tf_train")

        # cleanup memory (TODO: seems to not do much if anything)
        for key, ml_dataset in train.items():
            ml_dataset.cleanup()
        log_memory("train cleanup")

        # # determine the requested steps_per_epoch
        # # taken from hbw analysis but seems odd
        # if isinstance(self.steps_per_epoch, str):
        #     magic_smooth_factor = 1
        #     # steps_per_epoch is usually "iter_smallest_process" (TODO: check performance with other factors)
        #     steps_per_epoch = getattr(tf_train, self.steps_per_epoch) * magic_smooth_factor
        # else:
        #     raise Exception("self.steps_per_epoch is not a string, cannot determine steps_per_epoch")
        # if not isinstance(steps_per_epoch, int):
        #     raise Exception(
        #         f"steps_per_epoch is {self.steps_per_epoch} but has to be either an integer or"
        #         "a string corresponding to an integer attribute of the MultiDataset",
        #     )
        # logger.info(f"Training will be done with {steps_per_epoch} steps per epoch")

        # don't use iter_smallest_process - use a reasonable multiplier
        if isinstance(self.steps_per_epoch, str):
            base_steps = getattr(tf_train, self.steps_per_epoch)

            # Calculate what we actually need to see all data reasonably
            total_events = sum(len(data.train_weights) for data in train.values())
            total_batch_size = sum(batch_sizes.values())
            steps_to_see_all_data = total_events // total_batch_size

            # Use a reasonable fraction of all data per epoch
            steps_per_epoch = max(
                steps_to_see_all_data // 10,  # See 1/10th of all data per epoch
                base_steps * 50,              # Or 50x the smallest process
                200                           # Minimum 200 steps per epoch
            )

            logger.debug(f"Base steps (smallest process): {base_steps}")
            logger.debug(f"Steps to see all data: {steps_to_see_all_data}")
            logger.debug(f"Using steps per epoch: {steps_per_epoch}")

        else:
            steps_per_epoch = self.steps_per_epoch

        logger.info(f"Training will be done with {steps_per_epoch} steps per epoch")

        # Create validation dataset
        self.set_validation_weights(validation, batch_sizes, steps_per_epoch)
        tf_validation = MultiDataset(data=validation, kind="valid", buffersize=0)
        log_memory("tf_validation")

        for key, ml_dataset in validation.items():
            ml_dataset.cleanup()
        log_memory("validation cleanup")

        # check that the weights are set correctly
        # self._check_weights(train)
        # weight consistency check
        train_weight_sum = sum(np.sum(data.train_weights) for data in train.values())
        val_weight_sum = sum(np.sum(data.validation_weights) for data in validation.values())
        weight_ratio = val_weight_sum / train_weight_sum
        logger.debug(f"Weight ratio (validation/training): {weight_ratio:.2e}")

        if weight_ratio > 10 or weight_ratio < 0.1:
            logger.warning(f"Large weight imbalance detected: {weight_ratio:.2e}")

        # set the kwargs used for training
        model_fit_kwargs = {
            "validation_data": (x for x in tf_validation),
            "validation_steps": tf_validation.iter_smallest_process,
            "epochs": self.epochs,
            "verbose": 2,
            "steps_per_epoch": steps_per_epoch,
            "callbacks": self.get_callbacks(output),
        }
        # start training by iterating over the MultiDataset
        iterator = (x for x in tf_train)

        def debug_weights_and_labels(self, train, validation):
            """Debug function to check weight and label distributions"""

            logger.debug("=== WEIGHT DISTRIBUTION DEBUG ===")

            for split_name, data in [("Training", train), ("Validation", validation)]:
                logger.debug(f"\n{split_name} Data:")

                total_weight = 0
                label_counts = {}

                for proc_name, proc_data in data.items():
                    weight_attr = 'train_weights' if split_name == 'Training' else 'validation_weights'
                    weights = getattr(proc_data, weight_attr)
                    labels = proc_data.labels

                    proc_weight = np.sum(weights)
                    total_weight += proc_weight

                    unique_labels, counts = np.unique(labels, return_counts=True)
                    for label, count in zip(unique_labels, counts):
                        label_counts[label] = label_counts.get(label, 0) + count

                    logger.debug(f"  {proc_name}: weight_sum={proc_weight:.2e}, n_events={len(weights)}")

                logger.debug(f"  Total weight: {total_weight:.2e}")
                logger.debug(f"  Label distribution: {label_counts}")

        debug_weights_and_labels(self, train, validation)

        logger.info("Starting training...")
        # Removed debugger call for performance
        model.fit(
            iterator,
            **model_fit_kwargs,
        )

        # Log normalized losses for comparison
        final_train_loss = model.history.history['loss'][-1]
        final_val_loss = model.history.history['val_loss'][-1]
        logger.info(f"Final losses - Train: {final_train_loss:.4f}, Val: {final_val_loss:.4f}")

        # Explicit cleanup to prevent memory leaks
        try:
            tf_train.cleanup_resources()
            tf_validation.cleanup_resources()
        except AttributeError:
            # Fallback if cleanup_resources doesn't exist
            pass

        # Delete references to datasets
        del tf_train
        del tf_validation
        del iterator

        # create history plots
        for metric, ylabel, yscale in (
            ("loss", "Loss", "log"),
            ("categorical_accuracy", "Accuracy", "linear"),
            ("weighted_categorical_accuracy", "Weighted Accuracy", "linear"),
        ):
            call_func_safe(
                plot_history,
                model.history.history,
                output["plots"],
                metric=metric,
                ylabel=ylabel,
                yscale=yscale,
            )

        # Force garbage collection
        import gc
        gc.collect()

        log_memory("cleanup")
