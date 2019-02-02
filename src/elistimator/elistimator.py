from typing import Callable, Optional, Tuple, Union, Dict, List, Generator
from pathlib import Path
from tempfile import TemporaryDirectory
from itertools import count
from contextlib import suppress
import pickle
import abc

from tqdm import tqdm
import numpy as np
import tensorflow as tf

# <editor-fold desc="Data types">
ModelFn = Callable[[Dict[Union[str, int], tf.Tensor],  # Features
                    tf.Tensor,  # Training mode (is_training)
                    Optional[dict]],  # Parameters
                   Tuple['TrainSpec', 'ValidationSpec', Optional['PredictSpec']]]
InputFn = Callable[[], tf.data.Dataset]


# </editor-fold>

class ElistimatorSpec(abc.ABC):

    @abc.abstractmethod
    def to_dict(self) -> dict:
        pass

    @staticmethod
    @abc.abstractmethod
    def from_dict(graph: tf.Graph, dictionary: dict) -> 'ElistimatorSpec':
        pass


class TrainSpec(ElistimatorSpec):

    def __init__(self,
                 loss: tf.Tensor,
                 train_op: tf.Operation,
                 create_viz_op: bool = True):
        """

        :param loss: Loss tensor
        :param train_op: Training operation (notice, this is not the optimizer, but the optimization call [ex. optimizer.minimize()])
        :param create_viz_op: Create the visualization op (should be done once)
        """

        self.loss = loss
        self.train_op = train_op

        # Visualize
        if create_viz_op:
            tf.summary.scalar('loss', self.loss)

    def to_dict(self) -> dict:
        return {
            'loss': self.loss.name,
            'train_op': self.train_op.name}

    @staticmethod
    def from_dict(graph: tf.Graph, dictionary: dict) -> 'TrainSpec':
        return TrainSpec(loss=graph.get_tensor_by_name(dictionary['loss']),
                         train_op=graph.get_operation_by_name(dictionary['train_op']),
                         create_viz_op=False)


class EvaluationSpec(ElistimatorSpec):

    def __init__(self,
                 loss: tf.Tensor,
                 eval_metric_ops: Dict[str, Tuple[tf.Tensor, tf.Operation]]):
        """

        :param loss: Loss tensor
        :param eval_metric_ops: Evaluation metric ops, should be a dictionary where the values are a tuple of the metric, and its update op
        """

        self.loss = loss
        self.metric_ops = eval_metric_ops

    def get_running_variables(self):
        running_vars = list()

        for metric_tensor, _ in self.metric_ops.values():
            running_vars.extend(
                tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=self._get_scope(tensor_name=metric_tensor.name)))

        return running_vars

    def to_dict(self):

        # Metric ops
        dictionary = dict()
        dictionary['metric_ops'] = {key: (tensor.name, op.name) for key, (tensor, op) in self.metric_ops.items()}

        # Loss
        dictionary['loss'] = self.loss.name

        return dictionary

    @staticmethod
    def from_dict(graph: tf.Graph, dictionary: dict) -> 'EvaluationSpec':

        # Metric ops
        eval_metric_ops = {key: (graph.get_tensor_by_name(tensor), graph.get_tensor_by_name(op)) for
                           key, (tensor, op) in dictionary['metric_ops'].items()}

        # Loss
        loss = graph.get_tensor_by_name(dictionary['loss'])

        return EvaluationSpec(loss=loss,
                              eval_metric_ops=eval_metric_ops)

    @staticmethod
    def _get_scope(tensor_name: str) -> Optional[str]:
        """
        Gets the scope of a tensor. None of tensor has no scope

        :param tensor_name: Full tensor name
        :return: Isolated scope name (without /)
        """

        split_name = tensor_name.split('/')
        if len(split_name) < 2:
            return None

        return split_name[-2]


class PredictSpec(ElistimatorSpec):

    def __init__(self,
                 output: Dict[str, tf.Tensor]):
        """

        :param output: Network's output
        """

        self.output = output

    def to_dict(self):
        return {key: tensor.name for key, tensor in self.output.items()}

    @staticmethod
    def from_dict(graph: tf.Graph, dictionary: dict) -> 'PredictSpec':
        return PredictSpec(output={key: graph.get_tensor_by_name(tensor) for key, tensor in dictionary.items()})


class Elistimator:
    TQDM_NCOLS = 80
    ELISTIMATOR_META_FILENAME = 'elistimator_meta'

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    @property
    def train_logs(self) -> Path:
        return self._model_dir / 'train'

    @property
    def validation_logs(self) -> Path:
        return self._model_dir / 'validation'

    @property
    def train_checkpoints(self) -> Path:
        return self._model_dir / 'checkpoints'

    @property
    def global_step(self) -> int:
        return self._session.run(tf.train.get_global_step(graph=self._graph))

    @property
    def session(self) -> tf.Session:
        return self._session

    @property
    def graph(self) -> tf.Graph:
        return self._graph

    @property
    def checkpoints(self) -> Dict[int, dict]:
        return self._checkpoints

    def __init__(self,
                 model_fn: ModelFn,
                 model_dir: Optional[str] = None,
                 params: Optional[dict] = None,
                 session_args: Optional[dict] = None):
        """
        Creates a new instance of the Estimator class

        :param model_fn: Model creating function
        :param model_dir: Directory to host all files related to the model
        :param params: Parameters to be available to the model_fn function
        :param session_args: Parameters to pass to the tf.Session
        """

        if not session_args:
            session_args = {}

        self._model_fn = model_fn
        self._model_dir = self._get_or_create_directory(model_dir)
        self._params = params if params is not None else {}

        self._train_size = None
        self._validation_size = None
        self._train_count = 0

        # To be defined with the setup_training method
        self._handle = None
        self._train_handle = None
        self._train_iterator = None
        self._validation_handle = None
        self._validation_iterator = None
        self._train_spec = None
        self._evaluation_spec = None
        self._predict_spec = None
        self._is_training = None
        self._visualization_op = None
        self._running_variables_initializer = None
        self._input_signature = None
        self._train_summary_writer = None
        self._validation_summary_writer = None
        self._graph = tf.Graph()
        self._session = tf.Session(graph=self._graph, **session_args)
        self._last_validation_metrics = None
        self._checkpoints = dict()
        self._global_vars_initialized = False

    def setup(self,
              train_input_fn: InputFn,
              validation_input_fn: Optional[InputFn] = None,
              data_keys: Optional[List[str]] = None,
              train_saver_args: Optional[dict] = None) -> 'Elistimator':
        """
        Setup training flow, should be called only once.

        :param train_input_fn: Training input function
        :param validation_input_fn: Validation input function
        :param data_keys: Keys to match output of the training/validation inputs
        :param train_saver_args: Options to be passed to tf.train.Saver constructor
        :return: self, for chaining
        """

        # Init
        validation_iterator = None
        self._train_size = None
        self._validation_size = None

        # If data_keys not provided, use 0, 1, 2 ...
        data_keys = data_keys if data_keys else count()

        with self._graph.as_default():

            tf.train.get_or_create_global_step()

            with tf.variable_scope('input'):

                # Setup training data pipeline
                train_dataset = train_input_fn()
                train_iterator = train_dataset.make_initializable_iterator()

                if self._is_training is None:
                    self._is_training = tf.placeholder_with_default(False, shape=[], name='is_training')

                if validation_input_fn:
                    # Setup validation data pipeline
                    validation_dataset = validation_input_fn()
                    validation_iterator = validation_dataset.make_initializable_iterator()

                # Iterators handling
                data = {}
                if self._handle is None:
                    output_types = train_dataset.output_types
                    output_shapes = train_dataset.output_shapes
                    handle = tf.placeholder(tf.string, shape=[], name='iterator_handle')
                    iterator = tf.data.Iterator.from_string_handle(handle,
                                                                   output_types=output_types,
                                                                   output_shapes=output_shapes)

                    iterator_output = iterator.get_next()

                    # Determine whether to assign keys or already supplied by dataset
                    if isinstance(iterator_output, dict):
                        data = iterator_output
                    else:
                        data = {data_key: data_tensor
                                for data_key, data_tensor in zip(data_keys, iterator_output)}

            train_spec, validation_spec, predict_spec = self._model_fn(data, self._is_training, self._params)

            # Prepare running variables initializer for metric's running variables
            if validation_spec and self._running_variables_initializer is None:
                self._running_variables_initializer = tf.variables_initializer(
                    var_list=validation_spec.get_running_variables())

            # Create visualization op
            if self._visualization_op is None:
                self._visualization_op = tf.summary.merge_all()

            # Create a train saver
            self._train_saver = tf.train.Saver() if not train_saver_args else tf.train.Saver(**train_saver_args)

        # Create summary logs directories
        if validation_input_fn and not self.train_logs.exists():
            self.train_logs.mkdir(parents=True)
        if validation_input_fn and not self.validation_logs.exists():
            self.validation_logs.mkdir(parents=True)

        # Create checkpoints directory
        if not self.train_checkpoints.exists():
            self.train_checkpoints.mkdir(parents=True)

        # Assign
        self._train_handle = self._session.run(train_iterator.string_handle('train_handle')) if train_iterator else None
        self._train_iterator = train_iterator
        self._validation_handle = self._session.run(validation_iterator.string_handle(
            'validation_handle')) if validation_iterator else None
        self._validation_iterator = validation_iterator

        if self._handle is None:
            self._handle = handle

        self._train_spec = train_spec
        self._evaluation_spec = validation_spec
        self._predict_spec = predict_spec

        if not self._input_signature:
            self._input_signature = {data_key: data_tensor.name for data_key, data_tensor in data.items()}

        self._train_summary_writer = tf.summary.FileWriter(logdir=str(self.train_logs),
                                                           graph=self._graph)
        if validation_input_fn:
            self._validation_summary_writer = tf.summary.FileWriter(logdir=str(self.validation_logs),
                                                                    graph=self._graph)
        # First time initialization
        if not self._global_vars_initialized:
            with self._graph.as_default():
                self._session.run(tf.global_variables_initializer())
                self._global_vars_initialized = True

        self._session.run(self._train_iterator.initializer)

        return self

    def train(self,
              max_steps: Optional[int] = None) -> 'Elistimator':
        """
        Train. Runs for given number of steps or until iterator is exhausted

        :param max_steps: Steps to run for. (None = until iterator is exhausted)
        :return: self, for chaining
        """

        if self._train_spec is None or self._train_handle is None:
            raise RuntimeError('Must call setup with train_input_fn before using the train method')

        self._train_count += 1

        # Preparation
        if not max_steps:
            max_steps = float('inf')
        global_step = tf.train.get_global_step(graph=self._graph)
        session_vars = [self._train_spec.train_op,  # Training op
                        global_step,  # Global step
                        self._train_spec.loss,  # Loss
                        self._visualization_op]  # TB visualization op
        local_step = 0
        train_pbar = tqdm(total=self._train_size,
                          desc='Training ({})'.format(self._train_count),
                          ncols=self.TQDM_NCOLS)

        # Training
        while local_step < max_steps:
            try:

                # Training step
                _, global_step_, loss, visualization = self._session.run(fetches=session_vars,
                                                                         feed_dict={self._handle: self._train_handle,
                                                                                    self._is_training: True})
                # Visualizing
                self._train_summary_writer.add_summary(visualization, global_step=global_step_)

                # Update progress bar
                train_pbar.set_postfix(loss=format(round(loss, 6), '.6f'))
                train_pbar.update()

                local_step += 1

                if local_step >= max_steps:
                    raise StopIteration

            except (tf.errors.OutOfRangeError, StopIteration):

                # Initializer train iterator when needed
                if max_steps == float('inf') or local_step == max_steps:
                    self._session.run(self._train_iterator.initializer)

                # Save train size for next train calls
                self._train_size = local_step

                # Close progress bar
                train_pbar.close()

                break

        return self

    def validate(self) -> dict:
        """
        Validate

        :return: Loss and validation metrics as defined in the ValidationSpec
        """

        if self._evaluation_spec is None or self._validation_handle is None:
            raise RuntimeError('Must call setup with validation_input_fn before using the validate method')

        # Init
        global_step = tf.train.get_global_step(graph=self._graph)
        local_step = 0
        metric_update_ops = [v[1] for v in self._evaluation_spec.metric_ops.values()]  # Gather update ops
        losses = []  # Gather losses per batch
        self._session.run([self._validation_iterator.initializer,
                           self._running_variables_initializer])
        validation_pbar = tqdm(total=self._validation_size,
                               desc='Validation',
                               ncols=self.TQDM_NCOLS)

        session_vars = [self._evaluation_spec.loss,  # Loss
                        self._visualization_op,  # TB visualization op
                        ] + metric_update_ops

        # Evaluation
        while True:
            try:
                # Evaluation step
                batch_results = self._session.run(fetches=session_vars,
                                                  feed_dict={self._handle: self._validation_handle,
                                                             self._is_training: False})

                local_step += 1

                # Unpack
                batch_losses = batch_results[0]

                # Gather losses
                losses.append(batch_losses)

                # Update progress bar
                validation_pbar.update()

            except tf.errors.OutOfRangeError:

                # Calculate total loss
                validation_loss = np.array(losses).mean()

                # Gather metrics
                metric_names = list(self._evaluation_spec.metric_ops.keys())
                metric_tensors = [v[0] for v in self._evaluation_spec.metric_ops.values()]
                metric_values = self._session.run(fetches=metric_tensors + [global_step])
                global_step_ = metric_values[-1]
                metrics = dict(zip(metric_names, metric_values[:-1]))

                # Add loss
                metrics['loss'] = validation_loss

                # Close bar
                validation_pbar.close()

                # Write metrics
                tqdm.write(
                    ' '.join(['='.join([str(metric), str(metric_value)]) for metric, metric_value in metrics.items()]))

                self._validation_size = local_step

                # Visualize
                for tag, value in metrics.items():
                    summary = tf.Summary()
                    summary.value.add(tag=tag, simple_value=value)
                    self._validation_summary_writer.add_summary(summary, global_step=global_step_)

                # Save last validation metrics
                self._last_validation_metrics = (global_step_, metrics)

                # Associate with save
                if global_step_ in self._checkpoints:
                    self._checkpoints[global_step_]['metrics'] = metrics

                return metrics

    def predict(self, *args,
                input_fn: Optional[InputFn] = None,
                is_training: Optional[bool] = False,
                **kwargs) -> Union[Generator[dict, None, None], dict]:
        """
        Predict. Supply args / kwargs / input function (only one).

        :param args:
        :param input_fn:
        :param is_training:
        :param kwargs:
        :return:
        """

        if self._predict_spec is None:
            raise RuntimeError('Must define an PredictSpec before using the predict method')

        if sum([bool(args), bool(input_fn), bool(kwargs)]) != 1:
            raise ValueError('Must supply ONE AND ONLY ONE of positional args / key-word args / input function')

        # Predict from dataset
        if input_fn:
            return self._predict_from_dataset(input_fn=input_fn, is_training=is_training)

        # Predict using feed dict
        feed_dict = dict()
        if args:
            feed_dict = {self._input_signature[index]: value for index, value in enumerate(args)}
        elif kwargs:
            for key, value in kwargs.items():
                try:
                    feed_dict[self._input_signature[key]] = value
                except KeyError:
                    raise KeyError(
                        '{} not found in input signature. ({})'.format(key, ', '.join(self._input_signature.keys())))

        # Add mode
        feed_dict[self._is_training] = is_training

        # Get session vars
        prediction_keys = list(self._predict_spec.output.keys())
        prediction_values = list(self._predict_spec.output.values())
        session_vars = prediction_values

        # Run on graph
        results = self._session.run(fetches=session_vars, feed_dict=feed_dict)

        predictions = dict()
        for key, value in zip(prediction_keys, results):
            predictions[key] = value

        return predictions

    def evaluate(self,
                 input_fn: InputFn,
                 is_training: Optional[bool] = False) -> dict:
        """
        Evaluate

        :param input_fn: Input function
        :param is_training: Training mode
        :return: Evaluation metrics
        """

        if self._evaluation_spec is None:
            raise RuntimeError('Must define an EvaluationSpec before using the evaluate method')

        # Init
        with self._graph.as_default():
            dataset = input_fn()
            iterator = dataset.make_initializable_iterator()

        handle = self._session.run(iterator.string_handle())
        metric_update_ops = [v[1] for v in self._evaluation_spec.metric_ops.values()]  # Gather update op
        self._session.run([iterator.initializer,
                           self._running_variables_initializer])
        evaluation_pbar = tqdm(total=None,
                               desc='Evaluation',
                               ncols=self.TQDM_NCOLS)
        session_vars = metric_update_ops

        # Evaluation
        while True:
            try:
                # Evaluation step
                self._session.run(fetches=session_vars,
                                  feed_dict={self._handle: handle,
                                             self._is_training: is_training})

                # Update progress bar
                evaluation_pbar.update()

            except tf.errors.OutOfRangeError:

                # Gather metrics
                metric_names = list(self._evaluation_spec.metric_ops.keys())
                metric_tensors = [v[0] for v in self._evaluation_spec.metric_ops.values()]
                metric_values = self._session.run(fetches=metric_tensors)
                metrics = dict(zip(metric_names, metric_values))

                # Close bar
                evaluation_pbar.close()

                return metrics

    def save_ckpt(self) -> str:
        """
        Saves a checkpoint.

        :return: Path prefix to the checkpoint file
        """

        # Get global step
        global_step = self.global_step

        # Save checkpoint
        ckpt_file = self._train_saver.save(self._session,
                                           save_path=str(self.train_checkpoints / 'model_iter'),
                                           global_step=global_step)
        self._checkpoints[global_step] = {'file': ckpt_file}

        # Associate with metrics
        if self._last_validation_metrics and self._last_validation_metrics[0] == global_step:
            metrics = self._last_validation_metrics[1]
        else:
            metrics = {}
        self._checkpoints[global_step]['metrics'] = metrics

        # Remove old entries (not tracked by the train.Saver)
        keys_to_remove = []
        for ckpt_key, ckpt_values in self._checkpoints.items():
            if ckpt_values['file'] not in self._train_saver.last_checkpoints:
                keys_to_remove.append(ckpt_key)

        for key_to_remove in keys_to_remove:
            del self._checkpoints[key_to_remove]

        # Save Elistimator meta data
        self._save_meta()

        return ckpt_file

    def restore_ckpt(self, path_prefix: str):
        """
        Restores graph and variables from a checkpoint prefix

        :param path_prefix: Path prefix
        """

        with self._graph.as_default():
            self._train_saver.restore(self._session, path_prefix)

    def restore_best_ckpt(self):
        """
        Restores graph and variables from a the best (using validation loss as measure) checkpoint saved.
        If validation loss is not available, returns the last checkpoint
        """

        # Get best or latest checkpoint
        best_checkpoint = self._best_checkpoint()['file']

        self.restore_ckpt(path_prefix=best_checkpoint)

    def restore_latest_ckpt(self):
        """
        Restores graph and variables from the latest checkpoint saved
        """

        # Get latest checkpoint
        latest_checkpoint = self._latest_checkpoint()['file']

        self.restore_ckpt(path_prefix=latest_checkpoint)

    def _best_checkpoint(self) -> dict:
        """
        Gets best checkpoint entry from the

        :return: Best checkpoint entry
        """

        return sorted(self._checkpoints.items(), key=lambda k_v: k_v[1]['metrics'].get('loss', -k_v[0]))[0][1]

    def _latest_checkpoint(self) -> dict:
        """
        Gets latest checkpoint entry from the

        :return: Latest checkpoint entry
        """

        return sorted(self._checkpoints.items(), key=lambda k_v: -k_v[0])[0][1]

    def _predict_from_dataset(self, input_fn: InputFn,
                              is_training: Optional[bool] = False) -> Generator[dict, None, None]:
        """
        Predict from dataset

        :param input_fn: Input function
        :param is_training: Training mode
        :return: Predictions
        """

        # Init
        with self._graph.as_default():
            dataset = input_fn()
            iterator = dataset.make_initializable_iterator()

        self._session.run(iterator.initializer)
        handle = self._session.run(iterator.string_handle())
        prediction_keys = list(self._predict_spec.output.keys())
        prediction_values = list(self._predict_spec.output.values())
        session_vars = prediction_values

        # Predictions
        with suppress(tf.errors.OutOfRangeError):
            while True:
                # Prediction step
                predictions = dict()
                batch_results = self._session.run(fetches=session_vars,
                                                  feed_dict={self._handle: handle,
                                                             self._is_training: is_training})

                for key, value in zip(prediction_keys, batch_results):
                    predictions[key] = value

                yield predictions

    @staticmethod
    def _get_or_create_directory(model_dir: Union[str, None]) -> Path:
        """
        Gets if exists or creates a model directory. Will create a temporary one if model_dir == None

        :param model_dir: Path to model directory
        :return: Path to model directory
        """

        if isinstance(model_dir, str):
            model_dir = Path(model_dir)
            if not model_dir.parent.exists():
                raise FileNotFoundError("Parent directory doesn't exist. ({})".format(model_dir.parent))
            if not model_dir.exists():
                model_dir.mkdir()
        else:
            model_dir = TemporaryDirectory()
            model_dir = Path(model_dir.name)

        return model_dir

    def _save_meta(self):
        """
        Saves Estimators attributes

        """

        attrs_to_save = {
            '_handle': self._handle.name,
            '_train_spec': self._train_spec.to_dict() if self._train_spec else self._train_spec,
            '_evaluation_spec': self._evaluation_spec.to_dict() if self._evaluation_spec else self._evaluation_spec,
            '_predict_spec': self._predict_spec.to_dict() if self._predict_spec else self._predict_spec,
            '_is_training': self._is_training.name,
            '_running_variables_initializer': self._running_variables_initializer.name,
            '_visualization_op': self._visualization_op.name,
            '_input_signature': self._input_signature,
            '_checkpoints': self._checkpoints,
            '_train_count': self._train_count,
            '_global_vars_initialized': self._global_vars_initialized}

        with open(str(self.train_checkpoints / self.ELISTIMATOR_META_FILENAME), 'wb') as e_meta:
            pickle.dump(obj=attrs_to_save, file=e_meta)

    @staticmethod
    def _load_meta(meta_file: str) -> dict:
        """
        Loads an Elistimator meta file from disk

        :param meta_file: Meta file path
        :return: Elistimator meta data
        """

        with open(meta_file, 'rb') as mf:
            return pickle.load(mf)

    @staticmethod
    def from_disk(model_dir: str,
                  from_best: Optional[bool] = False,
                  session_args: Optional[dict] = None) -> 'Elistimator':
        """
        Load an Estimator from disk. Requires at least one checkpoint save.

        :param model_dir: Model directory
        :param from_best: Restore best checkpoint, if False, restores latest.
        :param session_args: Parameters to pass to the tf.Session
        :return: Estimator
        """

        model_dir = Path(model_dir)

        # Check whether Estimator was saved at least once
        meta_file = model_dir / 'checkpoints' / Elistimator.ELISTIMATOR_META_FILENAME
        if not meta_file.exists():
            raise ValueError('No checkpoints to restore from')

        # Get Estimator meta data
        meta = Elistimator._load_meta(str(meta_file))

        # Init empty Estimator
        estimator = Elistimator(model_fn=None,
                                model_dir=str(model_dir),
                                session_args=session_args)
        # Restore input signature
        estimator._input_signature = meta['_input_signature']

        # Restore checkpoints
        estimator._checkpoints = meta['_checkpoints']

        # Restore saver
        meta_graph = estimator._best_checkpoint() if from_best else estimator._latest_checkpoint()
        with estimator._graph.as_default():
            estimator._train_saver = tf.train.import_meta_graph(meta_graph_or_file='{}.meta'.format(meta_graph['file']))

            # Restore running vars initializer
            estimator._running_variables_initializer = estimator._graph.get_operation_by_name(
                meta['_running_variables_initializer'])

        # Restore graph from checkpoint
        if from_best:
            estimator.restore_best_ckpt()
        else:
            estimator.restore_latest_ckpt()

        # Restore Elistimator specs
        train_spec = TrainSpec.from_dict(graph=estimator.graph, dictionary=meta['_train_spec']) \
            if meta['_train_spec'] else None
        evaluation_spec = EvaluationSpec.from_dict(graph=estimator.graph, dictionary=meta['_evaluation_spec']) \
            if meta['_evaluation_spec'] else None
        predict_spec = PredictSpec.from_dict(graph=estimator.graph, dictionary=meta['_predict_spec']) \
            if meta['_predict_spec'] else None

        # Restore specs
        estimator._predict_spec = predict_spec
        estimator._evaluation_spec = evaluation_spec
        estimator._train_spec = train_spec

        # Restore model_fn
        model_fn = lambda *args: (train_spec, evaluation_spec, predict_spec)
        estimator._model_fn = model_fn

        # Restore handle
        estimator._handle = estimator._graph.get_tensor_by_name(meta['_handle'])

        # Restore is_training placeholder
        estimator._is_training = estimator._graph.get_tensor_by_name(meta['_is_training'])

        # Restore visualization op
        estimator._visualization_op = estimator._graph.get_tensor_by_name(meta['_visualization_op'])

        # Restore train count
        estimator._train_count = meta['_train_count']

        # Restore global var status
        estimator._global_vars_initialized = meta['_global_vars_initialized']

        return estimator
