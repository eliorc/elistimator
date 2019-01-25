from typing import Callable, Optional, Tuple, Union, Dict, List, Any
from pathlib import Path
from tempfile import TemporaryDirectory
from itertools import count
from collections import defaultdict
from contextlib import suppress

import petname
from tqdm import tqdm
import numpy as np
import tensorflow as tf

# <editor-fold desc="Data types">
ModelFn = Callable[[Dict[str, tf.Tensor], tf.Tensor, Optional[dict]], Tuple[
    'TrainSpec', 'ValidationSpec', Optional['PredictSpec']]]
InputFn = Callable[[], tf.data.Dataset]


# </editor-fold>

# TODO - Create evaluate() to be able to evaluate model on different datasets
# TODO - Create save_ckpt() which will save checkpoints, also allow to alter saving options
# TODO - Remove name of model if not necessary


class TrainSpec:

    def __init__(self,
                 loss: tf.Tensor,
                 train_op: tf.train.Optimizer):
        """

        :param loss: Loss tensor
        :param train_op: Training operation (notice, this is not the optimizer, but the optimization call [ex. optimizer.minimize()])
        """

        self.loss = loss
        self.train_op = train_op

        # Visualize
        tf.summary.scalar('loss', self.loss)


class EvaluationSpec:

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
                tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=_get_scope(tensor_name=metric_tensor.name)))

        return running_vars


class PredictSpec:

    def __init__(self,
                 output: Dict[str, tf.Tensor]):
        """

        :param output: Network's output
        """

        self.output = output


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


class Estimator:
    TQDM_NCOLS = 80

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    @property
    def train_logs(self) -> Path:
        return self._model_dir / 'train'

    @property
    def validation_logs(self) -> Path:
        return self._model_dir / 'validation'

    @train_logs.setter
    def train_logs(self, value):
        pass

    def __init__(self,
                 model_fn: ModelFn,
                 name: Optional[str] = None,
                 model_dir: Optional[str] = None,
                 params: Optional[dict] = None):
        """
        Creates a new instance of the Estimator class

        :param model_fn: Model creating function
        :param name: Name of the model
        :param model_dir: Directory to host all files related to the model
        :param params: Parameters to be available to the model_fn function
        """

        self._model_fn = model_fn
        self._name = name if name is not None else petname.Generate()  # Generate random name
        self._model_dir = self._get_or_create_directory(model_dir)
        self._params = params if params is not None else {}

        # TF references
        self._graph = tf.Graph()
        self._train_saver = None

        self._train_size = None
        self._validation_size = None
        self._train_count = 0
        self._validation_count = 0

        # To be defined with the setup_training method
        self._handle = None
        self._train_handle = None
        self._train_iterator = None
        self._validation_handle = None
        self._validation_iterator = None
        self._train_spec = None
        self._validation_spec = None
        self._predict_spec = None
        self._is_training = None
        self._visualization_op = None
        self._running_variables_initializer = None
        self._input_signature = None
        self._train_summary_writer = None
        self._validation_summary_writer = None
        self._session = None

    def _get_or_create_directory(self,
                                 model_dir: Union[str, None]) -> Path:
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

    def setup(self,
              train_input_fn: Optional[InputFn] = None,
              validation_input_fn: Optional[InputFn] = None,
              data_keys: Optional[List[str]] = None) -> 'Estimator':

        # Init
        validation_iterator = None
        train_iterator = None
        self._train_size = None
        self._validation_size = None

        if not train_input_fn and not validation_input_fn:
            return self

        # If data_keys not provided, use 0, 1, 2 ...
        data_keys = data_keys if data_keys else count()

        with self._graph.as_default():

            tf.train.create_global_step()

            with tf.variable_scope('input'):
                if train_input_fn:
                    # Setup training data pipeline
                    train_dataset = train_input_fn()
                    train_iterator = train_dataset.make_initializable_iterator()

                self._is_training = tf.placeholder_with_default(False, shape=[], name='is_training')

                if validation_input_fn:
                    # Setup validation data pipeline
                    validation_dataset = validation_input_fn()
                    validation_iterator = validation_dataset.make_initializable_iterator()

                # Iterators handling
                output_types = train_dataset.output_types if train_input_fn else validation_dataset.output_types
                output_shapes = train_dataset.output_shapes if train_input_fn else validation_dataset.output_shapes
                handle = tf.placeholder(tf.string, shape=[], name='iterator_handle')
                iterator = tf.data.Iterator.from_string_handle(handle,
                                                               output_types=output_types,
                                                               output_shapes=output_shapes)

                iterator_output = iterator.get_next()

                # Determine whether to assign keys or already supplied by dataset
                if isinstance(iterator_output, dict):
                    data = iterator_output
                else:
                    data = {data_key: tf.identity(data_tensor, name=data_key)
                            for data_key, data_tensor in zip(data_keys, iterator_output)}

            train_spec, validation_spec, predict_spec = self._model_fn(data, self._is_training, self._params)

            # Prepare running variables initializer for metric's running variables
            if validation_spec:
                self._running_variables_initializer = tf.variables_initializer(
                    var_list=validation_spec.get_running_variables())

            self._visualization_op = tf.summary.merge_all()

        # Create summary logs
        if validation_input_fn:
            self.train_logs.mkdir(parents=True)
        if validation_input_fn:
            self.validation_logs.mkdir(parents=True)

        # Assign
        self._session = tf.Session(graph=self._graph)
        self._train_handle = self._session.run(train_iterator.string_handle('train_handle')) if train_iterator else None
        self._train_iterator = train_iterator
        self._validation_handle = self._session.run(validation_iterator.string_handle(
            'validation_handle')) if validation_iterator else None
        self._validation_iterator = validation_iterator
        self._handle = handle
        self._train_spec = train_spec
        self._validation_spec = validation_spec
        self._predict_spec = predict_spec
        self._input_signature = {data_key: data_tensor.name for data_key, data_tensor in data.items()}
        if train_input_fn:
            self._train_summary_writer = tf.summary.FileWriter(logdir=str(self.train_logs),
                                                               graph=self._graph)
        if validation_input_fn:
            self._validation_summary_writer = tf.summary.FileWriter(logdir=str(self.validation_logs),
                                                                    graph=self._graph)

        return self

    def initialize_global_variables(self):
        """
        Calls the tf.global_variables_initializer()
        """
        with self._graph.as_default():
            self._session.run(tf.global_variables_initializer())

    def train(self,
              max_steps: Optional[int] = None,
              initialize_variables: Optional[bool] = False) -> 'Estimator':
        """
        Train. Runs for given number of steps or until iterator is exhausted

        :param max_steps: Steps to run for. (None = until iterator is exhausted)
        :param initialize_variables: Call global_variables_initializer
        :return: self, for chaining
        """

        self._train_count += 1

        if self._train_spec is None or self._train_handle is None:
            raise RuntimeError('Must call setup with train_input_fn before using the train method')

        # Variable initializing
        if initialize_variables:
            self.initialize_global_variables()

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

        # First time initialization
        if not self._session.run(global_step):
            self._session.run(self._train_iterator.initializer)

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
                train_pbar.set_postfix(loss=loss)
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

        if self._validation_spec is None or self._validation_handle is None:
            raise RuntimeError('Must call setup with validation_input_fn before using the train method')

        # Init
        global_step = tf.train.get_global_step(graph=self._graph)
        local_step = 0
        metric_update_ops = [v[1] for v in self._validation_spec.metric_ops.values()]  # Gather update ops
        losses = []  # Gather losses per batch
        self._session.run([self._validation_iterator.initializer,
                           self._running_variables_initializer])
        validation_pbar = tqdm(total=self._validation_size,
                               desc='Validation',
                               ncols=self.TQDM_NCOLS)

        session_vars = [self._validation_spec.loss,  # Loss
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
                metric_names = list(self._validation_spec.metric_ops.keys())
                metric_tensors = [v[0] for v in self._validation_spec.metric_ops.values()]
                metric_values = self._session.run(fetches=metric_tensors + [global_step])
                global_step_ = metric_values[-1]
                metrics = dict(zip(metric_names, metric_values[:-1]))

                # Add loss
                metrics['loss'] = validation_loss

                # Write metrics
                validation_pbar.write(
                    ' '.join(['='.join([str(metric), str(metric_value)]) for metric, metric_value in metrics.items()]))

                # Close bar
                validation_pbar.close()

                self._validation_size = local_step

                # Visualize
                for tag, value in metrics.items():
                    summary = tf.Summary()
                    summary.value.add(tag=tag, simple_value=value)
                    self._validation_summary_writer.add_summary(summary, global_step=global_step_)

                return metrics

    def predict(self, *args, input_fn: Optional[InputFn] = None, **kwargs) -> dict:
        """
        Predict. Supply args / kwargs / input function (only one).

        :param args:
        :param input_fn:
        :param kwargs:
        :return:
        """

        if sum([bool(args), bool(input_fn), bool(kwargs)]) != 1:
            raise ValueError('Must supply ONE AND ONLY ONE of positional args / key-word args / input function')

        # Predict from dataset
        if input_fn:
            return self._predict_from_dataset(input_fn=input_fn)

        # Predict using feed dict
        feed_dict = dict()
        input_keys = list(self._input_signature.keys())
        input_tensors = list(self._input_signature.values())
        if args:
            feed_dict = {self._input_signature[index]: value for index, value in enumerate(args)}
        elif kwargs:
            try:
                for key, value in kwargs.items():
                    feed_dict[self._input_signature[key]] = value
            except KeyError:
                raise KeyError(
                    '{} not found in input signature. ({})'.format(key, ', '.join(self._input_signature.keys())))

        # Run on graph
        results = self._session.run(fetches=input_tensors, feed_dict=feed_dict)

        predictions = dict(zip(input_keys, results))

        return predictions

    def _predict_from_dataset(self, input_fn: InputFn) -> dict:
        """
        Predict from dataset

        :param input_fn: Input function
        :return: Predictions
        """

        # Init
        dataset = input_fn()
        iterator = dataset.make_initializable_iterator()
        handle = iterator.string_handle()
        prediction_keys = list(self._predict_spec.output.keys())
        prediction_values = list(self._predict_spec.output.values())
        session_vars = prediction_values
        predictions = defaultdict(list)

        with suppress(tf.errors.OutOfRangeError):
            while True:
                # Evaluation step
                batch_results = self._session.run(fetches=session_vars,
                                                  feed_dict={self._handle: handle,
                                                             self._is_training: False})

                for key, value in zip(prediction_keys, batch_results):
                    predictions[key].append(value)

        return dict(predictions)
