from typing import Callable, Optional, Tuple, Union, Dict, List
from pathlib import Path
from tempfile import TemporaryDirectory
from itertools import count

import petname
import tensorflow as tf

# TODO - Implement eval_metric_ops plots

# <editor-fold desc="Data types">
ModelFn = Callable[[Dict[str, tf.Tensor], tf.Tensor, Optional[dict]], Tuple[
    'TrainSpec', 'ValidationSpec', Optional['PredictSpec']]]
InputFn = Callable[[], tf.data.Dataset]


# </editor-fold>


class EstimatorSpec:
    pass


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


class ValidationSpec:

    def __init__(self,
                 loss: tf.Tensor,
                 eval_metric_ops: Dict[str: Tuple[tf.Tensor, tf.Operation]]):
        """

        :param loss: Loss tensor
        :param eval_metric_ops: Evaluation metric ops, should be a dictionary where the values are a tuple of the metric, and its update op
        """

        self.loss = loss
        self.metric_ops = eval_metric_ops

        # Visualize
        tf.summary.scalar('loss', self.loss)


class PredictSpec:

    def __init__(self,
                 output: Dict[str: tf.Tensor]):
        """

        :param output: Network's output
        """

        self.output = output


class Estimator:

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
        self._session = tf.Session(graph=self._graph)
        self._train_saver = None

        # To be defined with the setup_training method
        self._handle = None
        self._train_handle = None
        self._validation_handle = None
        self._train_spec = None
        self._validation_spec = None
        self._predict_spec = None
        self._is_training = None
        self._visualization_op = None

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

    def setup_training(self,
                       train_input_fn: InputFn,
                       validation_input_fn: Optional[InputFn] = None,
                       data_keys: Optional[List[str]] = None) -> 'Estimator':

        validation_iterator = None

        # If data_keys not provided, use 0, 1, 2 ...
        data_keys = data_keys if data_keys else count()

        with self._graph.as_default():
            with tf.variable_scope('input'):
                # Setup training data pipeline
                train_dataset = train_input_fn()
                train_iterator = train_dataset.make_initializable_iterator()

                self._is_training = tf.placeholder_with_default(False, shape=[], name='is_training')

                if validation_input_fn:
                    # Setup validation data pipeline
                    validation_dataset = validation_input_fn()
                    validation_iterator = validation_dataset.make_one_shot_iterator()

                # Iterators handling
                handle = tf.placeholder(tf.string, shape=[], name='iterator_handle')
                iterator = tf.data.Iterator.from_string_handle(handle,
                                                               output_types=train_dataset.output_types,
                                                               output_shapes=train_dataset.output_shapes)

                data = {data_key: tf.identity(data_tensor, name=data_key)
                        for data_key, data_tensor in zip(data_keys, iterator.get_next())}

            train_spec, validation_spec, predict_spec = self._model_fn(data, self._is_training, self._params)

            self._visualization_op = tf.summary.merge_all()
            self.train_logs.mkdir()
            if validation_input_fn:
                self.validation_logs.mkdir()

        # Assign
        self._train_handle = train_iterator.string_handle('train_handle')
        self._validation_handle = validation_iterator.string_handle(
            'validation_handle') if validation_iterator else None
        self._handle = handle
        self._train_spec = train_spec
        self._validation_spec = validation_spec
        self._predict_spec = predict_spec

        return self

    def initialize_global_variables(self):
        """
        Calls the tf.global_variables_initializer()
        """

        self._session.run(tf.global_variables_initializer)

    def train(self,
              max_steps: Optional[int] = None,
              initialize_variables: Optional[bool] = False) -> 'Estimator':
        """
        Train. Runs for given number of steps or until iterator is exhausted

        :param max_steps: Steps to run for. (None = until iterator is exhausted)
        :param initialize_variables: Call global_variables_initializer
        :return: self, for chaining
        """

        if not self._train_spec or not self._train_handle:
            raise RuntimeError('Must call setup_training before using the train method')

        # Variable initializing
        if initialize_variables:
            self.initialize_global_variables()

        # Preparation
        if not max_steps:
            max_steps = float('inf')
        global_step = tf.train.get_global_step(graph=self._graph)
        session_vars = [self._train_spec.train.op,  # Training op
                        global_step,  # Global step
                        self._train_spec.loss,  # Loss
                        self._visualization_op]  # TB visualization op
        local_step = 0

        # Visualizing and persisting
        train_writer = tf.summary.FileWriter(logdir=str(self.train_logs),
                                             graph=self._graph)

        # Training
        while local_step < max_steps:
            try:
                # Training step
                local_step += 1
                _, global_step_, loss, visualization = self._session.run(fetches=session_vars,
                                                                         feed_dict={self._handle: self._train_handle,
                                                                                    self._is_training: True})

                # Visualizing
                train_writer.add_summary(visualization, global_step=global_step_)

                if local_step >= max_steps:
                    raise StopIteration

            except (tf.errors.OutOfRangeError, StopIteration):
                pass

        return self
