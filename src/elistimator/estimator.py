from typing import Callable, Optional, Tuple, Union, Dict, List
from pathlib import Path
from tempfile import TemporaryDirectory
from itertools import count

import petname
import tensorflow as tf

# <editor-fold desc="Data types">
ModelFn = Callable[[Dict[str, tf.Tensor], Optional[dict]], Tuple[
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

        self._loss = loss
        self._train_op = train_op


class ValidationSpec:

    def __init__(self,
                 loss: tf.Tensor,
                 eval_metric_ops: Dict[str: Tuple[tf.Tensor, tf.Operation]]):
        """

        :param loss: Loss tensor
        :param eval_metric_ops: Evaluation metric ops, should be a dictionary where the values are a tuple of the metric, and its update op
        """

        self._loss = loss
        self._metric_ops = eval_metric_ops


class Estimator:

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
                       data_keys: Optional[List[str]] = None):

        # If data_keys not provided, use 0, 1, 2 ...
        data_keys = data_keys if data_keys else count()

        with self._graph.as_default():
            with tf.variable_scope('input'):
                # Setup training data pipeline
                train_dataset = train_input_fn()
                train_iterator = train_dataset.make_initializable_iterator()

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

            train_spec, validation_spec, predict_spec = self._model_fn(data, self._params)

    def train(self,
              steps: Optional[int] = None) -> 'Estimator':
        """
        Train. Runs for given number of steps or until iterator is exhausted

        :param steps: Steps to run for. (None = until iterator is exhausted)
        :return: self, for chaining
        """

        pass

        return self
