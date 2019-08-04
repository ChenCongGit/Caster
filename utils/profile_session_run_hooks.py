import os
import logging
import collections

import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.training import training_util
from tensorflow.python.training import session_run_hook

class SessionRunHook():
  """
  这个类用于Session运行过程中，它的几个实例方法分别应用在session的不同时刻，
  包括开始创建session时，sess.run()之前，sess.run()之后或sess结束时
  """
  def begin(self):
    """Called once before using the session.
    When called, the default graph is the one that will be launched in the
    session.  The hook can modify the graph by adding new operations to it.
    After the `begin()` call the graph will be finalized and the other callbacks
    can not modify the graph anymore. Second call of `begin()` on the same
    graph, should not change the graph.
    """
    pass

  def after_create_session(self, session, coord):  # pylint: disable=unused-argument
    """Called when new TensorFlow session is created.
    This is called to signal the hooks that a new session has been created. This
    has two essential differences with the situation in which `begin` is called:
    * When this is called, the graph is finalized and ops can no longer be added
        to the graph.
    * This method will also be called as a result of recovering a wrapped
        session, not only at the beginning of the overall session.
    Args:
      session: A TensorFlow Session that has been created.
      coord: A Coordinator object which keeps track of all threads.
    """
    pass

  def before_run(self, run_context):  # pylint: disable=unused-argument
    """Called before each call to run().
    You can return from this call a `SessionRunArgs` object indicating ops or
    tensors to add to the upcoming `run()` call.  These ops/tensors will be run
    together with the ops/tensors originally passed to the original run() call.
    The run args you return can also contain feeds to be added to the run()
    call.
    The `run_context` argument is a `SessionRunContext` that provides
    information about the upcoming `run()` call: the originally requested
    op/tensors, the TensorFlow Session.
    At this point graph is finalized and you can not add ops.
    Args:
      run_context: A `SessionRunContext` object.
    Returns:
      None or a `SessionRunArgs` object.
    """
    return None

  def after_run(self,
                run_context,  # pylint: disable=unused-argument
                run_values):  # pylint: disable=unused-argument
    """Called after each call to run().
    The `run_values` argument contains results of requested ops/tensors by
    `before_run()`.
    The `run_context` argument is the same one send to `before_run` call.
    `run_context.request_stop()` can be called to stop the iteration.
    If `session.run()` raises any exceptions then `after_run()` is not called.
    Args:
      run_context: A `SessionRunContext` object.
      run_values: A SessionRunValues object.
    """
    pass

  def end(self, session):  # pylint: disable=unused-argument
    """Called at the end of session.
    The `session` argument can be used in case the hook wants to run final ops,
    such as saving a last checkpoint.
    If `session.run()` raises exception other than OutOfRangeError or
    StopIteration then `end()` is not called.
    Note the difference between `end()` and `after_run()` behavior when
    `session.run()` raises OutOfRangeError or StopIteration. In that case
    `end()` is called but `after_run()` is not called.
    Args:
      session: A TensorFlow Session that will be soon closed.
    """
    pass


class SessionRunArgs(
    collections.namedtuple("SessionRunArgs",
                           ["fetches", "feed_dict", "options"])):
  """Represents arguments to be added to a `Session.run()` call.
  Args:
    fetches: Exactly like the 'fetches' argument to Session.Run().
      Can be a single tensor or op, a list of 'fetches' or a dictionary
      of fetches.  For example:
        fetches = global_step_tensor
        fetches = [train_op, summary_op, global_step_tensor]
        fetches = {'step': global_step_tensor, 'summ': summary_op}
      Note that this can recurse as expected:
        fetches = {'step': global_step_tensor,
                   'ops': [train_op, check_nan_op]}
    feed_dict: Exactly like the `feed_dict` argument to `Session.Run()`
    options: Exactly like the `options` argument to `Session.run()`, i.e., a
      config_pb2.RunOptions proto.
  """

  def __new__(cls, fetches, feed_dict=None, options=None):
    return super(SessionRunArgs, cls).__new__(cls, fetches, feed_dict, options)



class ProfileAtStepHook(session_run_hook.SessionRunHook):
  """Hook that requests stop at a specified step."""

  def __init__(self, at_step=None, checkpoint_dir=None, trace_level=tf.RunOptions.FULL_TRACE):
    self._at_step = at_step
    self._do_profile = False
    self._writer = tf.summary.FileWriter(checkpoint_dir)
    self._trace_level = trace_level

  def begin(self):
    self._global_step_tensor = tf.train.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError("Global step should be created to use ProfileAtStepHook.")

  def before_run(self, run_context):  # pylint: disable=unused-argument
    if self._do_profile:
      options = tf.RunOptions(trace_level=self._trace_level)
    else:
      options = None
    return tf.train.SessionRunArgs(self._global_step_tensor, options=options)

  def after_run(self, run_context, run_values):
    global_step = run_values.results - 1
    if self._do_profile:
      self._do_profile = False
      self._writer.add_run_metadata(run_values.run_metadata,
                                    'trace_{}'.format(global_step), global_step)
      timeline_object = timeline.Timeline(run_values.run_metadata.step_stats)
      chrome_trace = timeline_object.generate_chrome_trace_format()
      chrome_trace_save_path = 'timeline_{}.json'.format(global_step)
      with open(chrome_trace_save_path, 'w') as f:
        f.write(chrome_trace)
      logging.info('Profile trace saved to {}'.format(chrome_trace_save_path))
    if global_step == self._at_step:
      self._do_profile = True
