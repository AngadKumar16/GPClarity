gpclarity.HyperparameterTracker
===============================

.. currentmodule:: gpclarity.hyperparam_tracker

.. autoclass:: HyperparameterTracker
   :no-members:
   :show-inheritance:

Constructor
-----------

.. method:: __init__(model)

   Initialize tracker with GP model.
   
   :param model: GPy model or compatible object with ``.parameters`` attribute and ``.optimize()`` method
   :type model: Any
   :raises TrackingError: If model lacks required attributes
   
   **Example:**
   
   .. code-block:: python
   
      import GPy
      from gpclarity.hyperparam_tracker import HyperparameterTracker
      
      kernel = GPy.kern.RBF(input_dim=1)
      model = GPy.models.GPRegression(X_train, y_train, kernel)
      
      tracker = HyperparameterTracker(model)

Properties
----------

.. attribute:: history
   :type: Dict[str, List[float]]
   
   Dictionary mapping parameter names to lists of values across iterations.
   
   **Example:**
   
   .. code-block:: python
   
      hist = tracker.history
      lengthscale_values = hist['rbf_lengthscale']
      
      import matplotlib.pyplot as plt
      plt.plot(lengthscale_values)
      plt.xlabel('Iteration')
      plt.ylabel('Lengthscale')

.. attribute:: iteration_count
   :type: int
   
   Number of recorded optimization states.

Methods
-------

.. method:: record_state(iteration=None) -> OptimizationState

   Snapshot current hyperparameter values.
   
   Returns an ``OptimizationState`` dataclass with these attributes:
   
   - ``iteration`` (int): Iteration number
   - ``parameters`` (Dict[str, Union[float, np.ndarray]]): Parameter values
   - ``log_likelihood`` (Optional[float]): Model log-likelihood
   - ``gradient_norm`` (Optional[float]): L2 norm of gradient
   - ``timestamp`` (float): Wall clock time
   
   :param iteration: Iteration number. Auto-incremented if None.
   :type iteration: int, optional
   :returns: Recorded state
   :rtype: OptimizationState
   
   **Example:**
   
   .. code-block:: python
   
      state = tracker.record_state()
      print(f"Iter {state.iteration}: LL={state.log_likelihood:.3f}")
      print(f"Parameters: {state.parameters}")

.. method:: wrapped_optimize(max_iters=100, callback=None, capture_every=1, convergence_tolerance=1e-6, patience=10, **optimize_kwargs) -> Dict[str, List[float]]

   Perform optimization with intelligent tracking and early stopping.
   
   :param max_iters: Maximum optimization iterations. Default: ``100``
   :type max_iters: int
   :param callback: Optional callback function called each iteration with ``(model, iteration, history)``
   :type callback: Callable, optional
   :param capture_every: Record state every N iterations. Default: ``1``
   :type capture_every: int
   :param convergence_tolerance: Minimum log-likelihood improvement for convergence. Default: ``1e-6``
   :type convergence_tolerance: float
   :param patience: Iterations without improvement before early stopping. Default: ``10``
   :type patience: int
   :param optimize_kwargs: Additional arguments passed to ``model.optimize()``
   :returns: Optimization history as dictionary
   :rtype: Dict[str, List[float]]
   :raises OptimizationError: If optimization fails
   
   **Example:**
   
   .. code-block:: python
   
      # Basic usage with early stopping
      history = tracker.wrapped_optimize(
          max_iters=200,
          patience=20,
          convergence_tolerance=1e-4
      )
      
      # With callback
      def callback(model, iteration, history):
          if iteration % 10 == 0:
              print(f"Iter {iteration}: LL={history[-1].log_likelihood:.2f}")
      
      history = tracker.wrapped_optimize(max_iters=100, callback=callback)

.. method:: get_parameter_trajectory(param_name) -> Tuple[np.ndarray, np.ndarray]

   Extract time-series data for a specific parameter.
   
   :param param_name: Name of parameter to extract
   :type param_name: str
   :returns: Tuple of ``(iterations, values)``
   :rtype: tuple
   :raises KeyError: If parameter not found
   :raises TrackingError: If no history recorded
   
   **Example:**
   
   .. code-block:: python
   
      iters, values = tracker.get_parameter_trajectory('rbf_lengthscale')
      
      plt.plot(iters, values)
      plt.xlabel('Iteration')
      plt.ylabel('Lengthscale')

.. method:: get_convergence_report(window=10, convergence_threshold=0.01) -> Dict[str, Dict]

   Analyze parameter convergence with statistical rigor.
   
   Returns convergence metrics for each parameter. The metrics dictionary contains:
   
   - ``initial_mean`` (float): Mean over first window
   - ``final_mean`` (float): Mean over last window
   - ``relative_change`` (float): Relative change between windows
   - ``final_std`` (float): Standard deviation over last window
   - ``coefficient_of_variation`` (float): CV = std / mean
   - ``is_converged`` (bool): Whether CV < threshold
   - ``converged`` (bool): Alias for is_converged
   - ``trend_direction`` (str): 'increasing', 'decreasing', or 'stable'
   
   :param window: Number of iterations for convergence analysis. Default: ``10``
   :type window: int
   :param convergence_threshold: CV threshold for convergence detection. Default: ``0.01``
   :type convergence_threshold: float
   :returns: Dictionary mapping parameter names to convergence metrics
   :rtype: Dict[str, Dict]
   :raises TrackingError: If insufficient history
   
   **Example:**
   
   .. code-block:: python
   
      report = tracker.get_convergence_report(window=20)
      
      for param, metrics in report.items():
          if metrics['converged']:
              print(f"✓ {param} converged to {metrics['final_mean']:.4f}")
          else:
              print(f"✗ {param} still {metrics['trend_direction']}")

.. method:: detect_optimization_issues() -> Dict[str, Any]

   Detect common optimization problems.
   
   :returns: Dictionary with ``warnings``, ``recommendations``, and ``metrics``
   :rtype: Dict[str, Any]
   
   **Example:**
   
   .. code-block:: python
   
      issues = tracker.detect_optimization_issues()
      
      for warning in issues['warnings']:
          print(f"⚠️  {warning}")
      
      for rec in issues['recommendations']:
          print(f"💡 {rec}")

.. method:: plot_evolution(params=None, figsize=None, show_convergence=True, show_ll=True, n_cols=2) -> plt.Figure

   Plot parameter trajectories with optional convergence indicators.
   
   :param params: List of parameter names to plot. All if None.
   :type params: List[str], optional
   :param figsize: Figure size as ``(width, height)``
   :type figsize: Tuple[float, float], optional
   :param show_convergence: Whether to show convergence bands. Default: ``True``
   :type show_convergence: bool
   :param show_ll: Whether to show log-likelihood subplot. Default: ``True``
   :type show_ll: bool
   :param n_cols: Number of columns in subplot grid. Default: ``2``
   :type n_cols: int
   :returns: Matplotlib figure object
   :rtype: plt.Figure
   :raises TrackingError: If no history recorded
   
   **Example:**
   
   .. code-block:: python
   
      fig = tracker.plot_evolution(
          params=['rbf_lengthscale', 'rbf_variance'],
          show_convergence=True,
          n_cols=2
      )
      plt.show()

.. method:: to_dataframe() -> pd.DataFrame

   Export history to pandas DataFrame for analysis.
   
   :returns: DataFrame with columns: ``iteration``, ``log_likelihood``, ``gradient_norm``, ``timestamp``, plus parameter columns
   :rtype: pd.DataFrame
   :raises ImportError: If pandas not installed
   
   **Example:**
   
   .. code-block:: python
   
      df = tracker.to_dataframe()
      
      # Analyze specific parameter
      print(df['rbf_lengthscale'].describe())
      
      # Export to CSV
      df.to_csv('optimization_history.csv', index=False)