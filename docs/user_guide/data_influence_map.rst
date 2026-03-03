gpclarity.DataInfluenceMap
==========================

.. currentmodule:: gpclarity.data_influence

.. autoclass:: DataInfluenceMap
   :no-members:
   :show-inheritance:

Constructor
-----------

.. method:: __init__(model)

   Initialize with GP model.
   
   :param model: Trained GP model with ``predict`` and ``kern`` attributes
   :type model: GPy.models.GPRegression
   :raises ValueError: If model lacks required attributes
   
   **Example:**
   
   .. code-block:: python
   
      import GPy
      from gpclarity import DataInfluenceMap
      
      kernel = GPy.kern.RBF(input_dim=2)
      model = GPy.models.GPRegression(X_train, y_train, kernel)
      model.optimize()
      
      influence = DataInfluenceMap(model)

Methods
-------

.. method:: compute_influence_scores(X_train, y_train=None, *, use_cache=True) -> InfluenceResult

   Compute influence scores using leverage scores (optimized O(n³)).
   
   Leverage scores computed via diagonal of hat matrix using cached Cholesky decomposition.
   
   :param X_train: Training input locations with shape ``(n_train, n_dims)``
   :type X_train: np.ndarray
   :param y_train: Training outputs with shape ``(n_train,)`` or ``(n_train, 1)``. Optional, validated if provided but not used for computation.
   :type y_train: np.ndarray, optional
   :param use_cache: Whether to use internal cache for kernel matrix. Default: ``True``
   :type use_cache: bool
   :returns: InfluenceResult containing scores and metadata
   :rtype: InfluenceResult
   :raises ValueError: If X_train is not 2D array-like
   :raises InfluenceError: If computation fails
   
   **Example:**
   
   .. code-block:: python
   
      result = influence.compute_influence_scores(X_train)
      
      # Get most influential point
      most_influential_idx = np.argmax(result.scores)
      print(f"Point {most_influential_idx}: score={result.scores[most_influential_idx]:.3f}")

.. method:: compute_loo_variance_increase(X_train, y_train, *, n_jobs=1, verbose=False) -> Tuple[np.ndarray, np.ndarray]

   Exact Leave-One-Out variance increase with optional parallelization.
   
   :param X_train: Training inputs with shape ``(n_train, n_dims)``
   :type X_train: np.ndarray
   :param y_train: Training outputs with shape ``(n_train,)`` or ``(n_train, 1)``
   :type y_train: np.ndarray
   :param n_jobs: Number of parallel jobs. ``-1`` for all cores, ``1`` for sequential. Default: ``1``
   :type n_jobs: int
   :param verbose: Whether to display progress bar. Requires ``tqdm``. Default: ``False``
   :type verbose: bool
   :returns: Tuple of ``(variance_increase, prediction_errors)``. Both arrays shape ``(n_train,)``.
   :rtype: tuple
   :raises InfluenceError: If computation fails
   
   **Example:**
   
   .. code-block:: python
   
      var_increase, pred_errors = influence.compute_loo_variance_increase(
          X_train, y_train, n_jobs=-1, verbose=True
      )
      
      # Identify outliers: high variance increase AND high prediction error
      outlier_mask = (var_increase > np.percentile(var_increase, 95)) & \\
                     (pred_errors > np.percentile(pred_errors, 95))
      outliers = np.where(outlier_mask)[0]

.. method:: get_influence_report(X_train, y_train, *, compute_loo=True, n_jobs=1) -> Dict[str, Any]

   Comprehensive influence analysis report.
   
   :param X_train: Training inputs
   :type X_train: np.ndarray
   :param y_train: Training outputs
   :type y_train: np.ndarray
   :param compute_loo: Whether to include LOO analysis (slow for large n). Default: ``True``
   :type compute_loo: bool
   :param n_jobs: Parallel jobs for LOO computation. Default: ``1``
   :type n_jobs: int
   :returns: Dictionary with influence statistics and diagnostics
   :rtype: dict
   
   **Return structure:**
   
   .. code-block:: python
   
      {
          'computation_summary': {
              'total_time': float,
              'leverage_time': float,
              'n_points': int,
              'method': str
          },
          'influence_scores': {
              'mean': float,
              'std': float,
              'median': float,
              'max': float,
              'min': float,
              'p95': float,
              'p5': float
          },
          'most_influential_point': {
              'index': int,
              'location': List[float],
              'score': float
          },
          'least_influential_point': {
              'index': int,
              'location': List[float],
              'score': float
          },
          'diagnostics': {
              'high_leverage_count': int,
              'low_influence_count': int,
              'non_finite_scores': int
          },
          'loo_analysis': {  # Only if compute_loo=True
              'variance_increase': List[float],
              'prediction_errors': List[float],
              'mean_error': float,
              'max_error': float
          }
      }
   
   **Example:**
   
   .. code-block:: python
   
      report = influence.get_influence_report(X_train, y_train, compute_loo=True)
      
      # Check influence distribution
      mean_score = report['influence_scores']['mean']
      high_leverage = report['diagnostics']['high_leverage_count']
      
      # Get most influential point details
      most_inf = report['most_influential_point']
      print(f"Most influential: Point {most_inf['index']} at {most_inf['location']}")

.. method:: plot_influence(X_train, influence_scores, ax=None, **scatter_kwargs) -> plt.Axes

   Visualize data point influence.
   
   Delegated to ``gpclarity.plotting.plot_influence_map``.
   
   :param X_train: Training input locations
   :type X_train: np.ndarray
   :param influence_scores: Computed scores or InfluenceResult
   :type influence_scores: np.ndarray or InfluenceResult
   :param ax: Matplotlib axes. Created if None.
   :type ax: plt.Axes, optional
   :param scatter_kwargs: Additional arguments passed to ``ax.scatter()``
   :returns: Matplotlib axes object
   :rtype: plt.Axes
   :raises ImportError: If matplotlib not installed
   :raises ValueError: If input dimensions > 2
   
   **Example:**
   
   .. code-block:: python
   
      import matplotlib.pyplot as plt
      
      fig, ax = plt.subplots()
      influence.plot_influence(X_train, result, ax=ax, s=100, alpha=0.6, cmap='hot')
      plt.show()

.. method:: clear_cache() -> None

   Clear internal computation cache to free memory.
   
   **Example:**
   
   .. code-block:: python
   
      influence.clear_cache()