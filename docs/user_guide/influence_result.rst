gpclarity.InfluenceResult
=========================

.. currentmodule:: gpclarity.data_influence

.. autoclass:: InfluenceResult
   :no-members:
   :show-inheritance:

Attributes
----------

.. attribute:: scores
   :type: np.ndarray
   
   Array of influence scores with shape ``(n_points,)``. Higher values indicate more influential points. Computed as inverse leverage scores from the GP hat matrix.

.. attribute:: method
   :type: str
   
   Computation method identifier. Currently always ``"leverage"``.

.. attribute:: computation_time
   :type: float
   
   Wall clock time for computation in seconds.

.. attribute:: n_points
   :type: int
   
   Number of training points analyzed.

.. attribute:: metadata
   :type: Optional[Dict[str, Any]]
   
   Dictionary containing computation metadata:
   
   - ``kernel_type``: Type of kernel used (e.g., ``"RBF"``, ``"Matern52"``)
   - ``noise_variance``: Noise variance from model
   - ``cache_used``: Whether kernel cache was utilized

Methods
-------

.. method:: __array__() -> np.ndarray

   Allow numpy operations on result directly.
   
   **Example:**
   
   .. code-block:: python
   
      result = influence.compute_influence_scores(X_train)
      
      # Direct numpy operations
      mean_influence = np.mean(result)
      top_10_percent = np.percentile(result, 90)
      normalized = result / np.max(result)

.. method:: __len__() -> int

   Return number of points.
   
   **Example:**
   
   .. code-block:: python
   
      n_points = len(result)  # Same as result.n_points

Examples
--------

**Basic usage:**

.. code-block:: python

   from gpclarity import DataInfluenceMap
   
   influence = DataInfluenceMap(model)
   result = influence.compute_influence_scores(X_train)
   
   # Access scores
   scores = result.scores
   most_influential_idx = np.argmax(scores)

**With numpy operations:**

.. code-block:: python

   # Direct numpy operations on result
   mean_score = np.mean(result)
   std_score = np.std(result)
   
   # Boolean indexing
   high_influence = result > np.percentile(result, 95)
   outlier_candidates = np.where(high_influence)[0]

**Accessing metadata:**

.. code-block:: python

   print(f"Method: {result.method}")
   print(f"Computation time: {result.computation_time:.3f}s")
   print(f"Kernel type: {result.metadata['kernel_type']}")
   print(f"Noise variance: {result.metadata['noise_variance']}")
