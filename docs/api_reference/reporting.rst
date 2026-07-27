Diagnostic Reporting
====================

The ``reporting`` module runs the full GPClarity battery — health check, kernel
summary, complexity score, uncertainty diagnostics, and data influence — against
a single model and packages the results into a
:class:`~gpclarity.DiagnosticReport` that prints, exports to Markdown or
self-contained HTML, or dumps to JSON. Each section is computed defensively: a
failing analysis is recorded as a section error instead of aborting the report.

**When to use:** to produce a shareable, one-call summary of a model's behaviour
for a notebook, a code review, or a hand-off.

.. code-block:: python

   import gpclarity

   report = gpclarity.generate_report(
       model, X=X_train, y=y_train, X_test=X_test, title="My GP diagnostics"
   )

   print(report.summary)          # headline verdicts
   report.save("diagnostics.html")
   report.save("diagnostics.md")
   data = report.to_dict()        # or report.to_json()

.. automodule:: gpclarity.reporting
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autosummary::
   :nosignatures:

   generate_report

Data Classes
------------

.. autosummary::
   :nosignatures:

   DiagnosticReport
