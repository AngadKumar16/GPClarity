Diagnostic Reporting
====================

:func:`~gpclarity.generate_report` runs every GPClarity analysis against one
model and assembles the results into a single document you can print, save, or
serialise — useful for notebooks, code review, and hand-offs.

Generating a report
-------------------

.. code-block:: python

   import gpclarity

   report = gpclarity.generate_report(
       model,
       X=X_train,
       y=y_train,
       X_test=X_test,
       title="My GP diagnostics",
   )

   print(report.summary)        # a few headline verdicts
   print(report.to_markdown())  # full document

The report contains up to five sections: ``health``, ``kernel``, ``complexity``,
``uncertainty``, and ``influence``. Sections that need data you did not pass are
skipped, and any section whose analysis raises is recorded as an error rather
than aborting the whole report — so you always get a complete document.

Restricting to specific sections
---------------------------------

.. code-block:: python

   report = gpclarity.generate_report(
       model, X=X_train, include=["health", "kernel", "complexity"]
   )

Exporting
---------

The format is inferred from the file extension.

.. code-block:: python

   report.save("diagnostics.md")     # Markdown
   report.save("diagnostics.html")   # self-contained HTML
   report.save("diagnostics.json")   # machine-readable

   # Or get the strings directly
   md = report.to_markdown()
   html = report.to_html()
   data = report.to_dict()
