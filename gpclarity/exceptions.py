"""
Custom exceptions for GPClarity.
"""


class GPClarityError(Exception):
    """Base exception for all GPClarity errors."""
    pass


class InfluenceError(GPClarityError):
    """Raised when influence computation fails."""
    pass


class KernelError(GPClarityError):
    """Raised when kernel operations fail."""
    pass


class ModelError(GPClarityError):
    """Raised when model validation fails."""
    pass

class TrackingError(GPClarityError):
    """Raised when parameter tracking fails."""
    pass


class OptimizationError(GPClarityError):
    """Raised when optimization fails."""
    pass


class ComplexityError(GPClarityError):
    """Raised when complexity computation or analysis fails."""
    pass

class UncertaintyError(GPClarityError):
    """Raised when uncertainty quantification or analysis fails."""
    pass

class LinAlgError(GPClarityError):
    """Raised when linear algebra operations fail."""
    pass


class ValidationError(GPClarityError):
    """Raised when input validation fails."""
    pass