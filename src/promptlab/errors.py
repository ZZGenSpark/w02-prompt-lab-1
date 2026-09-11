"""Errors used by the Week 2 local model lab."""


class UnknownModelError(ValueError):
    """Raised when a model identifier is not present in the configured model table."""


class TransientProviderError(Exception):
    """Raised on timeout, connection failure, or a temporary Ollama/server failure."""


class PermanentProviderError(Exception):
    """Raised on a malformed request, unavailable model, or other non-retryable failure."""


class TruncatedResponseError(Exception):
    """Raised when Ollama reports that the output token ceiling was reached."""
    