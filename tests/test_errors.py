from __future__ import annotations

import pytest

from promptlab.errors import (
    PermanentProviderError,
    TransientProviderError,
    TruncatedResponseError,
    UnknownModelError,
)

NEW_ERRORS: tuple[type[BaseException], ...] = (
    TransientProviderError,
    PermanentProviderError,
    TruncatedResponseError,
)


@pytest.mark.parametrize(
    ("error_type", "expected_name"),
    [
        (TransientProviderError, "TransientProviderError"),
        (PermanentProviderError, "PermanentProviderError"),
        (TruncatedResponseError, "TruncatedResponseError"),
    ],
)
def test_provider_error_is_exception_with_stable_name(
    error_type: type[BaseException],
    expected_name: str,
) -> None:
    error = error_type("fixture")

    assert isinstance(error, Exception)
    assert type(error).__name__ == expected_name
    assert error_type.__name__ == expected_name


def test_new_error_types_are_not_caught_as_each_other() -> None:
    for error_type in NEW_ERRORS:
        for other in NEW_ERRORS:
            if error_type is other:
                continue
            with pytest.raises(error_type):
                try:
                    raise error_type("fixture")
                except other:
                    pytest.fail(f"{error_type.__name__} was caught as {other.__name__}")


def test_unknown_model_error_remains_day1_value_error() -> None:
    error = UnknownModelError("missing-model")

    assert type(error) is UnknownModelError
    assert isinstance(error, ValueError)
    assert type(error).__name__ == "UnknownModelError"
    assert UnknownModelError.__mro__[1] is ValueError


def test_unknown_model_error_is_not_a_provider_or_truncation_error() -> None:
    error = UnknownModelError("missing-model")

    assert not isinstance(error, TransientProviderError)
    assert not isinstance(error, PermanentProviderError)
    assert not isinstance(error, TruncatedResponseError)

    with pytest.raises(UnknownModelError):
        try:
            raise UnknownModelError("missing-model")
        except TransientProviderError:
            pytest.fail("UnknownModelError was caught as TransientProviderError")
        except PermanentProviderError:
            pytest.fail("UnknownModelError was caught as PermanentProviderError")
        except TruncatedResponseError:
            pytest.fail("UnknownModelError was caught as TruncatedResponseError")
