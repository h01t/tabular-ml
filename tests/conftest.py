"""Shared test configuration."""

import pytest


@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    """Restrict async tests to asyncio only (skip trio)."""
    return request.param
