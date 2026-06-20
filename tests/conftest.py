"""Shared pytest fixtures and path setup."""

from __future__ import annotations

import pathlib
import sys

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genaction import CreateOrReuseEnv, Embedder, LossModel  # noqa: E402


@pytest.fixture(scope="session")
def faq_df() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "sample_faq_library.csv")


@pytest.fixture(scope="session")
def stream_df() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "sample_stream.csv")


@pytest.fixture
def env(faq_df, stream_df) -> CreateOrReuseEnv:
    """A fresh environment using the deterministic TF-IDF backend."""
    return CreateOrReuseEnv(
        faq_df=faq_df,
        stream_df=stream_df,
        embedder=Embedder(backend="tfidf"),
        loss_model=LossModel(),
        creation_cost=0.2,
        seed=0,
    )
