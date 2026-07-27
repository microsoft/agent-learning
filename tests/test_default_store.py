"""Tests for get_default_store backend selection (defaults to InMemoryStore)."""

from __future__ import annotations

import agent_learning.storage.cosmos as cosmos_mod
from agent_learning.storage import InMemoryStore, LocalFileStore, get_default_store


def _reset_singleton() -> None:
    cosmos_mod._default_store = None


def test_defaults_to_in_memory(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_LEARNING_STORE_BACKEND", raising=False)
    _reset_singleton()
    try:
        assert isinstance(get_default_store(), InMemoryStore)
    finally:
        _reset_singleton()


def test_explicit_memory_backend(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_LEARNING_STORE_BACKEND", "memory")
    _reset_singleton()
    try:
        assert isinstance(get_default_store(), InMemoryStore)
    finally:
        _reset_singleton()


def test_local_backend(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AGENT_LEARNING_STORE_BACKEND", "local")
    monkeypatch.setenv("AGENT_LEARNING_LOCAL_STORE_DIR", str(tmp_path))
    _reset_singleton()
    try:
        assert isinstance(get_default_store(), LocalFileStore)
    finally:
        _reset_singleton()


def test_unknown_backend_falls_back_to_memory(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_LEARNING_STORE_BACKEND", "nope")
    _reset_singleton()
    try:
        assert isinstance(get_default_store(), InMemoryStore)
    finally:
        _reset_singleton()


def test_singleton_is_cached(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_LEARNING_STORE_BACKEND", raising=False)
    _reset_singleton()
    try:
        assert get_default_store() is get_default_store()
    finally:
        _reset_singleton()
