from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from daylib_ursa import workset_api_cli as cli


def test_parse_args_defaults_and_no_bootstrap_flag() -> None:
    args = cli.parse_args([])
    assert args.region == "us-west-2"
    assert args.profile is None
    assert args.host == "0.0.0.0"
    assert args.port == 8913
    assert args.bootstrap_tapdb is True

    no_bootstrap = cli.parse_args(["--no-bootstrap-tapdb", "--port", "9000", "--verbose"])
    assert no_bootstrap.bootstrap_tapdb is False
    assert no_bootstrap.port == 9000
    assert no_bootstrap.verbose is True


def test_configure_logging_uses_expected_level(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(cli.logging, "basicConfig", lambda **kwargs: calls.append(kwargs))

    cli.configure_logging(verbose=False)
    cli.configure_logging(verbose=True)

    assert calls[0]["level"] == cli.logging.INFO
    assert calls[1]["level"] == cli.logging.DEBUG


def test_main_bootstraps_and_runs_with_atlas_client(monkeypatch: pytest.MonkeyPatch) -> None:
    args = argparse.Namespace(
        region="us-west-2",
        profile="lsmc",
        host="127.0.0.1",
        port=8915,
        bootstrap_tapdb=True,
        reload=True,
        ssl_certfile="/tmp/cert.pem",
        ssl_keyfile="/tmp/key.pem",
        verbose=True,
    )
    monkeypatch.setattr(cli, "parse_args", lambda _argv=None: args)

    configured: list[bool] = []
    monkeypatch.setattr(cli, "configure_logging", lambda verbose: configured.append(verbose))

    settings = SimpleNamespace(
        bloom_base_url="https://bloom.example",
        bloom_api_token="bloom-token",
        bloom_verify_ssl=True,
        atlas_base_url="https://atlas.example",
        atlas_internal_api_key="atlas-key",
        atlas_verify_ssl=False,
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    class _Store:
        def __init__(self) -> None:
            self.bootstrapped = False

        def bootstrap(self) -> None:
            self.bootstrapped = True

    created_store: dict[str, _Store] = {}

    def _store_factory() -> _Store:
        store = _Store()
        created_store["store"] = store
        return store

    monkeypatch.setattr(cli, "AnalysisStore", _store_factory)

    monkeypatch.setattr(
        cli,
        "BloomResolverClient",
        lambda **kwargs: SimpleNamespace(kind="bloom", kwargs=kwargs),
    )
    monkeypatch.setattr(
        cli,
        "AtlasResultClient",
        lambda **kwargs: SimpleNamespace(kind="atlas", kwargs=kwargs),
    )

    app_inputs: dict[str, object] = {}

    def _create_app(store, **kwargs):
        app_inputs["store"] = store
        app_inputs.update(kwargs)
        return "fake-app"

    monkeypatch.setattr(cli, "create_app", _create_app)
    monkeypatch.setattr(
        cli,
        "_resolve_https_cert_paths",
        lambda _host, *, cert=None, key=None: (cert, key),
    )

    run_calls: list[dict[str, object]] = []

    def _fake_run(app, **kwargs):
        run_calls.append({"app": app, **kwargs})

    monkeypatch.setattr(cli.uvicorn, "run", _fake_run)

    rc = cli.main([])

    assert rc == 0
    assert configured == [True]
    assert created_store["store"].bootstrapped is True
    assert app_inputs["atlas_client"].kind == "atlas"
    assert run_calls[0]["app"] == "fake-app"
    assert run_calls[0]["host"] == "127.0.0.1"
    assert run_calls[0]["port"] == 8915
    assert run_calls[0]["reload"] is True
    assert run_calls[0]["ssl_certfile"] == "/tmp/cert.pem"
    assert run_calls[0]["ssl_keyfile"] == "/tmp/key.pem"
    assert run_calls[0]["log_level"] == "debug"


def test_main_skips_bootstrap_and_fails_without_atlas_key(monkeypatch: pytest.MonkeyPatch) -> None:
    args = argparse.Namespace(
        region="us-west-2",
        profile=None,
        host="0.0.0.0",
        port=8914,
        bootstrap_tapdb=False,
        reload=False,
        ssl_certfile=None,
        ssl_keyfile=None,
        verbose=False,
    )
    monkeypatch.setattr(cli, "parse_args", lambda _argv=None: args)
    monkeypatch.setattr(cli, "configure_logging", lambda _verbose: None)

    settings = SimpleNamespace(
        bloom_base_url="https://bloom.example",
        bloom_api_token="bloom-token",
        bloom_verify_ssl=False,
        atlas_base_url="https://atlas.example",
        atlas_internal_api_key="",
        atlas_verify_ssl=False,
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    class _Store:
        def __init__(self) -> None:
            self.bootstrap_called = False

        def bootstrap(self) -> None:
            self.bootstrap_called = True

    store = _Store()
    monkeypatch.setattr(cli, "AnalysisStore", lambda: store)
    monkeypatch.setattr(
        cli,
        "BloomResolverClient",
        lambda **kwargs: SimpleNamespace(kind="bloom", kwargs=kwargs),
    )

    monkeypatch.setattr(cli, "create_app", lambda *_args, **_kwargs: "fake-app")
    monkeypatch.setattr(cli.uvicorn, "run", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="ATLAS_INTERNAL_API_KEY"):
        cli.main([])

    assert store.bootstrap_called is False
