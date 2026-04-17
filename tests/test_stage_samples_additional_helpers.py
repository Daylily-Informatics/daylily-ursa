from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from daylib_ursa import stage_samples as ss


def _write_analysis_samples(path: Path, rows: list[dict[str, str]]) -> None:
    header = [
        ss.RUN_ID,
        ss.SAMPLE_ID,
        ss.EXPERIMENT_ID,
        ss.SAMPLE_TYPE,
        ss.LIB_PREP,
        ss.SEQ_VENDOR,
        ss.SEQ_PLATFORM,
        ss.LANE,
        ss.SEQBC_ID,
        ss.PATH_TO_CONCORDANCE,
        ss.R1_FQ,
        ss.R2_FQ,
        ss.SUBSAMPLE_PCT,
        ss.IS_POS_CTRL,
        ss.IS_NEG_CTRL,
        ss.N_X,
        ss.N_Y,
        ss.EXTERNAL_SAMPLE_ID,
        "DEEP_MODEL",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_parse_args_uses_defaults_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_PROFILE", "profile-from-env")
    monkeypatch.setenv("AWS_REGION", "us-west-2")

    args = ss.parse_args(["analysis.tsv", "--reference-bucket", "s3://ref-bucket/base", "--debug"])

    assert args.analysis_samples == "analysis.tsv"
    assert args.reference_bucket == "s3://ref-bucket/base"
    assert args.stage_target == "/data/staged_sample_data"
    assert args.profile == "profile-from-env"
    assert args.region == "us-west-2"
    assert args.debug is True


def test_build_stage_paths_without_prefix_uses_bucket_root(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeDatetime:
        @classmethod
        def now(cls, _tz):
            class _Stamp:
                def strftime(self, _fmt):
                    return "20260309T020304Z"

            return _Stamp()

    monkeypatch.setattr(ss.dt, "datetime", _FakeDatetime)

    stage = ss.build_stage_paths("/data/staged_sample_data", "s3://bucket")

    assert stage.remote_stage_name == "remote_stage_20260309T020304Z"
    assert (
        stage.remote_s3_stage == "s3://bucket/data/staged_sample_data/remote_stage_20260309T020304Z"
    )


def test_run_command_wraps_called_process_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_args, **_kwargs):
        raise ss.subprocess.CalledProcessError(
            returncode=5,
            cmd=["aws", "s3", "ls"],
            output="stdout text",
            stderr="stderr text",
        )

    monkeypatch.setattr(ss.subprocess, "run", _raise)

    with pytest.raises(ss.CommandError) as exc:
        ss.run_command(["aws", "s3", "ls"], env={"A": "1"})

    msg = str(exc.value)
    assert "Command failed (5): aws s3 ls" in msg
    assert "STDOUT" in msg and "stdout text" in msg
    assert "STDERR" in msg and "stderr text" in msg


def test_aws_command_debug_prefixes_aws_and_prints(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    captured: dict[str, object] = {}

    def _fake_run_command(command, *, env=None, capture_output=False):
        captured["command"] = command
        captured["env"] = env
        captured["capture_output"] = capture_output
        return SimpleNamespace(stdout="ok")

    monkeypatch.setattr(ss, "run_command", _fake_run_command)

    result = ss.aws_command(
        ["s3", "ls", "s3://bucket"], aws_env={"AWS_PROFILE": "x"}, debug=True, capture_output=True
    )

    assert result.stdout == "ok"
    assert captured["command"] == ["aws", "s3", "ls", "s3://bucket"]
    assert captured["capture_output"] is True
    assert "[DEBUG] aws s3 ls s3://bucket" in capsys.readouterr().out


def test_check_path_helpers_and_validate_sources_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ss.CommandError, match="Local path not found"):
        ss.check_local_path("/path/does/not/exist")

    monkeypatch.setattr(ss, "aws_command", lambda *_args, **_kwargs: SimpleNamespace(stdout=""))
    with pytest.raises(ss.CommandError, match="not accessible"):
        ss.check_s3_path("s3://bucket/prefix", aws_env={}, debug=False)

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(ss, "check_s3_path", lambda uri, **_kwargs: calls.append(("s3", uri)))
    monkeypatch.setattr(
        ss, "check_concordance_path", lambda path, **_kwargs: calls.append(("conc", path))
    )
    monkeypatch.setattr(ss, "check_local_path", lambda path: calls.append(("local", path)))

    ss.validate_sources(
        [
            ("na", False),
            ("s3://bucket/key", False),
            ("/fsx/path/conc", True),
            ("~/local/file", False),
        ],
        reference_bucket="s3://ref/base",
        aws_env={},
        debug=False,
    )

    assert calls[0] == ("s3", "s3://bucket/key")
    assert calls[1] == ("conc", "/fsx/path/conc")
    assert calls[2][0] == "local"
    assert calls[2][1].endswith("/local/file")


def test_check_concordance_path_routes_fsx_and_local(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(ss, "check_s3_path", lambda uri, **_kwargs: calls.append(("s3", uri)))
    monkeypatch.setattr(ss, "check_local_path", lambda path: calls.append(("local", path)))

    ss.check_concordance_path(
        "/fsx/path/to/concordance",
        reference_bucket="s3://ref-bucket/base",
        aws_env={},
        debug=False,
    )
    ss.check_concordance_path(
        "~/local-concordance",
        reference_bucket="s3://ref-bucket/base",
        aws_env={},
        debug=False,
    )

    assert calls[0] == ("s3", "s3://ref-bucket/base/path/to/concordance")
    assert calls[1][0] == "local"


def test_ensure_remote_stage_writable_uploads_and_cleans_temp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def _fake_aws_command(args, **_kwargs):
        commands.append(list(args))
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(ss, "aws_command", _fake_aws_command)

    stage = ss.StagePaths(
        remote_fsx_root="/data/staged_sample_data",
        remote_stage_name="remote_stage_20260309T020304Z",
        remote_fsx_stage="/data/staged_sample_data/remote_stage_20260309T020304Z",
        remote_s3_stage="s3://bucket/stage/remote_stage_20260309T020304Z",
    )

    ss.ensure_remote_stage_writable(stage, aws_env={}, debug=False)

    assert commands[0][0:2] == ["s3", "cp"]
    assert commands[1] == [
        "s3",
        "rm",
        "s3://bucket/stage/remote_stage_20260309T020304Z/_write_test.txt",
    ]


def test_aws_copy_and_s3_object_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    aws_calls: list[list[str]] = []
    cleanup_calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        ss,
        "aws_command",
        lambda args, **_kwargs: aws_calls.append(list(args)) or SimpleNamespace(stdout=""),
    )

    ss.aws_copy("source", "dest", aws_env={}, debug=False, recursive=True)
    assert aws_calls[-1] == ["s3", "cp", "source", "dest", "--recursive"]

    def _copy_once_then_fail(_source, _destination, **_kwargs):
        if not cleanup_calls:
            cleanup_calls.append(("uploaded",))
        raise RuntimeError("copy failed")

    uploaded_for_cleanup: list[tuple[str, ...]] = []
    monkeypatch.setattr(ss, "aws_copy", _copy_once_then_fail)
    monkeypatch.setattr(
        ss,
        "cleanup_s3_objects",
        lambda uris, **_kwargs: uploaded_for_cleanup.append(tuple(uris)),
    )

    with pytest.raises(RuntimeError, match="copy failed"):
        ss.ensure_s3_objects(
            ["/tmp/a.fastq.gz"],
            dest_s3_dir="s3://bucket/stage",
            sample_prefix="sample",
            aws_env={},
            debug=False,
        )

    assert uploaded_for_cleanup == [tuple()]


def test_ensure_s3_objects_happy_path_and_cleanup_ignores_rm_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied: list[tuple[str, str]] = []

    monkeypatch.setattr(ss.uuid, "uuid4", lambda: SimpleNamespace(hex="abcdef"))
    monkeypatch.setattr(
        ss,
        "aws_copy",
        lambda src, dst, **_kwargs: copied.append((src, dst)),
    )

    resolved, uploaded = ss.ensure_s3_objects(
        ["s3://bucket/r1.fastq.gz", "~/reads/r2.fastq.gz"],
        dest_s3_dir="s3://bucket/stage",
        sample_prefix="sample",
        aws_env={},
        debug=False,
    )

    assert resolved[0] == "s3://bucket/r1.fastq.gz"
    assert resolved[1].startswith("s3://bucket/stage/_parts/sample_part2_abcdef_")
    assert uploaded == [resolved[1]]
    assert copied[0][0].endswith("/reads/r2.fastq.gz")

    rm_calls: list[str] = []

    def _rm_with_error(args, **_kwargs):
        uri = args[-1]
        rm_calls.append(uri)
        if uri.endswith("one"):
            raise ss.CommandError("ignore")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(ss, "aws_command", _rm_with_error)
    ss.cleanup_s3_objects(["s3://bucket/one", "s3://bucket/two"], aws_env={}, debug=False)
    assert rm_calls == ["s3://bucket/one", "s3://bucket/two"]


def test_multipart_concatenate_success_and_abort_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def _aws_ok(args, **_kwargs):
        calls.append(list(args))
        if args[:2] == ["s3api", "create-multipart-upload"]:
            return SimpleNamespace(stdout=json.dumps({"UploadId": "upload-1"}))
        if args[:2] == ["s3api", "upload-part-copy"]:
            part_num = args[args.index("--part-number") + 1]
            return SimpleNamespace(
                stdout=json.dumps({"CopyPartResult": {"ETag": f"etag-{part_num}"}})
            )
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(ss, "aws_command", _aws_ok)

    ss.multipart_concatenate(
        ["s3://source/a", "s3://source/b"],
        "s3://dest-bucket/path/key",
        aws_env={},
        debug=False,
    )

    assert any(cmd[:2] == ["s3api", "complete-multipart-upload"] for cmd in calls)

    abort_calls: list[list[str]] = []

    def _aws_missing_etag(args, **_kwargs):
        abort_calls.append(list(args))
        if args[:2] == ["s3api", "create-multipart-upload"]:
            return SimpleNamespace(stdout=json.dumps({"UploadId": "upload-2"}))
        if args[:2] == ["s3api", "upload-part-copy"]:
            return SimpleNamespace(stdout=json.dumps({"CopyPartResult": {}}))
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(ss, "aws_command", _aws_missing_etag)

    with pytest.raises(ss.CommandError, match="Failed to copy part"):
        ss.multipart_concatenate(
            ["s3://source/a"], "s3://dest-bucket/path/key", aws_env={}, debug=False
        )

    assert any(cmd[:2] == ["s3api", "abort-multipart-upload"] for cmd in abort_calls)

    with pytest.raises(ss.CommandError, match="No sources provided"):
        ss.multipart_concatenate([], "s3://dest-bucket/path/key", aws_env={}, debug=False)


def test_stage_concordance_covers_all_source_shapes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str, bool]] = []
    monkeypatch.setattr(
        ss,
        "aws_copy",
        lambda src, dst, *, recursive=False, **_kwargs: calls.append((src, dst, recursive)),
    )

    local_dir = tmp_path / "conc_dir"
    local_dir.mkdir()
    local_file = tmp_path / "conc_file.txt"
    local_file.write_text("x", encoding="utf-8")

    assert (
        ss.stage_concordance("na", "/fsx/ignored", "s3://ignored", aws_env={}, debug=False) == "na"
    )
    assert (
        ss.stage_concordance(
            "/fsx/data/existing", "/fsx/ignored", "s3://ignored", aws_env={}, debug=False
        )
        == "/fsx/data/existing"
    )
    assert (
        ss.stage_concordance(
            "s3://src/conc", "/fsx/target", "s3://dst/conc", aws_env={}, debug=False
        )
        == "/fsx/target"
    )
    assert (
        ss.stage_concordance(
            str(local_dir), "/fsx/target", "s3://dst/conc", aws_env={}, debug=False
        )
        == "/fsx/target"
    )
    assert (
        ss.stage_concordance(
            str(local_file), "/fsx/target", "s3://dst/conc", aws_env={}, debug=False
        )
        == "/fsx/target"
    )

    assert calls[0] == ("s3://src/conc", "s3://dst/conc", True)
    assert calls[1][2] is True
    assert calls[2][2] is False


def test_write_tsv_and_process_samples_vendor_and_conflict_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "nested" / "units.tsv"
    ss.write_tsv(output, ["A", "B"], [{"A": "1", "B": "2"}])
    assert output.exists()
    assert "A\tB" in output.read_text(encoding="utf-8")

    input_tsv = tmp_path / "analysis_samples.tsv"
    _write_analysis_samples(
        input_tsv,
        [
            {
                ss.RUN_ID: "RUN_A",
                ss.SAMPLE_ID: "SAMPLE_1",
                ss.EXPERIMENT_ID: "EXP_1",
                ss.SAMPLE_TYPE: "blood",
                ss.LIB_PREP: "wgs",
                ss.SEQ_VENDOR: "ONT",
                ss.SEQ_PLATFORM: "promethion",
                ss.LANE: "7",
                ss.SEQBC_ID: "BC01",
                ss.PATH_TO_CONCORDANCE: "na",
                ss.R1_FQ: "/tmp/r1.fastq.gz",
                ss.R2_FQ: "/tmp/r2.fastq.gz",
                ss.SUBSAMPLE_PCT: "0.1",
                ss.IS_POS_CTRL: "true",
                ss.IS_NEG_CTRL: "false",
                ss.N_X: "1",
                ss.N_Y: "1",
                ss.EXTERNAL_SAMPLE_ID: "EXT-1",
                "DEEP_MODEL": "model-a",
            }
        ],
    )

    monkeypatch.setattr(ss, "validate_sources", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ss, "stage_single_lane", lambda *_args, **_kwargs: ("/fsx/r1", "/fsx/r2"))
    monkeypatch.setattr(ss, "stage_concordance", lambda *_args, **_kwargs: "/existing/conc")

    stage = ss.StagePaths(
        remote_fsx_root="/data/staged_sample_data",
        remote_stage_name="remote_stage_20260309T020304Z",
        remote_fsx_stage="/data/staged_sample_data/remote_stage_20260309T020304Z",
        remote_s3_stage="s3://bucket/stage/remote_stage_20260309T020304Z",
    )

    samples, units, created, run_ids = ss.process_samples(
        input_tsv,
        stage,
        reference_bucket="s3://ref-bucket/base",
        aws_env={},
        debug=False,
    )

    assert run_ids == ["RUN-A"]
    assert created == ["/fsx/r1", "/fsx/r2"]
    assert samples[0]["BIOLOGICAL_SEX"] == "male"
    assert units[0]["ONT_R1_PATH"] == "/fsx/r1"
    assert units[0]["DEEP_MODEL"] == "model-a"
    assert units[0]["SAMPLEUSE"] == "posControl"

    conflict_tsv = tmp_path / "analysis_samples_conflict.tsv"
    _write_analysis_samples(
        conflict_tsv,
        [
            {
                ss.RUN_ID: "RUN_A",
                ss.SAMPLE_ID: "SAMPLE_9",
                ss.EXPERIMENT_ID: "EXP_1",
                ss.SAMPLE_TYPE: "blood",
                ss.LIB_PREP: "wgs",
                ss.SEQ_VENDOR: "ILMN",
                ss.SEQ_PLATFORM: "nova",
                ss.LANE: "1",
                ss.SEQBC_ID: "BC01",
                ss.PATH_TO_CONCORDANCE: "na",
                ss.R1_FQ: "/tmp/a_r1.fastq.gz",
                ss.R2_FQ: "/tmp/a_r2.fastq.gz",
                ss.SUBSAMPLE_PCT: "na",
                ss.IS_POS_CTRL: "false",
                ss.IS_NEG_CTRL: "false",
                ss.N_X: "1",
                ss.N_Y: "1",
                ss.EXTERNAL_SAMPLE_ID: "EXT-9",
                "DEEP_MODEL": "",
            },
            {
                ss.RUN_ID: "RUN_B",
                ss.SAMPLE_ID: "SAMPLE_9",
                ss.EXPERIMENT_ID: "EXP_2",
                ss.SAMPLE_TYPE: "blood",
                ss.LIB_PREP: "wgs",
                ss.SEQ_VENDOR: "ILMN",
                ss.SEQ_PLATFORM: "nova",
                ss.LANE: "2",
                ss.SEQBC_ID: "BC01",
                ss.PATH_TO_CONCORDANCE: "na",
                ss.R1_FQ: "/tmp/b_r1.fastq.gz",
                ss.R2_FQ: "/tmp/b_r2.fastq.gz",
                ss.SUBSAMPLE_PCT: "na",
                ss.IS_POS_CTRL: "false",
                ss.IS_NEG_CTRL: "false",
                ss.N_X: "2",
                ss.N_Y: "0",
                ss.EXTERNAL_SAMPLE_ID: "EXT-9",
                "DEEP_MODEL": "",
            },
        ],
    )

    with pytest.raises(ss.CommandError, match="Duplicate SAMPLEID with conflicting metadata"):
        ss.process_samples(
            conflict_tsv,
            stage,
            reference_bucket="s3://ref-bucket/base",
            aws_env={},
            debug=False,
        )


def test_process_samples_required_columns_and_blank_header_paths(tmp_path: Path) -> None:
    no_header = tmp_path / "no_header.tsv"
    no_header.write_text("", encoding="utf-8")

    stage = ss.StagePaths(
        remote_fsx_root="/data/staged_sample_data",
        remote_stage_name="remote_stage_20260309T020304Z",
        remote_fsx_stage="/data/staged_sample_data/remote_stage_20260309T020304Z",
        remote_s3_stage="s3://bucket/stage/remote_stage_20260309T020304Z",
    )

    with pytest.raises(ss.CommandError, match="missing a header row"):
        ss.process_samples(
            no_header, stage, reference_bucket="s3://bucket/ref", aws_env={}, debug=False
        )

    missing_cols = tmp_path / "missing_cols.tsv"
    missing_cols.write_text("RUN_ID\tSAMPLE_ID\nA\tB\n", encoding="utf-8")

    with pytest.raises(ss.CommandError, match="Missing required columns"):
        ss.process_samples(
            missing_cols,
            stage,
            reference_bucket="s3://bucket/ref",
            aws_env={},
            debug=False,
        )


def test_main_missing_input_and_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    missing_args = argparse.Namespace(
        analysis_samples=str(tmp_path / "missing.tsv"),
        stage_target="/data/staged_sample_data",
        reference_bucket="s3://bucket/ref",
        config_dir=None,
        profile="lsmc",
        region="us-west-2",
        debug=False,
    )
    monkeypatch.setattr(ss, "parse_args", lambda _argv=None: missing_args)

    with pytest.raises(ss.CommandError, match="not found"):
        ss.main([])

    input_tsv = tmp_path / "analysis_samples.tsv"
    input_tsv.write_text("header\n", encoding="utf-8")

    present_args = argparse.Namespace(
        analysis_samples=str(input_tsv),
        stage_target="/data/staged_sample_data",
        reference_bucket="s3://bucket/ref",
        config_dir=str(tmp_path / "cfg"),
        profile="lsmc",
        region="us-west-2",
        debug=False,
    )
    monkeypatch.setattr(ss, "parse_args", lambda _argv=None: present_args)
    monkeypatch.setattr(ss, "ensure_profile", lambda profile: profile)
    monkeypatch.setattr(ss, "build_aws_env", lambda _cfg: {"AWS_PROFILE": "lsmc"})

    stage = ss.StagePaths(
        remote_fsx_root="/data/staged_sample_data",
        remote_stage_name="remote_stage_20260309T040506Z",
        remote_fsx_stage="/data/staged_sample_data/remote_stage_20260309T040506Z",
        remote_s3_stage="s3://bucket/stage/remote_stage_20260309T040506Z",
    )
    monkeypatch.setattr(ss, "build_stage_paths", lambda *_args, **_kwargs: stage)
    monkeypatch.setattr(ss, "ensure_remote_stage_writable", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        ss,
        "process_samples",
        lambda *_args, **_kwargs: (
            [{"SAMPLEID": "S1"}],
            [{"RUNID": "RUN1", "ILMN_R1_PATH": "/data/r1", "ILMN_R2_PATH": "/data/r2"}],
            ["/data/r1", "/data/r2"],
            ["RUN1"],
        ),
    )
    monkeypatch.setattr(ss, "deduplicate_rows", lambda rows, _header: list(rows))
    monkeypatch.setattr(ss, "normalise_units_paths", lambda _rows: None)

    writes: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        ss,
        "write_tsv",
        lambda path, _header, rows: writes.append((str(path), [str(r) for r in rows])),
    )

    copies: list[tuple[str, str]] = []
    monkeypatch.setattr(
        ss,
        "aws_copy",
        lambda src, dst, **_kwargs: copies.append((src, dst)),
    )

    rc = ss.main([])

    assert rc == 0
    assert any(path.endswith("20260309T040506Z_samples.tsv") for path, _ in writes)
    assert any(path.endswith("20260309T040506Z_units.tsv") for path, _ in writes)
    assert copies[0][1].endswith("20260309T040506Z_samples.tsv")
    assert copies[1][1].endswith("20260309T040506Z_units.tsv")

    output = capsys.readouterr().out
    assert "Remote staging completed successfully." in output
    assert "Generated configuration files:" in output
