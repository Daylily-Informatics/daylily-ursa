from __future__ import annotations

import csv
from pathlib import Path

import pytest

from daylib_ursa import stage_samples as ss


def test_profile_s3_and_stage_target_validation():
    assert ss.ensure_profile("lsmc") == "lsmc"
    with pytest.raises(ss.CommandError, match="AWS profile is required"):
        ss.ensure_profile(None)

    assert ss.parse_s3_uri("s3://bucket/prefix/key") == ("bucket", "prefix/key")
    assert ss.parse_s3_uri("s3://bucket") == ("bucket", "")
    with pytest.raises(ss.CommandError, match="Expected an s3:// URI"):
        ss.parse_s3_uri("https://bucket/path")

    assert ss.normalise_stage_target("/data/staged_sample_data/") == "/data/staged_sample_data"
    with pytest.raises(ss.CommandError, match="expected to start with /data"):
        ss.normalise_stage_target("/tmp/not-fsx")


def test_build_stage_paths_and_env_helpers(monkeypatch):
    class _FakeDatetime:
        @classmethod
        def now(cls, _tz):
            class _Stamp:
                def strftime(self, _fmt):
                    return "20260309T010203Z"

            return _Stamp()

    monkeypatch.setattr(ss.dt, "datetime", _FakeDatetime)
    stage = ss.build_stage_paths("/data/staged_sample_data", "s3://ref-bucket/prefix")
    assert stage.remote_stage_name == "remote_stage_20260309T010203Z"
    assert stage.remote_fsx_stage == "/data/staged_sample_data/remote_stage_20260309T010203Z"
    assert (
        stage.remote_s3_stage
        == "s3://ref-bucket/prefix/data/staged_sample_data/remote_stage_20260309T010203Z"
    )

    env = ss.build_aws_env(ss.AwsConfig(profile="lsmc", region="us-west-2"))
    assert env["AWS_PROFILE"] == "lsmc"
    assert env["AWS_REGION"] == "us-west-2"


def test_subsample_and_row_normalization_helpers():
    assert ss.validate_subsample_pct("0.25") == "0.25"
    assert ss.validate_subsample_pct("0") == "na"
    assert ss.validate_subsample_pct("abc") == "na"
    assert ss.validate_subsample_pct("") == "na"

    assert ss.build_concordance_reference_uri("/fsx/path/to/conc", "s3://bucket/root") == "s3://bucket/root/path/to/conc"
    assert ss.determine_sex(2, 0) == "female"
    assert ss.determine_sex(1, 1) == "male"
    assert ss.determine_sex(0, 0) == "na"
    assert ss.safe_int("12") == 12
    assert ss.safe_int("x", default=7) == 7
    assert ss.normalise_identifier("a_b_c") == "a-b-c"

    rows = [
        {"A": "1", "B": "2"},
        {"A": "1", "B": "2"},
        {"A": "2", "B": "3"},
    ]
    deduped = ss.deduplicate_rows(rows, ["A", "B"])
    assert deduped == [{"A": "1", "B": "2"}, {"A": "2", "B": "3"}]

    units_rows = [{"ILMN_R1_PATH": "/data/x", "ILMN_R2_PATH": "/data"}]
    ss.normalise_units_paths(units_rows)
    assert units_rows == [{"ILMN_R1_PATH": "/fsx/data/x", "ILMN_R2_PATH": "/fsx/data"}]


def test_stage_single_lane_copies_two_files(monkeypatch):
    copied: list[tuple[str, str]] = []
    monkeypatch.setattr(
        ss,
        "aws_copy",
        lambda src, dst, **_kwargs: copied.append((src, dst)),
    )
    r1, r2 = ss.stage_single_lane(
        "s3://src/r1.fastq.gz",
        "s3://src/r2.fastq.gz",
        "/fsx/stage/SAMPLE",
        "s3://bucket/stage/SAMPLE",
        aws_env={},
        debug=False,
    )
    assert r1.endswith("/r1.fastq.gz")
    assert r2.endswith("/r2.fastq.gz")
    assert copied == [
        ("s3://src/r1.fastq.gz", "s3://bucket/stage/SAMPLE/r1.fastq.gz"),
        ("s3://src/r2.fastq.gz", "s3://bucket/stage/SAMPLE/r2.fastq.gz"),
    ]


def test_stage_multi_lane_runs_concat_and_cleanup(monkeypatch):
    def _fake_ensure(sources, **kwargs):  # noqa: ANN001
        if "R1" in kwargs["sample_prefix"]:
            return list(sources), ["s3://uploaded/r1"]
        return list(sources), ["s3://uploaded/r2"]

    concat_calls: list[tuple[tuple[str, ...], str]] = []
    cleanup_calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(ss, "ensure_s3_objects", _fake_ensure)
    monkeypatch.setattr(
        ss,
        "multipart_concatenate",
        lambda srcs, dst, **_kwargs: concat_calls.append((tuple(srcs), dst)),
    )
    monkeypatch.setattr(
        ss,
        "cleanup_s3_objects",
        lambda uris, **_kwargs: cleanup_calls.append(tuple(uris)),
    )

    r1, r2 = ss.stage_multi_lane(
        ["s3://in/r1_1.fastq.gz", "s3://in/r1_2.fastq.gz"],
        ["s3://in/r2_1.fastq.gz", "s3://in/r2_2.fastq.gz"],
        "sample-prefix",
        "/fsx/stage/SAMPLE",
        "s3://bucket/stage/SAMPLE",
        aws_env={},
        debug=False,
    )

    assert r1.endswith("sample-prefix_merged_R1.fastq.gz")
    assert r2.endswith("sample-prefix_merged_R2.fastq.gz")
    assert len(concat_calls) == 2
    assert cleanup_calls == [("s3://uploaded/r1",), ("s3://uploaded/r2",)]


def test_stage_multi_lane_cleans_r1_uploads_when_r2_prepare_fails(monkeypatch):
    calls = {"count": 0}
    cleanup_calls: list[tuple[str, ...]] = []

    def _fake_ensure(_sources, **kwargs):  # noqa: ANN001
        calls["count"] += 1
        if "R1" in kwargs["sample_prefix"]:
            return ["s3://in/r1"], ["s3://uploaded/r1"]
        raise RuntimeError("boom")

    monkeypatch.setattr(ss, "ensure_s3_objects", _fake_ensure)
    monkeypatch.setattr(
        ss,
        "cleanup_s3_objects",
        lambda uris, **_kwargs: cleanup_calls.append(tuple(uris)),
    )
    with pytest.raises(RuntimeError, match="boom"):
        ss.stage_multi_lane(
            ["s3://in/r1.fastq.gz"],
            ["s3://in/r2.fastq.gz"],
            "sample-prefix",
            "/fsx/stage/SAMPLE",
            "s3://bucket/stage/SAMPLE",
            aws_env={},
            debug=False,
        )
    assert cleanup_calls == [("s3://uploaded/r1",)]


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
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_process_samples_single_lane_builds_units_and_samples(monkeypatch, tmp_path: Path):
    input_tsv = tmp_path / "analysis_samples.tsv"
    _write_analysis_samples(
        input_tsv,
        [
            {
                ss.RUN_ID: "RUN_1",
                ss.SAMPLE_ID: "SAMPLE_1",
                ss.EXPERIMENT_ID: "EXP_1",
                ss.SAMPLE_TYPE: "blood",
                ss.LIB_PREP: "wgs",
                ss.SEQ_VENDOR: "ILMN",
                ss.SEQ_PLATFORM: "novaseq",
                ss.LANE: "1",
                ss.SEQBC_ID: "BC01",
                ss.PATH_TO_CONCORDANCE: "/fsx/concordance/path",
                ss.R1_FQ: "/tmp/r1.fastq.gz",
                ss.R2_FQ: "/tmp/r2.fastq.gz",
                ss.SUBSAMPLE_PCT: "0.25",
                ss.IS_POS_CTRL: "false",
                ss.IS_NEG_CTRL: "false",
                ss.N_X: "2",
                ss.N_Y: "0",
                ss.EXTERNAL_SAMPLE_ID: "EXT-1",
            }
        ],
    )

    monkeypatch.setattr(ss, "validate_sources", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        ss,
        "stage_single_lane",
        lambda *_args, **_kwargs: ("/fsx/data/stage/r1.fastq.gz", "/fsx/data/stage/r2.fastq.gz"),
    )
    monkeypatch.setattr(
        ss,
        "stage_concordance",
        lambda *_args, **_kwargs: "/fsx/data/stage/concordance_data",
    )

    stage = ss.StagePaths(
        remote_fsx_root="/data/staged_sample_data",
        remote_stage_name="remote_stage_20260309T010203Z",
        remote_fsx_stage="/data/staged_sample_data/remote_stage_20260309T010203Z",
        remote_s3_stage="s3://bucket/stage/remote_stage_20260309T010203Z",
    )
    samples, units, created_files, run_ids = ss.process_samples(
        input_tsv,
        stage,
        reference_bucket="s3://ref-bucket/base",
        aws_env={},
        debug=False,
    )

    assert run_ids == ["RUN-1"]
    assert created_files == ["/fsx/data/stage/r1.fastq.gz", "/fsx/data/stage/r2.fastq.gz"]
    assert samples[0]["SAMPLEID"] == "SAMPLE-1"
    assert samples[0]["BIOLOGICAL_SEX"] == "female"
    assert units[0]["RUNID"] == "RUN-1"
    assert units[0]["ILMN_R1_PATH"] == "/fsx/data/stage/r1.fastq.gz"
    assert units[0]["ILMN_R2_PATH"] == "/fsx/data/stage/r2.fastq.gz"


def test_process_samples_rejects_multi_lane_with_lane_zero(monkeypatch, tmp_path: Path):
    input_tsv = tmp_path / "analysis_samples.tsv"
    _write_analysis_samples(
        input_tsv,
        [
            {
                ss.RUN_ID: "RUN_1",
                ss.SAMPLE_ID: "SAMPLE_1",
                ss.EXPERIMENT_ID: "EXP_1",
                ss.SAMPLE_TYPE: "blood",
                ss.LIB_PREP: "wgs",
                ss.SEQ_VENDOR: "ILMN",
                ss.SEQ_PLATFORM: "novaseq",
                ss.LANE: "0",
                ss.SEQBC_ID: "BC01",
                ss.PATH_TO_CONCORDANCE: "na",
                ss.R1_FQ: "/tmp/r1_1.fastq.gz",
                ss.R2_FQ: "/tmp/r2_1.fastq.gz",
                ss.SUBSAMPLE_PCT: "na",
                ss.IS_POS_CTRL: "false",
                ss.IS_NEG_CTRL: "false",
                ss.N_X: "1",
                ss.N_Y: "1",
                ss.EXTERNAL_SAMPLE_ID: "EXT-1",
            },
            {
                ss.RUN_ID: "RUN_1",
                ss.SAMPLE_ID: "SAMPLE_1",
                ss.EXPERIMENT_ID: "EXP_1",
                ss.SAMPLE_TYPE: "blood",
                ss.LIB_PREP: "wgs",
                ss.SEQ_VENDOR: "ILMN",
                ss.SEQ_PLATFORM: "novaseq",
                ss.LANE: "2",
                ss.SEQBC_ID: "BC01",
                ss.PATH_TO_CONCORDANCE: "na",
                ss.R1_FQ: "/tmp/r1_2.fastq.gz",
                ss.R2_FQ: "/tmp/r2_2.fastq.gz",
                ss.SUBSAMPLE_PCT: "na",
                ss.IS_POS_CTRL: "false",
                ss.IS_NEG_CTRL: "false",
                ss.N_X: "1",
                ss.N_Y: "1",
                ss.EXTERNAL_SAMPLE_ID: "EXT-1",
            },
        ],
    )
    monkeypatch.setattr(ss, "validate_sources", lambda *_args, **_kwargs: None)
    stage = ss.StagePaths(
        remote_fsx_root="/data/staged_sample_data",
        remote_stage_name="remote_stage_20260309T010203Z",
        remote_fsx_stage="/data/staged_sample_data/remote_stage_20260309T010203Z",
        remote_s3_stage="s3://bucket/stage/remote_stage_20260309T010203Z",
    )

    with pytest.raises(ss.CommandError, match="Invalid LANE=0 for multi-lane sample"):
        ss.process_samples(
            input_tsv,
            stage,
            reference_bucket="s3://ref-bucket/base",
            aws_env={},
            debug=False,
        )
