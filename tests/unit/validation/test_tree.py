# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tree walking, artifact loading, and the scopes mapping.

Fixtures are one small builder (`_tree`) writing tiny artifacts on demand, not
stored files. Cases assert what each subject computes: `from_directory` the
walk shape and node classes, `read` its (value, error) with `MISSING`, `scopes`
its mapping.
"""

import json
import os

import pytest
from inference_endpoint.validation.catalog import Catalog
from inference_endpoint.validation.operations import Absent, Predicate
from inference_endpoint.validation.ranges import Interval
from inference_endpoint.validation.refs import Ref
from inference_endpoint.validation.regions import Regions
from inference_endpoint.validation.sentinels import MISSING
from inference_endpoint.validation.submission import Point
from inference_endpoint.validation.tree import (
    Node,
    ParetoNode,
    PointNode,
    RuleScopes,
    SubmissionNode,
    SystemNode,
)


def _tree(tmp_path, point_artifacts=None, system_artifacts=None):
    """One org-less submission: results/sys/model/r16 (+ r8), optional artifacts."""
    root = tmp_path / "acme"
    point = root / "results" / "sys" / "model" / "r16"
    point.mkdir(parents=True)
    (root / "results" / "sys" / "model" / "r8").mkdir()
    for name, value in (point_artifacts or {}).items():
        _write(point / name, value)
    for name, value in (system_artifacts or {}).items():
        _write(root / "results" / "sys" / name, value)
    return SubmissionNode.from_directory(root), root


def _write(path, value):
    """String values are written verbatim (for malformed content). Others are
    written as JSON."""
    path.write_text(value if isinstance(value, str) else json.dumps(value))


class TestWalk:
    """`SubmissionNode.from_directory`: kinds, classes, order, parentage, and the
    refusal of entries no kind can take."""

    @pytest.mark.unit
    def test_walk_shape(self, tmp_path):
        node, _ = _tree(tmp_path)
        assert isinstance(node, SubmissionNode)
        system = node.children[0]
        assert type(system) is SystemNode
        assert type(system.children[0]) is ParetoNode
        points = system.children[0].children
        assert [type(p) for p in points] == [PointNode, PointNode]
        assert [p.dir.name for p in points] == ["r16", "r8"]
        assert points[1].parent is system.children[0]
        assert points[1].parent.parent.parent is node

    @pytest.mark.unit
    def test_node_construction_follows_the_kind(self):
        """`Node(kind, ...)` yields the registered class. The registry is keyed on kind."""
        assert Node.registry == {
            RuleScopes.SUBMISSION: SubmissionNode,
            RuleScopes.SYSTEM: SystemNode,
            RuleScopes.PARETO: ParetoNode,
            RuleScopes.POINT: PointNode,
        }

    @pytest.mark.unit
    def test_a_directory_that_is_not_a_point_is_refused(self, tmp_path):
        _, root = _tree(tmp_path)
        (root / "results" / "sys" / "model" / "notes").mkdir()
        with pytest.raises(
            ValueError, match=r"no check can reach: notes \(not an `r<N>` point"
        ):
            SubmissionNode.from_directory(root)

    @pytest.mark.unit
    def test_a_point_named_file_is_refused(self, tmp_path):
        _, root = _tree(tmp_path)
        (root / "results" / "sys" / "model" / "r32").write_text("{}", encoding="utf-8")
        with pytest.raises(
            ValueError, match=r"no check can reach: r32 \(a point is a directory\)"
        ):
            SubmissionNode.from_directory(root)

    @pytest.mark.unit
    def test_a_symlinked_directory_is_refused_not_skipped(self, tmp_path):
        _, root = _tree(tmp_path)
        outside = tmp_path / "outside"
        outside.mkdir()
        (root / "results" / "sys" / "model" / "r99").symlink_to(
            outside, target_is_directory=True
        )
        with pytest.raises(
            ValueError, match=r"no check can reach: r99 \(a symlinked directory\)"
        ):
            SubmissionNode.from_directory(root)

    @pytest.mark.unit
    def test_a_stray_file_beside_the_points_is_ignored(self, tmp_path):
        _, root = _tree(tmp_path)
        (root / "results" / "sys" / "model" / "notes.txt").write_text(
            "run notes\n", encoding="utf-8"
        )
        node = SubmissionNode.from_directory(root)
        assert [p.dir.name for p in node.children[0].children[0].children] == [
            "r16",
            "r8",
        ]

    @pytest.mark.unit
    def test_missing_root_is_a_load_error(self, tmp_path):
        with pytest.raises(
            NotADirectoryError, match="the submission directory .* does not exist"
        ):
            SubmissionNode.from_directory(tmp_path / "nope")

    @pytest.mark.unit
    def test_a_file_where_the_upload_belongs_is_a_load_error(self, tmp_path):
        path = tmp_path / "upload.json"
        path.write_text("{}\n", encoding="utf-8")
        with pytest.raises(NotADirectoryError, match="is not a directory"):
            SubmissionNode.from_directory(path)


class TestRead:
    """`Node.read`: (value, error) over the eagerly loaded artifacts."""

    def _point(self, tmp_path, artifacts=None):
        node, _ = _tree(tmp_path, artifacts)
        return node.children[0].children[0].children[0]

    @pytest.mark.unit
    def test_json_artifact_reads(self, tmp_path):
        point = self._point(tmp_path, {"result_summary.json": {"n": 1}})
        assert point.read("result_summary.json") == ({"n": 1}, None)

    @pytest.mark.unit
    def test_the_clients_tps_surfaces_as_system_tps(self, tmp_path):
        point = self._point(
            tmp_path, {"result_summary.json": {"tps": 6000.0, "qps": 12.5}}
        )
        value, error = point.read("result_summary.json")
        assert error is None
        assert value["system_tps"] == 6000.0
        assert value["tps"] == 6000.0 and value["qps"] == 12.5

    def _video_point(self, tmp_path, summary, model="wan-2.2-t2v-a14b"):
        root = tmp_path / "acme"
        point_dir = root / "results" / "sys" / model / "r8"
        point_dir.mkdir(parents=True)
        (point_dir / "result_summary.json").write_text(json.dumps(summary))
        (point_dir / "point.yaml").write_text(f"model_name: {model}\n")
        catalog = Catalog(
            {"wan-2.2-t2v-a14b": {"frames_per_sample": 81}, "llm": {}}, {}
        )
        node = SubmissionNode.from_directory(root, ruleset=catalog)
        return node.children[0].children[0].children[0]

    @pytest.mark.unit
    def test_a_video_run_reads_its_system_fps_in_frames_per_second(self, tmp_path):
        point = self._video_point(
            tmp_path,
            {"tps": None, "qps": 1.0, "n_samples_completed": 4, "n_samples_failed": 0},
        )
        value, error = point.read("result_summary.json")
        assert error is None
        assert value["system_fps"] == 81.0
        assert value["full_run_system_fps"] == 81.0
        assert "system_tps" not in value
        assert "full_run_system_tps" not in value
        assert value["n_samples_succeeded"] == 4
        assert value["tps"] is None

    @pytest.mark.unit
    def test_failed_clips_add_no_frames(self, tmp_path):
        # qps counts failed samples; only the 3 generated clips of 4 give frames.
        point = self._video_point(
            tmp_path,
            {"tps": None, "qps": 1.0, "n_samples_completed": 4, "n_samples_failed": 1},
        )
        value, _ = point.read("result_summary.json")
        assert value["system_fps"] == pytest.approx(0.75 * 81)
        assert value["n_samples_succeeded"] == 3

    @pytest.mark.unit
    def test_a_video_run_without_qps_gets_no_system_fps(self, tmp_path):
        point = self._video_point(
            tmp_path,
            {"tps": None, "qps": None, "n_samples_completed": 4, "n_samples_failed": 0},
        )
        value, _ = point.read("result_summary.json")
        assert "system_fps" not in value

    @pytest.mark.unit
    def test_a_video_run_without_sample_counts_gets_no_system_fps(self, tmp_path):
        point = self._video_point(tmp_path, {"tps": None, "qps": 1.0})
        value, _ = point.read("result_summary.json")
        assert "system_fps" not in value

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "uploaded",
        [
            {"system_fps": 1.0, "full_run_system_fps": 1.0},
            {"system_tps": 1.0, "full_run_system_tps": 1.0},
            {"tps": 3.0},
        ],
    )
    def test_the_frame_rate_replaces_an_uploaded_throughput(self, tmp_path, uploaded):
        summary = {"qps": 1.0, "n_samples_completed": 4, "n_samples_failed": 0}
        point = self._video_point(tmp_path, {**summary, **uploaded})
        value, _ = point.read("result_summary.json")
        assert value["system_fps"] == 81.0
        assert value["full_run_system_fps"] == 81.0
        assert "system_tps" not in value
        assert "full_run_system_tps" not in value

    @pytest.mark.unit
    def test_an_uploaded_succeeded_count_is_replaced(self, tmp_path):
        point = self._video_point(
            tmp_path,
            {
                "qps": 1.0,
                "n_samples_completed": 4,
                "n_samples_failed": 4,
                "n_samples_succeeded": 4,
            },
        )
        value, _ = point.read("result_summary.json")
        assert value["n_samples_succeeded"] == 0

    @pytest.mark.unit
    def test_a_token_model_keeps_a_null_tps_unaliased(self, tmp_path):
        point = self._video_point(tmp_path, {"tps": None, "qps": 1.0}, model="llm")
        value, _ = point.read("result_summary.json")
        assert "system_tps" not in value
        assert "system_fps" not in value

    def _video_run(self, tmp_path, summary):
        root = tmp_path / "acme"
        run_dir = root / "results" / "sys" / "wan-2.2-t2v-a14b" / "r8"
        run_dir.mkdir(parents=True)
        (run_dir / "result_summary.json").write_text(json.dumps(summary))
        (run_dir / "point.yaml").write_text(
            "model_name: wan-2.2-t2v-a14b\nconcurrency: 8\n"
        )
        (run_dir / "system_desc.json").write_text(
            json.dumps({"max_supported_concurrency": 64})
        )
        catalog = Catalog({"wan-2.2-t2v-a14b": {"frames_per_sample": 81}}, {})
        node = SubmissionNode.from_directory(root, ruleset=catalog)
        return node.children[0].children[0].children[0]

    @pytest.mark.unit
    def test_a_video_point_takes_its_throughput_from_system_fps(self, tmp_path):
        node = self._video_run(
            tmp_path,
            {"tps": None, "qps": 1.0, "n_samples_completed": 4, "n_samples_failed": 1},
        )
        assert Point.from_node(node).system_tps == pytest.approx(0.75 * 81)

    @pytest.mark.unit
    def test_a_video_point_without_system_fps_is_refused(self, tmp_path):
        # No sample counts, so no frame rate; an uploaded system_tps is no stand-in.
        node = self._video_run(tmp_path, {"tps": 999.0, "system_tps": 999.0})
        with pytest.raises(ValueError, match="system_fps"):
            Point.from_node(node)

    def _gated_point(self, tmp_path, accuracy):
        root = tmp_path / "acme"
        point_dir = root / "results" / "sys" / "wan-2.2-t2v-a14b" / "r8"
        point_dir.mkdir(parents=True)
        (point_dir / "point.yaml").write_text("model_name: wan-2.2-t2v-a14b\n")
        (point_dir / "accuracy_results.json").write_text(json.dumps(accuracy))
        target = {"vbench_score": {"lower": 70.0, "factor": 0.5}}
        entry = {"accuracy_datasets": {"wan22_vbench": {"targets": target}}}
        catalog = Catalog({"wan-2.2-t2v-a14b": entry}, {})
        node = SubmissionNode.from_directory(root, ruleset=catalog)
        return node.children[0].children[0].children[0]

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "scores, unmet",
        [
            ([{"dataset_name": "wan22_vbench", "score": 35.0}], []),
            ([{"dataset_name": "wan22_vbench", "score": 34.9}], ["wan22_vbench"]),
            ([{"dataset_name": "other", "score": 99.0}], ["wan22_vbench"]),
        ],
    )
    def test_the_catalog_gate_is_judged_per_set(self, tmp_path, scores, unmet):
        point = self._gated_point(tmp_path, {"accuracy_scores": scores})
        value, _ = point.read("accuracy_results.json")
        assert value["gates"]["wan22_vbench"]["gate"] == 35.0
        assert value["gates_unmet"] == unmet
        assert value["gates_checked"] == 1

    @pytest.mark.unit
    def test_an_uploaded_gate_view_is_replaced(self, tmp_path):
        point = self._gated_point(
            tmp_path,
            {
                "accuracy_scores": [{"dataset_name": "wan22_vbench", "score": 1.0}],
                "gates_unmet": [],
            },
        )
        value, _ = point.read("accuracy_results.json")
        assert value["gates_unmet"] == ["wan22_vbench"]

    @pytest.mark.unit
    def test_scores_surface_by_dataset_name(self, tmp_path):
        point = self._point(
            tmp_path,
            {
                "accuracy_results.json": {
                    "accuracy_scores": [
                        {"dataset_name": "wan22_vbench", "score": 1.0},
                        {"dataset_name": "bad", "score": True},
                        {"score": 2.0},
                    ]
                }
            },
        )
        value, _ = point.read("accuracy_results.json")
        assert value["score_by_dataset"] == {"wan22_vbench": 1.0}

    @pytest.mark.unit
    def test_a_set_scored_twice_is_left_out(self, tmp_path):
        point = self._point(
            tmp_path,
            {
                "accuracy_results.json": {
                    "accuracy_scores": [
                        {"dataset_name": "wan22_vbench", "score": 70.0},
                        {"dataset_name": "wan22_vbench", "score": 5.0},
                        {"dataset_name": "other", "score": 1.0},
                    ]
                }
            },
        )
        value, _ = point.read("accuracy_results.json")
        assert value["score_by_dataset"] == {"other": 1.0}

    @pytest.mark.unit
    def test_an_uploaded_score_view_is_never_read(self, tmp_path):
        point = self._point(
            tmp_path,
            {
                "accuracy_results.json": {
                    "accuracy_scores": [{"dataset_name": "wan22_vbench", "score": 5.0}],
                    "score_by_dataset": {"wan22_vbench": 99.0},
                }
            },
        )
        value, _ = point.read("accuracy_results.json")
        assert "score_by_dataset" not in value

    @pytest.mark.unit
    def test_a_document_without_usable_scores_is_left_as_written(self, tmp_path):
        point = self._point(
            tmp_path, {"accuracy_results.json": {"accuracy_scores": "none"}}
        )
        value, _ = point.read("accuracy_results.json")
        assert value == {"accuracy_scores": "none"}

    @pytest.mark.unit
    def test_an_explicit_system_tps_wins_over_the_client_alias(self, tmp_path):
        point = self._point(
            tmp_path, {"result_summary.json": {"system_tps": 1.0, "tps": 999.0}}
        )
        value, _ = point.read("result_summary.json")
        assert value["system_tps"] == 1.0

    @pytest.mark.unit
    def test_a_valid_steady_window_is_the_system_tps(self, tmp_path):
        point = self._point(
            tmp_path,
            {
                "result_summary.json": {
                    "tps": 999.0,
                    "steady_state": {"window": {"sp_lo": 0}, "tps": {"system": 6000.0}},
                }
            },
        )
        value, _ = point.read("result_summary.json")
        assert value["system_tps"] == 6000.0
        assert value["full_run_system_tps"] == 999.0

    @pytest.mark.unit
    def test_a_windowless_steady_state_falls_back_to_the_whole_run(self, tmp_path):
        point = self._point(
            tmp_path,
            {
                "result_summary.json": {
                    "tps": 999.0,
                    "steady_state": {"window": None, "tps": {"system": 6000.0}},
                }
            },
        )
        value, _ = point.read("result_summary.json")
        assert value["system_tps"] == 999.0
        assert value["full_run_system_tps"] == 999.0

    @pytest.mark.unit
    def test_absent_artifact_is_missing(self, tmp_path):
        point = self._point(tmp_path)
        value, error = point.read("result_summary.json")
        assert value is MISSING
        assert error is None

    @pytest.mark.unit
    def test_missing_sentinel_is_the_shared_one_for_presence_ops(self, tmp_path):
        point = self._point(tmp_path)
        value, _ = point.read("result_summary.json")
        assert Absent().holds(Predicate(value, None), None, None) is True
        assert Absent().holds(Predicate(None, None), None, None) is False

    @pytest.mark.unit
    def test_present_null_is_none_not_missing(self, tmp_path):
        point = self._point(tmp_path, {"result_summary.json": None})
        assert point.read("result_summary.json") == (None, None)

    @pytest.mark.unit
    def test_artifacts_are_loaded_once_not_reread(self, tmp_path):
        point = self._point(tmp_path, {"result_summary.json": {"n": 1}})
        (point.dir / "result_summary.json").write_text(json.dumps({"n": 2}))
        assert point.read("result_summary.json") == ({"n": 1}, None)

    @pytest.mark.unit
    def test_duplicate_key_yaml_is_a_parse_error(self, tmp_path):
        point = self._point(tmp_path, {"point.yaml": "a: 1\na: 2\n"})
        value, error = point.read("point.yaml")
        assert value is None
        assert "duplicate key" in error

    @pytest.mark.unit
    def test_yaml_aliases_are_a_parse_error(self, tmp_path):
        point = self._point(tmp_path, {"point.yaml": "base: &b {x: 1}\np: *b\n"})
        value, error = point.read("point.yaml")
        assert value is None
        assert "aliases are not allowed" in error

    @pytest.mark.unit
    def test_yaml_merge_keys_are_a_parse_error(self, tmp_path):
        point = self._point(tmp_path, {"point.yaml": "p: {<<: {x: 1}, y: 2}\n"})
        value, error = point.read("point.yaml")
        assert value is None
        assert "merge keys are not allowed" in error

    @pytest.mark.unit
    def test_unparsable_json_reports_the_error(self, tmp_path):
        point = self._point(tmp_path, {"result_summary.json": "{bad"})
        value, error = point.read("result_summary.json")
        assert value is None
        assert "parse error" in error


class TestReadChannel:
    """`_read_artifact`: bounded, symlink-safe reads. Nothing here aborts the run."""

    def _point_with(self, tmp_path, name: str, data: bytes):
        root = tmp_path / "acme"
        point_dir = root / "results" / "sys" / "model" / "r8"
        point_dir.mkdir(parents=True)
        (point_dir / name).write_bytes(data)
        node = SubmissionNode.from_directory(root)
        return node.children[0].children[0].children[0]

    @pytest.mark.unit
    def test_oversized_artifact_reports_too_large(self, tmp_path, monkeypatch):
        monkeypatch.setattr("inference_endpoint.validation.tree._MAX_ARTIFACT_BYTES", 8)
        point = self._point_with(tmp_path, "result_summary.json", b'{"n": 12}')
        value, error = point.read("result_summary.json")
        assert value is None
        assert "too large" in error

    @pytest.mark.unit
    def test_non_utf8_artifact_reports_a_parse_error(self, tmp_path):
        point = self._point_with(tmp_path, "result_summary.json", b"\xff\xfe{")
        value, error = point.read("result_summary.json")
        assert value is None
        assert "parse error" in error

    @pytest.mark.unit
    def test_a_fifo_artifact_is_refused_not_awaited(self, tmp_path):
        root = tmp_path / "acme"
        point_dir = root / "results" / "sys" / "model" / "r8"
        point_dir.mkdir(parents=True)
        os.mkfifo(point_dir / "point.yaml")
        point = SubmissionNode.from_directory(root).children[0].children[0].children[0]
        value, error = point.read("point.yaml")
        assert value is None
        assert "not a regular file" in error

    @pytest.mark.unit
    def test_a_directory_where_an_artifact_belongs_is_refused(self, tmp_path):
        root = tmp_path / "acme"
        point_dir = root / "results" / "sys" / "model" / "r8"
        (point_dir / "result_summary.json").mkdir(parents=True)
        point = SubmissionNode.from_directory(root).children[0].children[0].children[0]
        value, error = point.read("result_summary.json")
        assert value is None
        assert "not a regular file" in error

    @pytest.mark.unit
    def test_json_duplicate_keys_are_a_parse_error(self, tmp_path):
        point = self._point_with(tmp_path, "result_summary.json", b'{"n": 1, "n": 2}')
        value, error = point.read("result_summary.json")
        assert value is None
        assert "duplicate key" in error

    @pytest.mark.unit
    def test_yaml_duplicate_null_keys_are_a_parse_error(self, tmp_path):
        point = self._point_with(tmp_path, "point.yaml", b"null: 1\nnull: 2\n")
        value, error = point.read("point.yaml")
        assert value is None
        assert "duplicate key" in error

    @pytest.mark.unit
    @pytest.mark.parametrize("payload", [b"", b"   \n", b"just a scalar\n", b"- 1\n"])
    def test_an_unusable_document_is_a_parse_error(self, tmp_path, payload):
        point = self._point_with(tmp_path, "point.yaml", payload)
        value, error = point.read("point.yaml")
        assert value is None
        assert "parse error" in error

    @pytest.mark.unit
    def test_symlinked_artifact_out_of_the_node_is_refused(self, tmp_path):
        outside = tmp_path / "outside.json"
        outside.write_text('{"n": 42}')
        root = tmp_path / "acme"
        point_dir = root / "results" / "sys" / "model" / "r8"
        point_dir.mkdir(parents=True)
        (point_dir / "result_summary.json").symlink_to(outside)
        point = SubmissionNode.from_directory(root).children[0].children[0].children[0]
        value, error = point.read("result_summary.json")
        assert value is None  # never the outside target's content
        assert "parse error" in error

    @pytest.mark.unit
    def test_symlinked_artifact_inside_the_node_is_a_parse_error(self, tmp_path):
        root = tmp_path / "acme"
        point_dir = root / "results" / "sys" / "model" / "r8"
        point_dir.mkdir(parents=True)
        (point_dir / "other.json").write_text('{"n": 1}')
        (point_dir / "result_summary.json").symlink_to(point_dir / "other.json")
        point = SubmissionNode.from_directory(root).children[0].children[0].children[0]
        value, error = point.read("result_summary.json")
        assert value is None
        assert "parse error" in error


class TestScopes:
    """`Node.scopes`: one mapping per node up to the submission. Fact names are keys."""

    def _point(self, tmp_path):
        node, _ = _tree(
            tmp_path,
            {"result_summary.json": {"n": 1}, "point.yaml": "concurrency: 16\n"},
            {"system_power.json": None},
        )
        return node.children[0].children[0].children[0]

    @pytest.mark.unit
    def test_keys_are_fact_names_per_kind(self, tmp_path):
        scopes = self._point(tmp_path).scopes()
        assert set(scopes) == {"submission", "system", "pareto", "point"}
        assert set(scopes["point"]) == {"result_summary", "config"}
        assert set(scopes["system"]) == {"system_power"}

    @pytest.mark.unit
    def test_absent_artifact_is_an_absent_key(self, tmp_path):
        scopes = self._point(tmp_path).scopes()
        assert "system_desc" not in scopes["point"]

    @pytest.mark.unit
    def test_stored_null_is_a_present_key(self, tmp_path):
        scopes = self._point(tmp_path).scopes()
        assert scopes["system"]["system_power"] is None

    @pytest.mark.unit
    def test_corrupt_artifact_reads_as_absent_but_keeps_its_error(self, tmp_path):
        root = tmp_path / "acme"
        point_dir = root / "results" / "sys" / "model" / "r8"
        point_dir.mkdir(parents=True)
        (point_dir / "result_summary.json").write_text("{bad")
        (point_dir / "point.yaml").write_text("concurrency: 8\n")
        point = SubmissionNode.from_directory(root).children[0].children[0].children[0]
        assert "result_summary" not in point.scopes()["point"]
        assert set(point.scopes()["point"]) == {"config"}
        value, error = point.read("result_summary.json")
        assert value is None and "parse error" in error

    @pytest.mark.unit
    def test_root_walks_to_the_submission(self, tmp_path):
        point = self._point(tmp_path)
        assert isinstance(point.root(), SubmissionNode)

    @pytest.mark.unit
    def test_scopes_serve_ref_resolution(self, tmp_path):
        scopes = self._point(tmp_path).scopes()
        assert Ref(("point", "result_summary", "n")).resolve(scopes) == (True, 1)
        assert Ref(("point", "result_summary", "nope")).resolve(scopes) == (
            False,
            None,
        )


class TestRegionsRoot:
    """`Node.scopes` exposes `Regions` under the `regions` root of `Ref`."""

    @pytest.mark.unit
    def test_regions_root_answers_coverage_lookups(self, tmp_path):
        root = tmp_path / "acme"
        (root / "results").mkdir(parents=True)
        scopes = SubmissionNode.from_directory(
            root, regions=Regions.for_bounds(c_min=32, c_max=8192)
        ).scopes()
        assert Ref(("regions", "coverage")).resolve(scopes) == (
            True,
            Interval(1, 9012),
        )
        # The root's `Interval` values are leaves: `Ref` returns one as-is and
        # never walks its attributes (`margin.lo` would be attribute traversal).
        assert Ref(("regions", "by_kind", "margin")).resolve(scopes) == (
            True,
            Interval(8193, 9012),
        )

    @pytest.mark.unit
    def test_without_regions_the_root_stays_away(self, tmp_path):
        root = tmp_path / "acme"
        (root / "results").mkdir(parents=True)
        assert "regions" not in SubmissionNode.from_directory(root).scopes()


class TestLoadBoundaries:
    """The per-tree byte budget and the symlink-root refusal."""

    @pytest.mark.unit
    def test_a_symlinked_upload_root_is_refused_clearly(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        with pytest.raises(ValueError, match="is a symlink"):
            SubmissionNode.from_directory(link)

    @pytest.mark.unit
    def test_the_tree_artifact_budget_refuses_a_bloated_upload(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr("inference_endpoint.validation.tree._MAX_TREE_BYTES", 32)
        root = tmp_path / "acme"
        point = root / "results" / "sys" / "model" / "r16"
        point.mkdir(parents=True)
        _write(point / "point.yaml", {"concurrency": 16})
        _write(point / "system_desc.json", {"max_supported_concurrency": 16})
        with pytest.raises(ValueError, match="tree budget"):
            SubmissionNode.from_directory(root)

    @pytest.mark.unit
    def test_the_tree_budget_accumulates_across_nodes(self, tmp_path, monkeypatch):
        # Individually affordable artifacts in separate run directories whose
        # combined bytes cross the pot: the budget is submission-wide, not
        # per-node (a fresh pot per node is the regression this pins).
        root = tmp_path / "acme"
        for concurrency in (16, 8192):
            point = root / "results" / "sys" / "model" / f"r{concurrency}"
            point.mkdir(parents=True)
            _write(point / "point.yaml", {"concurrency": concurrency})
        total = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        monkeypatch.setattr(
            "inference_endpoint.validation.tree._MAX_TREE_BYTES", total - 1
        )
        with pytest.raises(ValueError, match="tree budget"):
            SubmissionNode.from_directory(root)

    @pytest.mark.unit
    def test_the_tree_budget_accepts_an_exact_spend(self, tmp_path, monkeypatch):
        root = tmp_path / "acme"
        point = root / "results" / "sys" / "model" / "r16"
        point.mkdir(parents=True)
        _write(point / "point.yaml", {"concurrency": 16})
        total = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        monkeypatch.setattr("inference_endpoint.validation.tree._MAX_TREE_BYTES", total)
        SubmissionNode.from_directory(root)

    @pytest.mark.unit
    def test_an_oversized_artifact_still_spends_the_tree_budget(
        self, tmp_path, monkeypatch
    ):
        # The size refusal comes after the charge: two oversized artifacts in
        # one run directory exhaust a small pot even though neither parses.
        monkeypatch.setattr("inference_endpoint.validation.tree._MAX_ARTIFACT_BYTES", 8)
        monkeypatch.setattr("inference_endpoint.validation.tree._MAX_TREE_BYTES", 15)
        root = tmp_path / "acme"
        point = root / "results" / "sys" / "model" / "r16"
        point.mkdir(parents=True)
        (point / "system_desc.json").write_text(
            '{"max_supported_concurrency": 16}', encoding="utf-8"
        )
        (point / "result_summary.json").write_text(
            '{"system_tps": 5400.0}', encoding="utf-8"
        )
        with pytest.raises(ValueError, match="tree budget"):
            SubmissionNode.from_directory(root)


class TestNodeIdentity:
    """Nodes compare by identity; frozen holders stay hashable."""

    @pytest.mark.unit
    def test_nodes_compare_by_identity_and_points_stay_hashable(self, tmp_path):
        node = SubmissionNode(RuleScopes.SUBMISSION, tmp_path)
        twin = SubmissionNode(RuleScopes.SUBMISSION, tmp_path)
        assert node != twin
        point = Point(
            offline=None, concurrency=16, system_tps=1.0, system_desc={}, node=node
        )
        assert isinstance(hash(point), int)
