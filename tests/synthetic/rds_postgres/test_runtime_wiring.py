from __future__ import annotations

from typing import Any

from app.agent.investigation import (
    _RDS_CLOUDWATCH_SEED_TOOL_NAMES,
    _RDS_EC2_SEED_TOOL_NAMES,
    _build_seed_calls,
    _merge_tool_evidence,
)
from app.tools.registered_tool import RegisteredTool
from tests.synthetic.rds_postgres.run_suite import _build_resolved_integrations
from tests.synthetic.rds_postgres.scenario_loader import SUITE_DIR, load_scenario


class _FakeLLM:
    pass


def _tool(name: str, source: str) -> RegisteredTool:
    return RegisteredTool(
        name=name,
        description=f"{name} test tool",
        input_schema={"type": "object", "properties": {}, "required": []},
        source=source,
        run=lambda: {},
    )


def test_seed_calls_exclude_injected_and_null_params() -> None:
    tool = RegisteredTool(
        name="query_grafana_logs",
        description="test logs",
        input_schema={
            "type": "object",
            "properties": {
                "service_name": {"type": "string"},
                "execution_run_id": {"type": "string"},
            },
            "required": ["service_name"],
        },
        source="grafana",
        run=lambda **_kwargs: {},
        injected_params=("grafana_backend", "grafana_endpoint"),
        extract_params=lambda _sources: {
            "service_name": "payments-prod",
            "execution_run_id": None,
            "grafana_backend": object(),
            "grafana_endpoint": "",
        },
    )
    state: dict[str, Any] = {
        "alert_source": "cloudwatch",
        "raw_alert": {
            "commonLabels": {"service": "rds"},
            "commonAnnotations": {"db_instance_identifier": "payments-prod"},
        },
        "resolved_integrations": {},
    }

    calls = _build_seed_calls(state, [tool], _FakeLLM())

    assert calls[0].input == {"service_name": "payments-prod"}


def test_mock_runner_exposes_rds_backend_and_identity() -> None:
    fixture = load_scenario(SUITE_DIR / "001-replication-lag")

    resolved = _build_resolved_integrations(fixture, use_mock_grafana=True)

    assert resolved is not None
    assert resolved["rds"]["db_instance_identifier"] == "payments-prod"
    assert resolved["rds"]["region"] == "us-east-1"
    assert resolved["rds"]["_backend"] is resolved["aws"]["ec2_backend"]


def test_mock_runner_exposes_ec2_only_when_fixture_has_topology() -> None:
    rds_fixture = load_scenario(SUITE_DIR / "001-replication-lag")
    ec2_fixture = load_scenario(SUITE_DIR / "015-mysql-ec2-load-attribution")

    rds_resolved = _build_resolved_integrations(rds_fixture, use_mock_grafana=True)
    ec2_resolved = _build_resolved_integrations(ec2_fixture, use_mock_grafana=True)

    assert rds_resolved is not None
    assert "ec2" not in rds_resolved
    assert ec2_resolved is not None
    assert ec2_resolved["ec2"]["vpc_id"] == "vpc-0a1b2c3d"
    assert ec2_resolved["ec2"]["_backend"] is ec2_resolved["aws"]["ec2_backend"]


def test_rds_cloudwatch_seed_uses_expected_tool_trajectory() -> None:
    tools = [
        *[_tool(name, "grafana" if "grafana" in name else "rds") for name in _RDS_CLOUDWATCH_SEED_TOOL_NAMES],
        _tool("get_hermes_logs", "hermes"),
        _tool("query_grafana_traces", "grafana"),
        _tool("get_sre_guidance", "knowledge"),
    ]
    state: dict[str, Any] = {
        "alert_source": "cloudwatch",
        "raw_alert": {
            "commonLabels": {"service": "rds", "engine": "postgres"},
            "commonAnnotations": {"db_instance_identifier": "payments-prod"},
        },
        "resolved_integrations": {},
    }

    calls = _build_seed_calls(state, tools, _FakeLLM())

    assert [call.name for call in calls] == list(_RDS_CLOUDWATCH_SEED_TOOL_NAMES)


def test_rds_cloudwatch_seed_includes_connected_ec2_topology() -> None:
    source_by_name = {
        "describe_rds_instance": "rds",
        "describe_rds_events": "rds",
        "ec2_instances_by_tag": "ec2",
        "get_elb_target_health": "ec2",
        "query_grafana_metrics": "grafana",
        "query_grafana_logs": "grafana",
    }
    tools = [_tool(name, source_by_name[name]) for name in _RDS_EC2_SEED_TOOL_NAMES]
    state: dict[str, Any] = {
        "alert_source": "cloudwatch",
        "raw_alert": {
            "commonLabels": {"service": "rds", "engine": "mysql"},
            "commonAnnotations": {"db_instance_identifier": "orders-prod"},
        },
        "resolved_integrations": {"ec2": {"vpc_id": "vpc-0a1b2c3d"}},
    }

    calls = _build_seed_calls(state, tools, _FakeLLM())

    assert [call.name for call in calls] == list(_RDS_EC2_SEED_TOOL_NAMES)


def test_grafana_metrics_split_cloudwatch_and_k8s_sources() -> None:
    evidence: dict[str, Any] = {}
    metrics = [
        {
            "metric": {
                "__name__": "aws_rds_cpu_utilization_average",
                "dbinstanceidentifier": "payments-prod",
            },
            "values": [[1, "91"]],
        },
        {
            "metric": {
                "__name__": "k8s_pod_metrics_error_rate_pct",
                "source_type": "k8s_pod_metrics",
            },
            "values": [[1, "12"]],
        },
    ]

    _merge_tool_evidence(
        evidence,
        "query_grafana_metrics",
        {"metric_name": "all", "metrics": metrics},
        {"metric_name": "all"},
    )

    assert evidence["aws_cloudwatch_metrics"]["metrics"] == [metrics[0]]
    assert evidence["k8s_pod_metrics"]["metrics"] == [metrics[1]]


def test_grafana_logs_split_rds_pi_and_k8s_sources() -> None:
    evidence: dict[str, Any] = {}
    logs = [
        {"source_type": "db-instance", "message": "storage autoscaling started"},
        {
            "source_type": "aws_performance_insights",
            "message": "Top SQL Activity: SELECT 1 | Avg Load: 4.2 AAS | Waits: CPU",
        },
        {"source_type": "k8s_events", "message": "ScalingReplicaSet"},
        {"source_type": "k8s_rollout", "message": "rollout completed"},
    ]

    _merge_tool_evidence(
        evidence,
        "query_grafana_logs",
        {"logs": logs, "error_logs": [], "query": "", "service_name": "payments-prod"},
        {},
    )

    assert evidence["aws_rds_events"] == [logs[0]]
    assert "Top SQL Activity" in evidence["aws_performance_insights"]["observations"][0]
    assert evidence["k8s_events"]["logs"] == [logs[2]]
    assert evidence["k8s_rollout"]["logs"] == [logs[3]]
