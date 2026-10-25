# -*- coding: utf-8 -*-
"""
benchmarks/dataset_adapter.py — Public Incident Benchmark Adapter Layer (Phase 6)
Part of AdaRAG-Ops: Research-Grade Operational Incident Diagnosis.

Primary Objective:
Provides a unified, extensible adapter layer to ingest public microservice root-cause
analysis datasets (starting with RCAEval and OpenRCA), mapping external records into
the project's incident schema while preserving:
1. Original record IDs (e.g. RE1_frontend_CPU_1, OpenRCA_telecom_001, CASE_1).
2. Original labels, systems, fault types, and ground-truth root causes.
3. Strict data-leakage prevention via grouped train/val/test splitting.
4. Generalization testing to unseen microservice systems and unseen fault categories.
5. Supplementary retention of existing database, Kubernetes, and TLS cases.
6. Air-gapped / offline operation with configurable paths and zero assumed internet access.
"""

import os
import re
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any, Optional

from agents.state import AgentState


# Default dataset root locations (configurable via environment variables)
DEFAULT_RCAEVAL_DIR = Path(__file__).parent.parent / "datasets" / "rcaeval"
DEFAULT_OPENRCA_DIR = Path(__file__).parent.parent / "datasets" / "openrca"


@dataclass
class BenchmarkIncident:
    """
    Unified operational incident record mapping external benchmark datasets
    into the project's standardized schema.
    """
    # 1. Identity & Origin
    id: str                               # Original case ID (preserved strictly)
    source_dataset: str                   # "RCAEval", "OpenRCA", "Supplementary"
    system_name: str                      # "online-boutique", "sock-shop", "train-ticket", "telecom", etc.
    benchmark_suite: str                  # "RE1", "RE2", "RE3", "enterprise", "supplementary"

    # 2. Original Ground Truth & Fault Labels
    fault_type: str                       # e.g. "CPU", "MEM", "DELAY", "F2", "db_deadlock"
    root_cause_service: str               # Ground truth service, e.g. "frontend", "cartservice"
    root_cause_indicator: str             # Specific metric/log signature, e.g. "container_cpu_usage"
    ground_truth_root_cause: str          # Concise ground truth explanation

    # 3. Project Schema Fields (AgentState & Benchmark compatibility)
    query: str                            # Technical support / incident query
    domain: str                           # "Cloud & Container Ops", "Microservice Application", etc.
    category: str                         # "cpu", "database", "network", "deployment", "application", "general"
    severity: str                         # "L1", "L2", "L3"
    context: str                          # Runbook, SOP, and telemetry context provided to RAG
    reference: str                        # Ground truth reference text for F1/BLEU evaluation
    has_ground_truth: bool = True

    # 4. Telemetry Summaries
    metrics_summary: Optional[str] = None
    logs_summary: Optional[str] = None
    traces_summary: Optional[str] = None
    simulated_l2_dist: float = 0.50

    # 5. Raw Record Preservation (Auditability)
    raw_record: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_benchmark_case(self) -> Dict[str, Any]:
        """Converts into the benchmark case dictionary required by benchmark runners."""
        return {
            "id": self.id,
            "domain": self.domain,
            "query": self.query,
            "context": self.context,
            "reference": self.reference,
            "simulated_l2_dist": self.simulated_l2_dist,
            "ground_truth_service": self.root_cause_service,
            "fault_type": self.fault_type,
            "system_name": self.system_name,
            "source_dataset": self.source_dataset
        }

    def to_agent_state(self) -> AgentState:
        """Converts into initial AgentState for LangGraph pipeline execution."""
        return {
            "user_query": self.query,
            "category": self.category,
            "severity": self.severity,
            "summary": self.query[:80],
            "kb_results": [
                {"source": f"{self.system_name}_runbook.md", "text": self.context, "score": self.simulated_l2_dist}
            ],
            "kb_context": self.context,
            "root_cause": "",
            "confidence": "",
            "resolution_steps": "",
            "resolution_status": "",
            "verified": False,
            "verification_notes": "",
            "escalated": False,
            "escalation_reason": "",
            "final_answer": "",
            "selected_model": None,
            "selected_provider": None,
            "routing_metadata": None,
            "needs_model_escalation": False,
            "escalation_level": 0
        }


class RCAEvalAdapter:
    """
    Adapter for the RCAEval microservice root cause analysis benchmark.
    Spans Online Boutique, Sock Shop, and Train Ticket across RE1, RE2, and RE3 suites.
    """

    def __init__(self, dataset_dir: Optional[Path] = None):
        env_dir = os.getenv("RCAEVAL_DATASET_DIR")
        self.dataset_dir = Path(env_dir) if env_dir else (dataset_dir or DEFAULT_RCAEVAL_DIR)

    @staticmethod
    def map_fault_category(fault_type: str) -> str:
        """Maps RCAEval fault types to project category schema."""
        ft = fault_type.upper()
        if "CPU" in ft:
            return "cpu"
        elif "MEM" in ft or "OOM" in ft:
            return "application"
        elif any(k in ft for k in ["DELAY", "LOSS", "SOCKET", "NET"]):
            return "network"
        elif "DISK" in ft:
            return "database"
        elif ft.startswith("F"):  # F1-F5 code-level bugs in RE3
            return "application"
        return "general"

    @staticmethod
    def map_severity(fault_type: str, system: str) -> str:
        """Grounds severity based on microservice failure criticality."""
        ft = fault_type.upper()
        if ft in ("F4", "F5") or ("auth" in system or "payment" in system):
            return "L3"
        elif ft in ("CPU", "MEM", "DELAY", "F1", "F2", "SOCKET"):
            return "L2"
        return "L1"

    def load(self) -> List[BenchmarkIncident]:
        """Loads and adapts RCAEval benchmark records."""
        cases_file = self.dataset_dir / "cases.json"
        if not cases_file.exists():
            raise FileNotFoundError(
                f"RCAEval dataset index not found at '{cases_file}'. "
                f"Configure RCAEVAL_DATASET_DIR or see PUBLIC_BENCHMARKS.md for setup instructions."
            )

        with open(cases_file, "r", encoding="utf-8") as f:
            raw_cases = json.load(f)

        incidents: List[BenchmarkIncident] = []
        for r in raw_cases:
            # Validate required ground truth (do not fabricate missing ground truth)
            rc_service = r.get("root_cause_service")
            f_type = r.get("fault_type")
            if not rc_service or not f_type:
                # Log filtered item without ground truth
                continue

            category = self.map_fault_category(f_type)
            severity = r.get("severity") or self.map_severity(f_type, r.get("system", ""))
            domain_label = f"Microservice: {r.get('system', 'System').replace('-', ' ').title()}"

            # Format context with runbook text and telemetry
            ctx_parts = [r.get("context", "")]
            telem = r.get("telemetry", {})
            if telem:
                telem_lines = []
                if telem.get("metrics_summary"):
                    telem_lines.append(f"Telemetry Metrics: {telem['metrics_summary']}")
                if telem.get("logs_summary"):
                    telem_lines.append(f"Telemetry Logs: {telem['logs_summary']}")
                if telem.get("traces_summary"):
                    telem_lines.append(f"Telemetry Traces: {telem['traces_summary']}")
                if telem_lines:
                    ctx_parts.append("\n".join(telem_lines))
            full_context = "\n\n".join(p for p in ctx_parts if p.strip())

            inc = BenchmarkIncident(
                id=r.get("id", f"RCAEval_{len(incidents)+1}"),
                source_dataset="RCAEval",
                system_name=r.get("system", "microservice"),
                benchmark_suite=r.get("benchmark", "RE1"),
                fault_type=f_type,
                root_cause_service=rc_service,
                root_cause_indicator=r.get("root_cause_indicator", "telemetry_anomaly"),
                ground_truth_root_cause=r.get("reference", f"Fault on {rc_service}"),
                has_ground_truth=True,
                query=r.get("query", r.get("symptom", "Microservice performance degradation")),
                domain=domain_label,
                category=category,
                severity=severity,
                context=full_context,
                reference=r.get("reference", ""),
                metrics_summary=telem.get("metrics_summary"),
                logs_summary=telem.get("logs_summary"),
                traces_summary=telem.get("traces_summary"),
                simulated_l2_dist=0.35 if r.get("benchmark") == "RE1" else 0.65,
                raw_record=r
            )
            incidents.append(inc)

        return incidents


class OpenRCAAdapter:
    """
    Adapter for the Microsoft OpenRCA enterprise benchmark dataset.
    Covers Telecom, Banking, and Market enterprise systems.
    """

    def __init__(self, dataset_dir: Optional[Path] = None):
        env_dir = os.getenv("OPENRCA_DATASET_DIR")
        self.dataset_dir = Path(env_dir) if env_dir else (dataset_dir or DEFAULT_OPENRCA_DIR)

    @staticmethod
    def map_fault_category(fault_type: str) -> str:
        ft = fault_type.lower()
        if "cpu" in ft:
            return "cpu"
        elif "deadlock" in ft or "sql" in ft or "db" in ft:
            return "database"
        elif "pool" in ft or "socket" in ft or "network" in ft or "delay" in ft:
            return "network"
        elif "memory" in ft or "leak" in ft:
            return "application"
        return "general"

    def load(self) -> List[BenchmarkIncident]:
        cases_file = self.dataset_dir / "cases.json"
        if not cases_file.exists():
            raise FileNotFoundError(
                f"OpenRCA dataset index not found at '{cases_file}'. "
                f"Configure OPENRCA_DATASET_DIR or see PUBLIC_BENCHMARKS.md for setup instructions."
            )

        with open(cases_file, "r", encoding="utf-8") as f:
            raw_cases = json.load(f)

        incidents: List[BenchmarkIncident] = []
        for r in raw_cases:
            rc_service = r.get("root_cause_service")
            f_type = r.get("fault_type")
            if not rc_service or not f_type:
                continue

            category = self.map_fault_category(f_type)
            severity = r.get("severity", "L2")
            domain_label = f"Enterprise System: {r.get('system', 'System').title()}"

            telem = r.get("telemetry", {})
            ctx_parts = [r.get("context", "")]
            if telem:
                telem_lines = []
                if telem.get("metrics_summary"):
                    telem_lines.append(f"Telemetry Metrics: {telem['metrics_summary']}")
                if telem.get("logs_summary"):
                    telem_lines.append(f"Telemetry Logs: {telem['logs_summary']}")
                if telem.get("traces_summary"):
                    telem_lines.append(f"Telemetry Traces: {telem['traces_summary']}")
                if telem_lines:
                    ctx_parts.append("\n".join(telem_lines))
            full_context = "\n\n".join(p for p in ctx_parts if p.strip())

            inc = BenchmarkIncident(
                id=r.get("id", f"OpenRCA_{len(incidents)+1}"),
                source_dataset="OpenRCA",
                system_name=r.get("system", "enterprise"),
                benchmark_suite=r.get("benchmark", "enterprise"),
                fault_type=f_type,
                root_cause_service=rc_service,
                root_cause_indicator=r.get("root_cause_indicator", "kpi_anomaly"),
                ground_truth_root_cause=r.get("reference", f"Fault on {rc_service}"),
                has_ground_truth=True,
                query=r.get("query", r.get("symptom", "Enterprise service anomaly")),
                domain=domain_label,
                category=category,
                severity=severity,
                context=full_context,
                reference=r.get("reference", ""),
                metrics_summary=telem.get("metrics_summary"),
                logs_summary=telem.get("logs_summary"),
                traces_summary=telem.get("traces_summary"),
                simulated_l2_dist=0.45,
                raw_record=r
            )
            incidents.append(inc)

        return incidents


class SupplementaryAdapter:
    """
    Retains the existing database, Kubernetes, and TLS scenarios from
    benchmark_adaptive_rag.py as a permanent supplementary scenario suite.
    """

    @staticmethod
    def load() -> List[BenchmarkIncident]:
        from benchmark_adaptive_rag import BENCHMARK_CASES

        incidents: List[BenchmarkIncident] = []
        for c in BENCHMARK_CASES:
            cid = c.get("id", "")
            if cid == "CASE_1":
                rc_service = "postgresql-database"
                f_type = "connection_pooling_exhaustion"
                cat = "database"
                sev = "L2"
            elif cid == "CASE_2":
                rc_service = "kubernetes-api-gateway"
                f_type = "OOMKilled_exit_137"
                cat = "application"
                sev = "L2"
            else:  # CASE_3
                rc_service = "payment-gateway-ingress"
                f_type = "tls_cert_expired"
                cat = "network"
                sev = "L2"

            inc = BenchmarkIncident(
                id=cid,
                source_dataset="Supplementary",
                system_name="core-infrastructure",
                benchmark_suite="supplementary",
                fault_type=f_type,
                root_cause_service=rc_service,
                root_cause_indicator="runbook_diagnostic_claim",
                ground_truth_root_cause=c.get("reference", ""),
                has_ground_truth=True,
                query=c.get("query", ""),
                domain=c.get("domain", "Infrastructure"),
                category=cat,
                severity=sev,
                context=c.get("context", ""),
                reference=c.get("reference", ""),
                simulated_l2_dist=c.get("simulated_l2_dist", 0.50),
                raw_record=c
            )
            incidents.append(inc)

        return incidents


class IncidentDatasetSplitter:
    """
    Rigorous splitting methods ensuring:
    1. Zero leakage: Splitting by incident or fault instance (system, service, fault).
    2. Out-of-Distribution (OOD) System Generalization: Unseen microservice system.
    3. Out-of-Distribution (OOD) Fault Generalization: Unseen fault category.
    """

    @staticmethod
    def split_grouped(
        incidents: List[BenchmarkIncident],
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        test_ratio: float = 0.20,
        seed: int = 42
    ) -> Dict[str, List[BenchmarkIncident]]:
        """
        Grouped split partitioned by (system_name, root_cause_service, fault_type).
        Guarantees that all related instances of a fault on a service remain
        strictly inside the same split to prevent data leakage.
        """
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

        # Form unique group clusters
        groups: Dict[str, List[BenchmarkIncident]] = {}
        for inc in incidents:
            group_key = f"{inc.system_name}::{inc.root_cause_service}::{inc.fault_type}"
            groups.setdefault(group_key, []).append(inc)

        # Deterministic hashing for grouping
        train_list, val_list, test_list = [], [], []
        sorted_keys = sorted(groups.keys())

        for idx, key in enumerate(sorted_keys):
            # Stable hash bucket
            h = int(hashlib.md5(f"{key}_{seed}".encode("utf-8")).hexdigest(), 16) % 1000
            score = h / 1000.0

            if score < train_ratio:
                train_list.extend(groups[key])
            elif score < (train_ratio + val_ratio):
                val_list.extend(groups[key])
            else:
                test_list.extend(groups[key])

        # Safety fallback if any split ended up empty due to small total count
        if not test_list and sorted_keys:
            last_key = sorted_keys[-1]
            if groups[last_key] in train_list:
                for item in groups[last_key]:
                    train_list.remove(item)
                test_list.extend(groups[last_key])

        return {
            "train": train_list,
            "val": val_list,
            "test": test_list
        }

    @staticmethod
    def split_cross_system(
        incidents: List[BenchmarkIncident],
        unseen_system: str = "train-ticket"
    ) -> Dict[str, List[BenchmarkIncident]]:
        """
        Cross-System Generalization Split:
        Evaluates zero-shot transfer to an entirely unseen microservice system topology.
        Train: Known systems (e.g. online-boutique, sock-shop)
        Test: Unseen system (e.g. train-ticket with 40+ microservices)
        """
        train_list = [inc for inc in incidents if inc.system_name.lower() != unseen_system.lower()]
        test_list = [inc for inc in incidents if inc.system_name.lower() == unseen_system.lower()]

        # Split known systems into 80% train / 20% val
        val_count = max(1, int(len(train_list) * 0.20))
        val_list = train_list[-val_count:]
        train_list = train_list[:-val_count]

        return {
            "train": train_list,
            "val": val_list,
            "test": test_list,
            "unseen_target": unseen_system
        }

    @staticmethod
    def split_cross_fault(
        incidents: List[BenchmarkIncident],
        unseen_fault_types: Optional[List[str]] = None
    ) -> Dict[str, List[BenchmarkIncident]]:
        """
        Cross-Fault Generalization Split:
        Evaluates zero-shot transfer to unseen failure modes.
        Train: Resource faults (CPU, MEM, DISK)
        Test: Network and code-level faults (DELAY, LOSS, SOCKET, F1-F5, db_deadlock)
        """
        if unseen_fault_types is None:
            unseen_fault_types = ["DELAY", "LOSS", "SOCKET", "F1", "F2", "F3", "F4", "F5", "network_delay", "db_deadlock"]

        unseen_set = {f.upper() for f in unseen_fault_types}
        train_list = [inc for inc in incidents if inc.fault_type.upper() not in unseen_set]
        test_list = [inc for inc in incidents if inc.fault_type.upper() in unseen_set]

        val_count = max(1, int(len(train_list) * 0.20))
        val_list = train_list[-val_count:]
        train_list = train_list[:-val_count]

        return {
            "train": train_list,
            "val": val_list,
            "test": test_list,
            "unseen_faults": list(unseen_set)
        }


class IncidentDatasetRegistry:
    """
    Unified loader for public and supplementary benchmark suites.
    Supports RCAEval, OpenRCA, and Supplementary scenarios.
    """

    @classmethod
    def load_dataset(
        cls,
        dataset_name: str = "all",
        include_supplementary: bool = True
    ) -> List[BenchmarkIncident]:
        """
        Loads incidents by dataset name: 'rcaeval', 'openrca', 'supplementary', or 'all'.
        """
        name = dataset_name.lower().strip()
        all_incidents: List[BenchmarkIncident] = []

        if name in ("rcaeval", "all"):
            try:
                rca_adapter = RCAEvalAdapter()
                all_incidents.extend(rca_adapter.load())
            except Exception as e:
                print(f"  [!] Note: RCAEval loading notice: {e}")

        if name in ("openrca", "all"):
            try:
                open_adapter = OpenRCAAdapter()
                all_incidents.extend(open_adapter.load())
            except Exception as e:
                print(f"  [!] Note: OpenRCA loading notice: {e}")

        if name in ("supplementary", "all") or (include_supplementary and name not in ("rcaeval", "openrca")):
            all_incidents.extend(SupplementaryAdapter.load())

        return all_incidents
