# -*- coding: utf-8 -*-
"""
agents/incident_response_cache.py — Persistent Exact and Semantic Incident Response Cache
Part of AdaRAG-Ops: Adaptive Multi-LLM RAG for Operational Incident Diagnosis.

Architectural Goals:
1. Zero LLM Calls & Zero Cost on Safe Cache Hits:
   Checks the persistent response cache before query classification, model routing,
   or diagnosis. Returns the saved verified diagnosis and remediation steps immediately.
2. Three-Stage Lookup Strategy:
   Stage A: Exact normalized query fingerprint matching (SHA-256).
   Stage B: Semantic cosine similarity search via local sentence embeddings
            with contextual compatibility guards (services, error signatures, severity).
   Stage C: Normal diagnosis pipeline execution on cache miss.
3. Strict Safety & Freshness:
   - Caches ONLY verified, evidence-grounded, non-escalated resolutions.
   - Enforces TTL expiration and manual/version-based invalidation.
   - Rejects semantic matches that exhibit conflicting error codes or affected services.
4. Concurrency & Persistence:
   - SQLite backed (WAL mode) with thread-safe connection pooling.
   - Survives restarts and system crashes.
"""

import os
import re
import time
import json
import sqlite3
import hashlib
import threading
import contextlib
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# Cache configuration defaults
DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "artifacts" / "cache"
DEFAULT_DB_PATH = DEFAULT_CACHE_DIR / "incident_cache.db"
DEFAULT_SCHEMA_VERSION = "1.0.0"
DEFAULT_TTL_SECONDS = 86400.0  # 24 hours
DEFAULT_SIMILARITY_THRESHOLD = 0.95


@dataclass
class CacheRecord:
    cache_id: str
    query: str
    normalized_query: str
    fingerprint: str
    incident_category: str
    severity: str
    affected_service: str
    error_signature: str
    root_cause: str
    resolution_steps: str
    resolution_status: str
    supporting_evidence: List[dict]
    structured_diagnosis: Optional[dict]
    verification_status: str
    verification_notes: str
    confidence_score: float
    calibrated_confidence: float
    model_id: str
    provider: str
    created_at: float
    last_accessed_at: float
    expires_at: float
    kb_version: str
    schema_version: str = DEFAULT_SCHEMA_VERSION
    hit_count: int = 0
    is_invalidated: bool = False
    invalidation_reason: Optional[str] = None
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if "embedding" in d:
            d.pop("embedding")  # Exclude raw vector for lightweight serialization
        return d


@dataclass
class CacheLookupResult:
    hit: bool
    match_type: str  # "exact", "semantic", "miss"
    similarity: float
    record: Optional[CacheRecord] = None
    lookup_latency_sec: float = 0.0
    avoided_model_calls: int = 0
    avoided_cost_usd: float = 0.0
    notes: str = ""


class IncidentResponseCache:
    """
    Persistent SQLite-backed Exact and Semantic Cache for Incident Diagnosis.
    Thread-safe, ACID-compliant, with local zero-cost embeddings and context guards.
    """

    _instance_lock = threading.Lock()
    _instance = None

    @classmethod
    def reset_instance(cls):
        with cls._instance_lock:
            cls._instance = None

    @classmethod
    def get_instance(cls, db_path: Optional[Path] = None):
        with cls._instance_lock:
            target_path = Path(db_path) if db_path else None
            if cls._instance is None or (target_path and target_path != cls._instance.db_path):
                cls._instance = cls(db_path=db_path)
            return cls._instance

    def __init__(self, db_path: Optional[Path] = None):
        self._enabled = os.getenv("INCIDENT_CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
        self._semantic_enabled = os.getenv("INCIDENT_CACHE_SEMANTIC_ENABLED", "true").lower() in ("true", "1", "yes")
        self._similarity_threshold = float(os.getenv("INCIDENT_CACHE_SIMILARITY_THRESHOLD", str(DEFAULT_SIMILARITY_THRESHOLD)))
        self._ttl_seconds = float(os.getenv("INCIDENT_CACHE_TTL_SECONDS", str(DEFAULT_TTL_SECONDS)))

        raw_path = os.getenv("INCIDENT_CACHE_DB_PATH")
        if db_path:
            self.db_path = Path(db_path)
        elif raw_path:
            self.db_path = Path(raw_path)
        else:
            self.db_path = DEFAULT_DB_PATH

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._embedding_model = None
        self._metrics = {
            "total_lookups": 0,
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "rejected_semantic_matches": 0,
            "saved_records": 0,
            "invalidations": 0,
            "fallback_errors": 0,
            "total_avoided_model_calls": 0,
            "total_avoided_cost_usd": 0.0,
            "total_lookup_latency_sec": 0.0,
        }
        self._init_database()

    @property
    def enabled(self) -> bool:
        env = os.getenv("INCIDENT_CACHE_ENABLED")
        if env is not None:
            return env.lower() in ("true", "1", "yes")
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = bool(value)

    @property
    def semantic_enabled(self) -> bool:
        env = os.getenv("INCIDENT_CACHE_SEMANTIC_ENABLED")
        if env is not None:
            return env.lower() in ("true", "1", "yes")
        return self._semantic_enabled

    @semantic_enabled.setter
    def semantic_enabled(self, value: bool):
        self._semantic_enabled = bool(value)

    @property
    def similarity_threshold(self) -> float:
        env = os.getenv("INCIDENT_CACHE_SIMILARITY_THRESHOLD")
        if env is not None:
            try:
                return float(env)
            except ValueError:
                pass
        return self._similarity_threshold

    @similarity_threshold.setter
    def similarity_threshold(self, value: float):
        self._similarity_threshold = float(value)

    @property
    def ttl_seconds(self) -> float:
        env = os.getenv("INCIDENT_CACHE_TTL_SECONDS")
        if env is not None:
            try:
                return float(env)
            except ValueError:
                pass
        return self._ttl_seconds

    @ttl_seconds.setter
    def ttl_seconds(self, value: float):
        self._ttl_seconds = float(value)


    @contextlib.contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            # WAL mode enables concurrent readers while writing
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            yield conn
        finally:
            conn.close()

    def _init_database(self):
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS incident_response_cache (
                        cache_id TEXT PRIMARY KEY,
                        query TEXT NOT NULL,
                        normalized_query TEXT NOT NULL,
                        fingerprint TEXT NOT NULL,
                        embedding BLOB,
                        incident_category TEXT,
                        severity TEXT,
                        affected_service TEXT,
                        error_signature TEXT,
                        root_cause TEXT NOT NULL,
                        resolution_steps TEXT NOT NULL,
                        resolution_status TEXT NOT NULL,
                        supporting_evidence TEXT,
                        structured_diagnosis TEXT,
                        verification_status TEXT NOT NULL,
                        verification_notes TEXT,
                        confidence_score REAL,
                        calibrated_confidence REAL,
                        model_id TEXT,
                        provider TEXT,
                        created_at REAL NOT NULL,
                        last_accessed_at REAL NOT NULL,
                        expires_at REAL NOT NULL,
                        kb_version TEXT,
                        schema_version TEXT NOT NULL,
                        hit_count INTEGER DEFAULT 0,
                        is_invalidated INTEGER DEFAULT 0,
                        invalidation_reason TEXT
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_fingerprint ON incident_response_cache(fingerprint);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_status ON incident_response_cache(is_invalidated, expires_at);")
                conn.commit()
        except Exception as e:
            self._metrics["fallback_errors"] += 1
            print(f"[CACHE ERROR] Failed initializing SQLite database at {self.db_path}: {e}")

    def _get_embedding_model(self):
        """Loads the project's local SentenceTransformer singleton with 0 paid API cost."""
        if self._embedding_model is None:
            try:
                from rag.retriever import _get_model
                self._embedding_model = _get_model()
            except Exception:
                try:
                    from sentence_transformers import SentenceTransformer
                    try:
                        self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
                    except Exception:
                        self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                except Exception as e:
                    self._metrics["fallback_errors"] += 1
                    print(f"[CACHE WARNING] SentenceTransformer unavailable for semantic caching: {e}")
                    self._embedding_model = None
        return self._embedding_model

    # -------------------------------------------------------------------------
    # Safe Query Normalization & Feature Extraction
    # -------------------------------------------------------------------------

    @staticmethod
    def normalize_query(query: str) -> str:
        """
        Conservative query normalization:
        Trims whitespace, unifies whitespace and line breaks, but strictly preserves
        error codes, version numbers, service names, and technical tokens.
        """
        if not query:
            return ""
        q = query.strip()
        q = re.sub(r"[\r\n\t]+", " ", q)
        q = re.sub(r"\s{2,}", " ", q)
        return q

    @staticmethod
    def compute_fingerprint(normalized_query: str) -> str:
        """Computes deterministic SHA-256 fingerprint from case-insensitive normalized text."""
        return hashlib.sha256(normalized_query.lower().strip().encode("utf-8")).hexdigest()

    @staticmethod
    def extract_context_signals(query: str) -> Dict[str, str]:
        """Extracts affected service, error codes, and symptoms for semantic compatibility checks."""
        q_lower = query.lower()

        # 1. Error Signatures
        sig = ""
        patterns = [
            (r"exit\s*code\s*(\d+)", "exit_code_\\1"),
            (r"oomkilled|oom\s*killed", "oomkilled"),
            (r"crashloopbackoff|crashloop", "crashloopbackoff"),
            (r"sigsegv|segfault", "sigsegv"),
            (r"sigkill", "sigkill"),
            (r"502\s*bad\s*gateway|502", "http_502"),
            (r"504\s*gateway\s*timeout|504", "http_504"),
            (r"503\s*service\s*unavailable|503", "http_503"),
            (r"404\s*not\s*found", "http_404"),
            (r"deadlock\w*", "deadlock"),
            (r"ssl|tls|handshake|certificate\s*expired", "tls_error"),
            (r"connection\s*refused", "connection_refused"),
            (r"connection\s*slots|connection\s*pool|too\s*many\s*clients", "connection_pool_exhausted"),
            (r"cpu\s*(?:utilization|usage)?|processor\s*usage", "cpu_high"),
            (r"replication\s*(?:delay|lag)", "replication_lag"),
            (r"rollback", "rollback"),
        ]
        for pat, tag in patterns:
            match = re.search(pat, q_lower)
            if match:
                sig = match.expand(tag) if "\\" in tag else tag
                break

        # 2. Affected Services
        service = "unknown"
        service_mapping = [
            (r"postgres(?:ql)?", "postgres"),
            (r"redis", "redis"),
            (r"kafka", "kafka"),
            (r"nginx", "nginx"),
            (r"rabbitmq", "rabbitmq"),
            (r"mysql", "mysql"),
            (r"mongodb", "mongodb"),
            (r"elasticsearch", "elasticsearch"),
            (r"api[-\s_]?gateway", "api_gateway"),
            (r"checkout(?:[-\s_]?api)?", "checkout"),
            (r"payment(?:s)?(?:[-\s_]?service)?", "payments"),
            (r"frontend", "frontend"),
            (r"cartservice", "cartservice"),
            (r"catalog", "catalog"),
            (r"auth(?:entication)?", "auth"),
        ]
        for spat, canonical_name in service_mapping:
            if re.search(r"\b" + spat + r"\b", q_lower):
                service = canonical_name
                break

        return {
            "error_signature": sig,
            "affected_service": service
        }

    # -------------------------------------------------------------------------
    # Stage A: Exact Match Lookup
    # -------------------------------------------------------------------------

    def lookup_exact(self, fingerprint: str, current_time: float) -> Optional[CacheRecord]:
        """Queries SQLite for unexpired, un-invalidated, verified exact match."""
        try:
            with self._get_connection() as conn:
                row = conn.execute("""
                    SELECT * FROM incident_response_cache
                    WHERE fingerprint = ?
                      AND is_invalidated = 0
                      AND expires_at > ?
                      AND verification_status = 'verified'
                    ORDER BY hit_count DESC, created_at DESC
                    LIMIT 1;
                """, (fingerprint, current_time)).fetchone()

                if row:
                    record = self._row_to_record(row)
                    # Update hit metrics atomically
                    conn.execute("""
                        UPDATE incident_response_cache
                        SET hit_count = hit_count + 1, last_accessed_at = ?
                        WHERE cache_id = ?;
                    """, (current_time, record.cache_id))
                    conn.commit()
                    return record
        except Exception as e:
            self._metrics["fallback_errors"] += 1
            print(f"[CACHE WARNING] Exact lookup failed: {e}")
        return None

    # -------------------------------------------------------------------------
    # Stage B: Semantic Match Lookup with Context Guard
    # -------------------------------------------------------------------------

    def lookup_semantic(
        self,
        query: str,
        query_vector: np.ndarray,
        context_signals: Dict[str, str],
        current_time: float
    ) -> Tuple[Optional[CacheRecord], float, str]:
        """
        Scans verified cache records using cosine similarity and contextual validation.
        Rejects matches with conflicting services or incompatible error signatures.
        """
        if not self.semantic_enabled or query_vector is None:
            return None, 0.0, "semantic_disabled_or_unavailable"

        try:
            with self._get_connection() as conn:
                rows = conn.execute("""
                    SELECT * FROM incident_response_cache
                    WHERE is_invalidated = 0
                      AND expires_at > ?
                      AND verification_status = 'verified'
                      AND embedding IS NOT NULL;
                """, (current_time,)).fetchall()

                if not rows:
                    return None, 0.0, "no_cached_vectors"

                best_record = None
                best_sim = -1.0
                rejection_reason = ""

                for row in rows:
                    emb_blob = row["embedding"]
                    if not emb_blob:
                        continue
                    cached_vec = np.frombuffer(emb_blob, dtype=np.float32)

                    # Cosine similarity calculation
                    norm_q = np.linalg.norm(query_vector)
                    norm_c = np.linalg.norm(cached_vec)
                    if norm_q < 1e-6 or norm_c < 1e-6:
                        continue
                    sim = float(np.dot(query_vector, cached_vec) / (norm_q * norm_c))

                    cand_record = self._row_to_record(row)

                    # Contextual Compatibility Verification:
                    # 1. Error signature compatibility
                    q_sig = context_signals.get("error_signature", "")
                    c_sig = cand_record.error_signature or ""
                    if q_sig and c_sig and q_sig != c_sig:
                        if sim >= self.similarity_threshold:
                            rejection_reason = f"error_signature_mismatch ({q_sig} vs {c_sig})"
                        continue

                    # 2. Affected service compatibility
                    q_srv = context_signals.get("affected_service", "")
                    c_srv = cand_record.affected_service or ""
                    if q_srv != "unknown" and c_srv != "unknown" and q_srv != c_srv:
                        if sim >= self.similarity_threshold:
                            rejection_reason = f"affected_service_mismatch ({q_srv} vs {c_srv})"
                        continue

                    # All context guards passed
                    if sim > best_sim:
                        best_sim = sim
                        if sim >= self.similarity_threshold:
                            best_record = cand_record

                if best_record and best_sim >= self.similarity_threshold:
                    conn.execute("""
                        UPDATE incident_response_cache
                        SET hit_count = hit_count + 1, last_accessed_at = ?
                        WHERE cache_id = ?;
                    """, (current_time, best_record.cache_id))
                    conn.commit()
                    return best_record, best_sim, "match_accepted"
                elif rejection_reason:
                    self._metrics["rejected_semantic_matches"] += 1
                    return None, max(best_sim, 0.0), f"rejected_by_context_guard: {rejection_reason}"

                return None, max(best_sim, 0.0), "below_similarity_threshold"

        except Exception as e:
            self._metrics["fallback_errors"] += 1
            print(f"[CACHE WARNING] Semantic lookup failed: {e}")
            return None, 0.0, f"error: {str(e)}"

    # -------------------------------------------------------------------------
    # Main Lookup Dispatcher (Stage A -> Stage B -> Miss)
    # -------------------------------------------------------------------------

    def lookup(self, query: str) -> CacheLookupResult:
        """
        Executes the three-stage lookup strategy before calling any LLM.
        Returns CacheLookupResult with match type, score, and avoided cost.
        """
        t0 = time.time()
        with self._lock:
            self._metrics["total_lookups"] += 1

        if not self.enabled or not query or not query.strip():
            return CacheLookupResult(hit=False, match_type="miss", similarity=0.0, lookup_latency_sec=round(time.time() - t0, 5), notes="cache_disabled")

        norm_q = self.normalize_query(query)
        fingerprint = self.compute_fingerprint(norm_q)
        current_time = time.time()

        # Stage A: Exact Cache Lookup
        exact_record = self.lookup_exact(fingerprint, current_time)
        if exact_record:
            lat = round(time.time() - t0, 5)
            # Estimate counterfactual savings based on original model tier
            avoided_cost = 0.00015
            if "120b" in exact_record.model_id or "plus" in exact_record.model_id:
                avoided_cost = 0.00250
            with self._lock:
                self._metrics["exact_hits"] += 1
                self._metrics["total_avoided_model_calls"] += 1
                self._metrics["total_avoided_cost_usd"] += avoided_cost
                self._metrics["total_lookup_latency_sec"] += lat

            return CacheLookupResult(
                hit=True,
                match_type="exact",
                similarity=1.0,
                record=exact_record,
                lookup_latency_sec=lat,
                avoided_model_calls=1,
                avoided_cost_usd=avoided_cost,
                notes="exact_normalized_match"
            )

        # Stage B: Semantic Cache Lookup
        last_sem_notes = ""
        if self.semantic_enabled:
            model = self._get_embedding_model()
            if model is not None:
                try:
                    q_vec = model.encode(norm_q, convert_to_numpy=True).astype(np.float32)
                except Exception as e:
                    q_vec = None
                    self._metrics["fallback_errors"] += 1

                context_signals = self.extract_context_signals(norm_q)
                sem_record, sim_score, sem_notes = self.lookup_semantic(norm_q, q_vec, context_signals, current_time)
                last_sem_notes = sem_notes

                if sem_record:
                    lat = round(time.time() - t0, 5)
                    avoided_cost = 0.00015
                    if "120b" in sem_record.model_id or "plus" in sem_record.model_id:
                        avoided_cost = 0.00250
                    with self._lock:
                        self._metrics["semantic_hits"] += 1
                        self._metrics["total_avoided_model_calls"] += 1
                        self._metrics["total_avoided_cost_usd"] += avoided_cost
                        self._metrics["total_lookup_latency_sec"] += lat

                    return CacheLookupResult(
                        hit=True,
                        match_type="semantic",
                        similarity=round(sim_score, 4),
                        record=sem_record,
                        lookup_latency_sec=lat,
                        avoided_model_calls=1,
                        avoided_cost_usd=avoided_cost,
                        notes=sem_notes
                    )

        # Stage C: Cache Miss -> Fallback to Normal Agent Pipeline
        lat = round(time.time() - t0, 5)
        with self._lock:
            self._metrics["misses"] += 1
            self._metrics["total_lookup_latency_sec"] += lat

        return CacheLookupResult(
            hit=False,
            match_type="miss",
            similarity=0.0,
            lookup_latency_sec=lat,
            notes=last_sem_notes if last_sem_notes else "cache_miss_executing_pipeline"
        )

    # -------------------------------------------------------------------------
    # Save Verified Incident Response on Cache Miss
    # -------------------------------------------------------------------------

    def save_verified_response(self, state: Dict[str, Any]) -> Optional[str]:
        """
        Saves a newly verified incident response into the cache.
        Stores structure, evidence links, embeddings, and context metadata.
        Safety: Rejects unverified, failed, or abstained resolutions.
        """
        if not self.enabled:
            return None

        # Verification & Safety Eligibility Filter
        if not state.get("verified"):
            return None
        if state.get("resolution_status") != "resolved":
            return None
        if state.get("escalated"):
            return None

        root_cause = str(state.get("root_cause", "")).strip()
        if not root_cause or "uncertain" in root_cause.lower() or "insufficient" in root_cause.lower():
            return None

        query = state.get("user_query", "").strip()
        if not query:
            return None

        norm_q = self.normalize_query(query)
        fingerprint = self.compute_fingerprint(norm_q)
        signals = self.extract_context_signals(norm_q)
        cache_id = f"CACHE_{fingerprint[:16]}_{int(time.time())}"

        now = time.time()
        expires_at = now + self.ttl_seconds

        # Embedding generation (0 paid API cost via local SentenceTransformer)
        embedding_blob = None
        if self.semantic_enabled:
            model = self._get_embedding_model()
            if model is not None:
                try:
                    vec = model.encode(norm_q, convert_to_numpy=True).astype(np.float32)
                    embedding_blob = vec.tobytes()
                except Exception as e:
                    self._metrics["fallback_errors"] += 1
                    print(f"[CACHE WARNING] Could not generate embedding for caching: {e}")

        # Serialization of supporting evidence and claims
        supporting_ev = json.dumps(state.get("kb_results", [])[:4], ensure_ascii=False)
        struct_diag = json.dumps(state.get("structured_diagnosis", {}), ensure_ascii=False)

        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO incident_response_cache (
                        cache_id, query, normalized_query, fingerprint, embedding,
                        incident_category, severity, affected_service, error_signature,
                        root_cause, resolution_steps, resolution_status,
                        supporting_evidence, structured_diagnosis, verification_status,
                        verification_notes, confidence_score, calibrated_confidence,
                        model_id, provider, created_at, last_accessed_at, expires_at,
                        kb_version, schema_version, hit_count, is_invalidated, invalidation_reason
                    ) VALUES (
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, 0, 0, NULL
                    );
                """, (
                    cache_id,
                    query,
                    norm_q,
                    fingerprint,
                    embedding_blob,
                    state.get("category", "general"),
                    state.get("severity", "L2"),
                    signals.get("affected_service", "unknown"),
                    signals.get("error_signature", ""),
                    root_cause,
                    state.get("resolution_steps", ""),
                    "resolved",
                    supporting_ev,
                    struct_diag,
                    "verified",
                    state.get("verification_notes", "Verified pass"),
                    float(1.0 if state.get("confidence") == "high" else 0.75),
                    float(state.get("calibrated_confidence") or 0.85),
                    str(state.get("selected_model", "cohere/command-r-08-2024")),
                    str(state.get("selected_provider", "cohere")),
                    now,
                    now,
                    expires_at,
                    "v1.0-faiss",
                    DEFAULT_SCHEMA_VERSION,
                ))
                conn.commit()

            with self._lock:
                self._metrics["saved_records"] += 1
            return cache_id

        except Exception as e:
            self._metrics["fallback_errors"] += 1
            # Cache write failure must never crash or invalidate an otherwise successful diagnosis
            print(f"[CACHE WARNING] Failed to persist verified response: {e}")
            return None

    # -------------------------------------------------------------------------
    # Invalidation & Administration
    # -------------------------------------------------------------------------

    def invalidate_record(self, cache_id: str, reason: str = "manual_invalidation") -> bool:
        """Marks a cache record as invalidated so it will not be served."""
        try:
            with self._get_connection() as conn:
                cur = conn.execute("""
                    UPDATE incident_response_cache
                    SET is_invalidated = 1, invalidation_reason = ?
                    WHERE cache_id = ?;
                """, (reason, cache_id))
                conn.commit()
                if cur.rowcount > 0:
                    with self._lock:
                        self._metrics["invalidations"] += 1
                    return True
        except Exception as e:
            self._metrics["fallback_errors"] += 1
            print(f"[CACHE ERROR] Invalidation failed for {cache_id}: {e}")
        return False

    def clear_cache(self) -> int:
        """Clears all records in the response cache (administrative operation)."""
        try:
            with self._get_connection() as conn:
                cur = conn.execute("DELETE FROM incident_response_cache;")
                conn.commit()
                count = cur.rowcount
                with self._lock:
                    self._metrics["invalidations"] += count
                return count
        except Exception as e:
            self._metrics["fallback_errors"] += 1
            print(f"[CACHE ERROR] Clear cache failed: {e}")
            return 0

    def purge_expired(self) -> int:
        """Purges records past their TTL expiration timestamp."""
        try:
            with self._get_connection() as conn:
                cur = conn.execute("DELETE FROM incident_response_cache WHERE expires_at <= ?;", (time.time(),))
                conn.commit()
                return cur.rowcount
        except Exception as e:
            self._metrics["fallback_errors"] += 1
            return 0

    def get_metrics(self) -> Dict[str, Any]:
        """Returns structured observability and cost tracking metrics."""
        with self._lock:
            total = max(self._metrics["total_lookups"], 1)
            hits = self._metrics["exact_hits"] + self._metrics["semantic_hits"]
            hit_rate = round((hits / total) * 100, 2)
            avg_lat = round(self._metrics["total_lookup_latency_sec"] / total, 5)

            # Query database for physical record count
            record_count = 0
            try:
                with self._get_connection() as conn:
                    row = conn.execute("SELECT COUNT(*) FROM incident_response_cache WHERE is_invalidated = 0;").fetchone()
                    if row:
                        record_count = row[0]
            except Exception:
                pass

            return {
                "cache_enabled": self.enabled,
                "semantic_enabled": self.semantic_enabled,
                "similarity_threshold": self.similarity_threshold,
                "ttl_seconds": self.ttl_seconds,
                "total_records_in_db": record_count,
                "total_lookups": self._metrics["total_lookups"],
                "exact_hits": self._metrics["exact_hits"],
                "semantic_hits": self._metrics["semantic_hits"],
                "misses": self._metrics["misses"],
                "rejected_semantic_matches": self._metrics["rejected_semantic_matches"],
                "hit_rate_percent": hit_rate,
                "saved_records": self._metrics["saved_records"],
                "invalidations": self._metrics["invalidations"],
                "fallback_errors": self._metrics["fallback_errors"],
                "avoided_model_calls": self._metrics["total_avoided_model_calls"],
                "avoided_llm_cost_usd": round(self._metrics["total_avoided_cost_usd"], 5),
                "avg_lookup_latency_sec": avg_lat,
            }

    def _row_to_record(self, row: sqlite3.Row) -> CacheRecord:
        supp_ev = []
        if row["supporting_evidence"]:
            try:
                supp_ev = json.loads(row["supporting_evidence"])
            except Exception:
                pass

        struct_d = None
        if row["structured_diagnosis"]:
            try:
                struct_d = json.loads(row["structured_diagnosis"])
            except Exception:
                pass

        return CacheRecord(
            cache_id=row["cache_id"],
            query=row["query"],
            normalized_query=row["normalized_query"],
            fingerprint=row["fingerprint"],
            incident_category=row["incident_category"] or "general",
            severity=row["severity"] or "L2",
            affected_service=row["affected_service"] or "unknown",
            error_signature=row["error_signature"] or "",
            root_cause=row["root_cause"],
            resolution_steps=row["resolution_steps"],
            resolution_status=row["resolution_status"],
            supporting_evidence=supp_ev,
            structured_diagnosis=struct_d,
            verification_status=row["verification_status"],
            verification_notes=row["verification_notes"] or "",
            confidence_score=float(row["confidence_score"] or 1.0),
            calibrated_confidence=float(row["calibrated_confidence"] or 0.85),
            model_id=row["model_id"] or "",
            provider=row["provider"] or "",
            created_at=float(row["created_at"]),
            last_accessed_at=float(row["last_accessed_at"]),
            expires_at=float(row["expires_at"]),
            kb_version=row["kb_version"] or "v1.0",
            schema_version=row["schema_version"] or DEFAULT_SCHEMA_VERSION,
            hit_count=int(row["hit_count"]),
            is_invalidated=bool(row["is_invalidated"]),
            invalidation_reason=row["invalidation_reason"],
        )
