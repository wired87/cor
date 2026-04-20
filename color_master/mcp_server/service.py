from __future__ import annotations

import sys
from pathlib import Path

# gien: same ``cor/`` bootstrap as mcp_routes when this module is imported alone.
_repo_root = Path(__file__).resolve().parent.parent.parent
_core = str(_repo_root / "cor")
if _core not in sys.path:
    sys.path.insert(0, _core)

import base64
import importlib.util
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
import networkx as nx

from qbrain import GUtils
from mcp_server.types import (
    DeleteRequest,
    DeleteResponse,
    EntryResponse,
    GraphEdgeOut,
    GraphNodeOut,
    GraphResponse,
    UpsertRequest,
    UpsertResponse,
)



class MCPServerService:
    _MAX_PDF_SCAN_BYTES = 400_000
    _MAX_TEXT_ANALYZE_CHARS = 120_000
    def __init__(self, db) -> None:
        """Initialize DB access, extractor cache, and in-memory working graph."""
        self.db = db
        self._eq_extractor_cls = None
        # Keep extraction graph lightweight: no history/datastore writes needed here.
        self.g = GUtils(G=nx.MultiGraph(), nx_only=True, enable_data_store=False)

    @staticmethod
    def _now() -> datetime:
        """Return UTC timestamp used for persisted records."""
        return datetime.utcnow()

    @staticmethod
    def _as_text(payload_bytes: bytes) -> str:
        """Decode bytes to text with UTF-8 fallback to latin-1."""
        try:
            return payload_bytes.decode("utf-8")
        except Exception:
            print("Err color_master.mcp_server.service::MCPServerService._as_text | handler_line=54 | Exception handler triggered")
            print("[exception] color_master.mcp_server.service.MCPServerService._as_text: caught Exception")
            return payload_bytes.decode("latin-1", errors="ignore")

    @staticmethod
    def _to_file_bytes(file_payload: Any) -> bytes:
        """
        Normalize incoming file payload into raw bytes.

        Most clients send base64 text. Decoding here prevents equation extraction
        from scanning huge base64 blobs, which can make parsing appear "stuck".
        """
        if isinstance(file_payload, bytes):
            return file_payload
        if isinstance(file_payload, str):
            text = file_payload.strip()
            if not text:
                return b""
            try:
                # validate=True keeps accidental non-base64 text from being decoded.
                return base64.b64decode(text, validate=True)
            except Exception:
                # Fallback for plain-text content.
                print("Err color_master.mcp_server.service::MCPServerService._to_file_bytes | handler_line=75 | Exception handler triggered")
                print("[exception] color_master.mcp_server.service.MCPServerService._to_file_bytes: caught Exception")
                return text.encode("utf-8", errors="ignore")
        return str(file_payload).encode("utf-8", errors="ignore")

    @staticmethod
    def _extract_text_from_pdf_bytes(payload_bytes: bytes or str) -> str:
        """
        Fast, bounded PDF text extraction.

        Why bounded:
        - Full-PDF regex scans on binary streams can be very slow.
        - For equation detection we only need a representative text slice.
        """
        if isinstance(payload_bytes, str):
            payload_bytes = payload_bytes.encode("latin-1", errors="ignore")
        raw_slice = bytes(payload_bytes[:MCPServerService._MAX_PDF_SCAN_BYTES])
        raw = MCPServerService._as_text(raw_slice)

        print(
            f"get chunks... scan_bytes={len(raw_slice)} of total={len(payload_bytes)}",
            file=sys.stderr,
        )
        chunks: List[str] = []
        # Single-pass token capture is much faster than nested block scans.
        chunks.extend(re.findall(r"\(([^)]{1,500})\)\s*Tj", raw, flags=re.DOTALL))
        for arr in re.findall(r"\[([^\]]{1,3000})\]\s*TJ", raw, flags=re.DOTALL):
            chunks.extend(re.findall(r"\(([^)]{1,500})\)", arr, flags=re.DOTALL))

        if not chunks:
            return raw

        cleaned = []
        for c in chunks:
            txt = c.replace(r"\(", "(").replace(r"\)", ")").replace(r"\\", "\\")
            txt = " ".join(txt.split())
            if txt:
                cleaned.append(txt)
        finalized = "\n".join(cleaned) if cleaned else raw
        print(
            f"get chunks... extracted_tokens={len(cleaned)} text_len={len(finalized)}",
            file=sys.stderr,
        )
        print("get chunks... done", file=sys.stderr)
        return finalized


    def _load_eq_extractor_class(self):
        """Lazily import and cache EqExtractor class from math/eq_extractor.py."""
        if self._eq_extractor_cls is not None:
            return self._eq_extractor_cls
        try:
            eq_path = Path(__file__).resolve().parents[1] / "math" / "eq_extractor.py"
            spec = importlib.util.spec_from_file_location("eq_storage_math_eq_extractor", str(eq_path))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self._eq_extractor_cls = getattr(mod, "EqExtractor", None)
        except Exception:
            print("Err color_master.mcp_server.service::MCPServerService._load_eq_extractor_class | handler_line=134 | Exception handler triggered")
            print("[exception] color_master.mcp_server.service.MCPServerService._load_eq_extractor_class: caught Exception")
            self._eq_extractor_cls = None
        return self._eq_extractor_cls

    def _extract_content_parts(self, text: str, file_name, user_id):
        """
        Run the equation extractor and append results into the internal graph.

        Args:
            text: Source text to analyze.
            file_name: Context/module identifier attached to extracted nodes.
            user_id: Owning user identifier used in graph metadata.
        """
        EqExtractorCls = self._load_eq_extractor_class()
        if EqExtractorCls is None:
            return []

        try:
            if len(text) > self._MAX_TEXT_ANALYZE_CHARS:
                print(
                    f"_extract_content_parts: truncating text {len(text)} -> "
                    f"{self._MAX_TEXT_ANALYZE_CHARS} chars for bounded latency",
                    file=sys.stderr,
                )
                text = text[: self._MAX_TEXT_ANALYZE_CHARS]
            extractor = EqExtractorCls(debug=False)
            extractor.text_to_multigraph(
                text=text,
                context_id=file_name,
                module_id=file_name,
                user_id=user_id,
                g=self.g,
            )
            print("extracted", len(self.g.G.nodes), "nodes", file=sys.stderr)

        except Exception as e:
            print(f"Err color_master.mcp_server.service::MCPServerService._extract_content_parts | handler_line=170 | {type(e).__name__}: {e}")
            print(f"[exception] color_master.mcp_server.service.MCPServerService._extract_content_parts: {e}")
            print("Err", e, file=sys.stderr)


    def get_graph(self, user_id: str, test: bool = False) -> GraphResponse:
        """
        Build and return a graph view for a user from persisted data.

        Args:
            user_id: User identifier for graph filtering.
            test: If true, keep graph artifacts in project output directory.
        """
        print("[MCPServerService.get_graph] START", file=sys.stderr)
        print(f"[MCPServerService.get_graph] LOGIC_GATE user_id={user_id} test={test}", file=sys.stderr)
        if not user_id:
            return GraphResponse(status="error", user_id="", stats={"message": "user_id is required"})

        try:

            visualizer = VisualizeGraph(db=self.db)
            result = visualizer.run(user_id=user_id, test=test)

            nodes = [
                GraphNodeOut(id=str(n.get("id") or ""), attrs=dict(n.get("attrs") or {}))
                for n in (result.get("nodes") or [])
            ]
            edges = [
                GraphEdgeOut(
                    source=str(e.get("src") or ""),
                    target=str(e.get("trgt") or ""),
                    attrs=dict(e.get("attrs") or {}),
                )
                for e in (result.get("edges") or [])
            ]

            stats = dict(result.get("stats") or {})
            stats["artifacts"] = result.get("artifacts") or {}
            print("[MCPServerService.get_graph] END ok", file=sys.stderr)
            return GraphResponse(
                status="ok",
                user_id=user_id,
                nodes=nodes,
                edges=edges,
                stats=stats,
            )
        except Exception as exc:
            print(f"Err color_master.mcp_server.service::MCPServerService.get_graph | handler_line=217 | {type(exc).__name__}: {exc}")
            print(f"[exception] color_master.mcp_server.service.MCPServerService.get_graph: {exc}")
            print(f"[MCPServerService.get_graph] END error={exc}", file=sys.stderr)
            return GraphResponse(
                status="error",
                user_id=user_id,
                stats={"message": str(exc)},
            )

    def upsert(self, request: UpsertRequest):
        """
        Ingest text/files, extract graph entities, and upsert rows into storage.
        Args:
            request: Upsert payload containing user_id, files, and optional equation.
        """
        print("upsert...", file=sys.stderr)
        # Reset per request so runtime stays stable and does not grow unbounded.
        self.g = GUtils(G=nx.MultiGraph(), nx_only=True, enable_data_store=False)
        user_id =request.user_id
        if not request.user_id:
            return UpsertResponse(status="error", message="user_id is required")

        normalized = [
            (f"file_{user_id}_{generate_id(20)}", self._to_file_bytes(file))
            for file in request.data.files
        ]
        print(f"files normalized count={len(normalized)}", file=sys.stderr)

        file_ids: List[str] = []
        file_rows: List[Dict[str, Any]] = []
        method_rows: List[Dict[str, Any]] = []
        param_rows: List[Dict[str, Any]] = []
        operator_rows: List[Dict[str, Any]] = []

        for file_id, file_bytes in normalized:
            print(f"work file {file_id}", file=sys.stderr)
            file_text = (
                self._extract_text_from_pdf_bytes(file_bytes)
                if file_bytes.startswith(b"%PDF")
                else self._as_text(file_bytes)
            )
            print(f"file_text len={len(file_text)}", file=sys.stderr)

            # EXTRACT EQUATIONS
            self._extract_content_parts(file_text, file_id, user_id)
            print("_extract_content_parts... done", file=sys.stderr)

            file_ids.append(file_id)
            file_rows.append(
                {
                    "id": file_id,
                    "user_id": request.user_id,
                    # Store as base64 text to avoid DB unicode decoding errors.
                    "content": base64.b64encode(file_bytes).decode("ascii"),
                    "created_at": self._now(),
                }
            )
            print("file id... done", file=sys.stderr)


        # EXTRACT EQUATIONS
        if request.data.equation:
            self._extract_content_parts(request.data.equation, f"{user_id}_{generate_id(20)}", user_id)
            print("request.data.equation... done", file=sys.stderr)

        # Map EqExtractor node types to DB tables: EQUATION->methods, SYMBOL->params,
        # OPERATOR->operators. METHOD/PARAM from other sources also supported.
        check = {}
        for k, v in self.g.G.nodes(data=True):
            ntype = (v.get("type") or "").upper()
            if not ntype:
                continue
            if ntype == "METHOD" or ntype == "EQUATION":
                v["param_neighbors"] = self.g.get_neighbor_list(node=k, target_type=["PARAM", "SYMBOL"], just_ids=True) or []
                v["operator_neighbors"] = self.g.get_neighbor_list(node=k, target_type="OPERATOR", just_ids=True) or []
                if ntype not in check:
                    check[ntype] = []
                check[ntype].append(k)
                method_rows.append({"id": k, **v})

            elif ntype == "PARAM" or ntype == "SYMBOL":
                v["method_neighbors"] = self.g.get_neighbor_list(node=k, target_type=["METHOD", "EQUATION"], just_ids=True) or []
                v["operator_neighbors"] = self.g.get_neighbor_list(node=k, target_type="OPERATOR", just_ids=True) or []
                if ntype not in check:
                    check[ntype] = []
                check[ntype].append(k)
                param_rows.append({"id": k, **v})

            elif ntype == "OPERATOR":
                v["method_neighbors"] = self.g.get_neighbor_list(node=k, target_type=["METHOD", "EQUATION"], just_ids=True) or []
                v["operator_neighbors"] = self.g.get_neighbor_list(node=k, target_type="OPERATOR", just_ids=True) or []
                if ntype not in check:
                    check[ntype] = []
                check[ntype].append(k)
                operator_rows.append({"id": k, **v})

        # EDGES
        edge_rows = []
        for src, trgt, attrs in self.g.G.edges(data=True):
            edge_rows.append({"src":src,"trgt":trgt,"attrs":attrs})

        self.db.insert("edges", edge_rows)
        self.db.insert("methods", method_rows)
        self.db.insert("params", param_rows)
        self.db.insert("operators", operator_rows)
        self.db.insert("files", file_rows)
        print(
            "[upsert] graph summary "
            f"nodes={len(self.g.G.nodes)} edges={len(self.g.G.edges)} "
            f"methods={len(method_rows)} params={len(param_rows)} operators={len(operator_rows)}",
            file=sys.stderr,
        )
        print("upsert... done", file=sys.stderr)

    def get_entry(self, entry_id: str, table: str = "methods", user_id: Optional[str] = None) -> EntryResponse:
        """
        Fetch a single entry by id from a table, optionally scoped by user.

        Args:
            entry_id: Primary identifier of the requested row.
            table: Target table name.
            user_id: Optional owner filter.
        """
        try:
            row = self.db.row_from_id(entry_id, table=table, user_id=user_id)
        except ValueError as exc:
            print(f"Err color_master.mcp_server.service::MCPServerService.get_entry | handler_line=342 | {type(exc).__name__}: {exc}")
            print(f"[exception] color_master.mcp_server.service.MCPServerService.get_entry: {exc}")
            return EntryResponse(status="error", table=table, message=str(exc))
        if not row:
            return EntryResponse(status="not_found1", table=table, message="Entry not found")
        return EntryResponse(status="ok", entry=row, table=table)


    def delete_entries(self, request: DeleteRequest) -> DeleteResponse:
        """
        Delete a single entry or all entries for a user in a table.
        Args:
            request: Delete payload with user_id, table, and optional entry_id.
        """
        if not request.user_id:
            return DeleteResponse(status="error", message="user_id is required")
        if request.entry_id:
            deleted:int = self.db.del_entry(id=request.entry_id, table=request.table, user_id=request.user_id, )
            return DeleteResponse(status="ok", deleted_count=deleted, mode="single")
        self.db.delete(table=request.table, where_clause="user_id = ?", params=[request.user_id])
        return DeleteResponse(status="ok", deleted_count=-1, mode="all")
