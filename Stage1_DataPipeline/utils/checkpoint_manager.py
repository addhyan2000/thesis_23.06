"""
checkpoint_manager.py — Atomic Checkpoint Persistence
=====================================================

Implements ``CheckpointManager``, which maintains a JSON file
(``processed_state.json``) recording exactly which processing blocks
have been completed.  If the pipeline crashes or is interrupted, it
reads this file on restart and **skips** every block already marked
as ``"completed"``.

Atomicity Strategy:
    We write to a temporary ``.tmp`` file first, then atomically
    rename it to the real checkpoint path.  On Windows ``os.replace``
    is atomic at the filesystem level, guaranteeing we never end up
    with a half-written state file after a power failure.

State Schema (JSON)::

    {
        "version": 1,
        "created_at": "2026-04-01T18:00:00",
        "last_updated": "2026-04-01T18:05:32",
        "blocks": {
            "CASME_II_metadata": "completed",
            "CASME2_Squared_metadata": "completed",
            "merge_and_export": "pending"
        }
    }

Author  : Addhyan
Stage   : 1 — Data Pipeline
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from utils.logger import get_logger


class CheckpointManager:
    """
    Manages pipeline checkpoint state via an atomic JSON file.

    Parameters
    ----------
    checkpoint_path : Path
        Absolute path to the ``processed_state.json`` file.
        Parent directories are created automatically.

    Attributes
    ----------
    state : dict
        In-memory representation of the checkpoint file.
    """

    # ── Class-level constants ───────────────────────────────────────
    _STATE_VERSION: int = 1
    _STATUS_PENDING: str = "pending"
    _STATUS_COMPLETED: str = "completed"
    _STATUS_FAILED: str = "failed"

    def __init__(self, checkpoint_path: Path) -> None:
        self._path: Path = checkpoint_path
        self._log = get_logger(self.__class__.__name__)
        self.state: Dict[str, Any] = self._load_or_create()

    # ─────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────

    def is_completed(self, block_name: str) -> bool:
        """
        Check whether *block_name* has already been completed.

        Parameters
        ----------
        block_name : str
            Logical processing block (e.g. ``"CASME_II_metadata"``).

        Returns
        -------
        bool
            ``True`` if the block's status is ``"completed"``.
        """
        status = self.state.get("blocks", {}).get(block_name)
        completed = status == self._STATUS_COMPLETED
        if completed:
            self._log.info(
                "Checkpoint HIT - block '%s' already completed. Skipping.",
                block_name,
            )
        return completed

    def mark_completed(self, block_name: str) -> None:
        """
        Mark *block_name* as completed and persist atomically.

        Parameters
        ----------
        block_name : str
            Logical processing block to mark.
        """
        self.state.setdefault("blocks", {})[block_name] = self._STATUS_COMPLETED
        self.state["last_updated"] = datetime.now().isoformat()
        self._save()
        self._log.info(
            "Checkpoint SAVED - block '%s' marked completed.", block_name
        )

    def mark_failed(self, block_name: str, reason: str = "") -> None:
        """
        Mark *block_name* as failed (so it will be retried next run).

        Parameters
        ----------
        block_name : str
            Logical processing block that failed.
        reason : str, optional
            Human-readable failure reason (logged, not persisted).
        """
        self.state.setdefault("blocks", {})[block_name] = self._STATUS_FAILED
        self.state["last_updated"] = datetime.now().isoformat()
        self._save()
        self._log.warning(
            "Checkpoint SAVED - block '%s' marked FAILED. Reason: %s",
            block_name,
            reason or "(none)",
        )

    def mark_pending(self, block_name: str) -> None:
        """
        Explicitly set *block_name* to pending (useful for resets).

        Parameters
        ----------
        block_name : str
            Logical processing block to reset.
        """
        self.state.setdefault("blocks", {})[block_name] = self._STATUS_PENDING
        self.state["last_updated"] = datetime.now().isoformat()
        self._save()
        self._log.info(
            "Checkpoint RESET - block '%s' set to pending.", block_name
        )

    def get_status(self, block_name: str) -> Optional[str]:
        """Return the raw status string, or ``None`` if untracked."""
        return self.state.get("blocks", {}).get(block_name)

    def summary(self) -> str:
        """Return a human-readable summary of all block statuses."""
        blocks = self.state.get("blocks", {})
        if not blocks:
            return "No blocks registered yet."
        lines = [f"  - {name:40s} -> {status}" for name, status in blocks.items()]
        return "Checkpoint Summary:\n" + "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────
    #  Private helpers
    # ─────────────────────────────────────────────────────────────────

    def _load_or_create(self) -> Dict[str, Any]:
        """
        Load existing checkpoint or create a fresh one.

        Returns
        -------
        dict
            The checkpoint state dictionary.
        """
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self._log.info(
                    "Loaded existing checkpoint from %s  "
                    "(last updated: %s, blocks: %d)",
                    self._path,
                    data.get("last_updated", "unknown"),
                    len(data.get("blocks", {})),
                )
                return data
            except (json.JSONDecodeError, KeyError) as exc:
                self._log.error(
                    "Corrupt checkpoint file at %s - re-creating. Error: %s",
                    self._path,
                    exc,
                )
        # ── Fresh state ─────────────────────────────────────────────
        now = datetime.now().isoformat()
        fresh: Dict[str, Any] = {
            "version": self._STATE_VERSION,
            "created_at": now,
            "last_updated": now,
            "blocks": {},
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._save(fresh)
        self._log.info("Created new checkpoint file at %s", self._path)
        return fresh

    def _save(self, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Atomically persist *data* (or ``self.state``) to disk.

        Strategy: write to ``.tmp`` → ``os.replace`` → done.
        ``os.replace`` is atomic on both POSIX and modern Windows
        (NTFS), guaranteeing no half-written files.
        """
        payload = data if data is not None else self.state
        tmp_path = self._path.with_suffix(".tmp")

        self._path.parent.mkdir(parents=True, exist_ok=True)

        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())            # Force write to physical disk

        os.replace(str(tmp_path), str(self._path))   # Atomic rename
