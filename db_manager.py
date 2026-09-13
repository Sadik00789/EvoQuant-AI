"""
Deprecated compatibility shim (Phase 4 consolidation).

This module previously contained a divergent duplicate of the portfolio manager
with an inconsistent schema (`trade_logs.action VARCHAR(8)` vs the canonical
`VARCHAR(32)`, and no migration path). Two competing implementations of the same
persistence layer were a correctness hazard.

The canonical implementation now lives solely in `engine.CrossAssetPortfolioManager`
(aliased as `PostgresPortfolioManager`). This shim re-exports it so any legacy
import path continues to work against the single, migrated schema.
"""

from engine import CrossAssetPortfolioManager, PostgresPortfolioManager, logger  # noqa: F401

__all__ = ["PostgresPortfolioManager", "CrossAssetPortfolioManager", "logger"]
