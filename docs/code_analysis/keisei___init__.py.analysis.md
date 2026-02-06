# Analysis: keisei/__init__.py

**Lines:** 30
**Role:** Package entry point. Defines the public API surface for the `keisei` package by re-exporting core Shogi types (`Color`, `PieceType`, `Piece`, `MoveTuple`, `ShogiGame`) and the `EvaluationManager`.
**Key dependencies:**
- Imports from: `keisei.evaluation.core_manager.EvaluationManager`, `keisei.shogi.shogi_core_definitions` (Color, MoveTuple, Piece, PieceType), `keisei.shogi.shogi_game.ShogiGame`
- Imported by: Any code doing `from keisei import ...` or `import keisei`
**Analysis depth:** FULL

## Summary

This file is small and structurally sound, but contains one significant design issue: the top-level import of `EvaluationManager` directly contradicts the comment explaining why it should NOT be imported at this level. This forces a transitive import chain through the evaluation subsystem, pulling in `torch`, `asyncio`, and many other heavy dependencies on any `import keisei`. No critical bugs, but the design contradicts stated intent.

## Warnings

### [15] EvaluationManager import contradicts its own comment and pulls in heavy dependencies

**What:** Lines 11-14 contain a comment explicitly stating that legacy evaluation entrypoints "are not imported here to avoid heavy dependencies (e.g. `torch`) when the top-level package is imported during lightweight operations such as running unit tests." Yet line 15 immediately imports `EvaluationManager` from `keisei.evaluation.core_manager`, which itself imports `torch` (line 91 of `core_manager.py`, inside `evaluate_checkpoint`), `asyncio`, `strategies` (which triggers factory registration and further imports), `OpponentPool`, and `EvaluationPerformanceManager`.

**Why it matters:** Any `import keisei` statement -- including during test discovery, documentation generation, or lightweight CLI tools -- will trigger the full evaluation subsystem import chain. The `core_manager.py` file does `from . import strategies` at module level (line 13), which forces all strategy modules to load and register. While `torch` is imported lazily inside methods in `core_manager.py`, the strategy modules and their dependencies are not lazy. This increases startup time and memory usage for operations that never use evaluation, and can cause import errors in environments where optional dependencies are not installed.

**Evidence:**
```python
# Lines 11-15:
# Import the new evaluation manager. Legacy evaluation entrypoints are
# available via ``keisei.evaluation.evaluate`` but are not imported here to
# avoid heavy dependencies (e.g. ``torch``) when the top-level package is
# imported during lightweight operations such as running unit tests.
from .evaluation.core_manager import EvaluationManager
```

The comment describes the exact problem the code then creates.

### [18] Re-exported types create a coupling surface between the package API and internal module paths

**What:** The `__init__.py` re-exports `Color`, `MoveTuple`, `Piece`, `PieceType`, and `ShogiGame` as the package's public API. However, these types are also available (and used) via their original module paths throughout the codebase.

**Why it matters:** If internal module paths change (e.g., `shogi_core_definitions` is renamed or reorganized), both the `__init__.py` and all direct importers must be updated. This is a maintenance burden but not a bug. The more pressing concern is that `__all__` lists these names, suggesting they ARE the public API, but the actual downstream code (training managers, evaluation strategies, tests) typically imports from the specific submodules directly, creating two parallel import paths for the same types.

**Evidence:** Looking at the codebase, most files import from `keisei.shogi.shogi_core_definitions` directly, not from `keisei`. The re-exports exist but are not the primary consumption path.

## Observations

### [21] `__all__` is well-defined but incomplete for actual usage patterns

**What:** The `__all__` list includes 6 items. However, consumers of the `keisei` package may also expect access to `AppConfig`, `load_config`, or other commonly-used types. The comment "Let the other modules be imported explicitly" is appropriate but could be more specific about what constitutes the intended public API.

### [1-9] Docstring accurately describes the package contents

**What:** The module docstring is accurate and helpful. No issues.

## Verdict

**Status:** NEEDS_ATTENTION
**Recommended action:** The `EvaluationManager` import should either be removed from the top-level `__init__.py` (making it a lazy import or moving it to a subpackage-level import), or the comment should be removed/updated to reflect the actual design decision. The current state is contradictory and misleading to maintainers. This is a low-effort fix with meaningful impact on import performance and clarity.
**Confidence:** HIGH -- The contradictory comment and import are unambiguous, and the transitive dependency chain is verifiable from the source.
