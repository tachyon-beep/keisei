# Codex Bug Hunt Log

| Timestamp (UTC) | Status | File | Output | Model | Duration_s | Note |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-04-05T05:06:30+00:00 | ok | keisei/sl/__init__.py | docs/bugs/generated/sl/__init__.py.md |  | 29.40 | evidence_gate=1 |
| 2026-04-05T05:06:32+00:00 | ok | keisei/server/__init__.py | docs/bugs/generated/server/__init__.py.md |  | 30.88 |  |
| 2026-04-05T05:06:32+00:00 | ok | keisei/__init__.py | docs/bugs/generated/__init__.py.md |  | 31.42 |  |
| 2026-04-05T05:07:07+00:00 | ok | keisei/sl/prepare.py | docs/bugs/generated/sl/prepare.py.md |  | 65.91 |  |
| 2026-04-05T05:07:27+00:00 | ok | keisei/sl/dataset.py | docs/bugs/generated/sl/dataset.py.md |  | 86.51 |  |
| 2026-04-05T05:07:40+00:00 | ok | keisei/server/app.py | docs/bugs/generated/server/app.py.md |  | 98.90 |  |
| 2026-04-05T05:07:46+00:00 | ok | keisei/config.py | docs/bugs/generated/config.py.md |  | 104.83 |  |
| 2026-04-05T05:07:57+00:00 | ok | keisei/sl/parsers.py | docs/bugs/generated/sl/parsers.py.md |  | 115.98 |  |
| 2026-04-05T05:08:22+00:00 | ok | keisei/sl/trainer.py | docs/bugs/generated/sl/trainer.py.md |  | 140.81 |  |
| 2026-04-05T05:08:34+00:00 | ok | keisei/db.py | docs/bugs/generated/db.py.md |  | 152.66 |  |
| 2026-04-05T05:09:18+00:00 | ok | keisei/training/__init__.py | docs/bugs/generated/training/__init__.py.md |  | 43.87 |  |
| 2026-04-05T05:09:29+00:00 | ok | keisei/training/distributed.py | docs/bugs/generated/training/distributed.py.md |  | 55.54 |  |
| 2026-04-05T05:09:33+00:00 | ok | keisei/training/gae.py | docs/bugs/generated/training/gae.py.md |  | 59.03 |  |
| 2026-04-05T05:09:33+00:00 | ok | keisei/training/algorithm_registry.py | docs/bugs/generated/training/algorithm_registry.py.md |  | 59.04 |  |
| 2026-04-05T05:09:39+00:00 | ok | keisei/training/evaluate.py | docs/bugs/generated/training/evaluate.py.md |  | 65.52 |  |
| 2026-04-05T05:10:17+00:00 | ok | keisei/training/frontier_promoter.py | docs/bugs/generated/training/frontier_promoter.py.md |  | 103.50 |  |
| 2026-04-05T05:10:34+00:00 | ok | keisei/training/checkpoint.py | docs/bugs/generated/training/checkpoint.py.md |  | 120.22 |  |
| 2026-04-05T05:10:37+00:00 | ok | keisei/training/concurrent_matches.py | docs/bugs/generated/training/concurrent_matches.py.md |  | 123.03 |  |
| 2026-04-05T05:10:45+00:00 | ok | keisei/training/demonstrator.py | docs/bugs/generated/training/demonstrator.py.md |  | 131.83 |  |
| 2026-04-05T05:10:47+00:00 | ok | keisei/training/dynamic_trainer.py | docs/bugs/generated/training/dynamic_trainer.py.md |  | 132.95 |  |
| 2026-04-05T05:11:17+00:00 | ok | keisei/training/models/__init__.py | docs/bugs/generated/training/models/__init__.py.md |  | 30.42 |  |
| 2026-04-05T05:11:45+00:00 | ok | keisei/training/models/base.py | docs/bugs/generated/training/models/base.py.md |  | 57.97 |  |
| 2026-04-05T05:11:45+00:00 | ok | keisei/training/models/katago_base.py | docs/bugs/generated/training/models/katago_base.py.md |  | 58.81 |  |
| 2026-04-05T05:12:13+00:00 | ok | keisei/training/historical_gauntlet.py | docs/bugs/generated/training/historical_gauntlet.py.md |  | 86.82 |  |
| 2026-04-05T05:12:17+00:00 | ok | keisei/training/match_scheduler.py | docs/bugs/generated/training/match_scheduler.py.md |  | 90.38 |  |
| 2026-04-05T05:12:18+00:00 | ok | keisei/training/historical_library.py | docs/bugs/generated/training/historical_library.py.md |  | 91.25 |  |
| 2026-04-05T05:12:32+00:00 | ok | keisei/training/match_utils.py | docs/bugs/generated/training/match_utils.py.md |  | 105.01 |  |
| 2026-04-05T05:12:35+00:00 | ok | keisei/training/katago_loop.py | docs/bugs/generated/training/katago_loop.py.md |  | 108.63 |  |
| 2026-04-05T05:12:41+00:00 | ok | keisei/training/model_registry.py | docs/bugs/generated/training/model_registry.py.md |  | 114.52 |  |
| 2026-04-05T05:13:00+00:00 | ok | keisei/training/katago_ppo.py | docs/bugs/generated/training/katago_ppo.py.md |  | 133.42 |  |
| 2026-04-05T05:13:41+00:00 | ok | keisei/training/priority_scorer.py | docs/bugs/generated/training/priority_scorer.py.md |  | 41.27 |  |
| 2026-04-05T05:14:12+00:00 | ok | keisei/training/models/se_resnet.py | docs/bugs/generated/training/models/se_resnet.py.md |  | 72.12 |  |
| 2026-04-05T05:14:16+00:00 | ok | keisei/training/models/transformer.py | docs/bugs/generated/training/models/transformer.py.md |  | 76.31 |  |
| 2026-04-05T05:14:29+00:00 | ok | keisei/training/models/resnet.py | docs/bugs/generated/training/models/resnet.py.md |  | 89.19 |  |
| 2026-04-05T05:14:31+00:00 | ok | keisei/training/role_elo.py | docs/bugs/generated/training/role_elo.py.md |  | 90.87 |  |
| 2026-04-05T05:14:37+00:00 | ok | keisei/training/tiered_pool.py | docs/bugs/generated/training/tiered_pool.py.md |  | 96.90 |  |
| 2026-04-05T05:14:42+00:00 | ok | keisei/training/tournament.py | docs/bugs/generated/training/tournament.py.md |  | 101.85 |  |
| 2026-04-05T05:14:58+00:00 | ok | keisei/training/models/mlp.py | docs/bugs/generated/training/models/mlp.py.md |  | 118.44 |  |
| 2026-04-05T05:14:59+00:00 | ok | keisei/training/opponent_store.py | docs/bugs/generated/training/opponent_store.py.md |  | 118.96 |  |
| 2026-04-05T05:15:12+00:00 | ok | keisei/training/tier_managers.py | docs/bugs/generated/training/tier_managers.py.md |  | 132.27 |  |
| 2026-04-05T05:16:55+00:00 | ok | keisei/training/value_adapter.py | docs/bugs/generated/training/value_adapter.py.md |  | 102.83 |  |
| 2026-04-05T05:17:28+00:00 | ok | keisei/training/transition.py | docs/bugs/generated/training/transition.py.md |  | 135.53 |  |
