# Codex Bug Hunt Log

| Timestamp (UTC) | Status | File | Output | Model | Duration_s | Note |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-04-05T11:30:34+00:00 | ok | keisei/server/__init__.py | docs/bugs/generated/server/__init__.py.md |  | 25.92 |  |
| 2026-04-05T11:30:35+00:00 | ok | keisei/__init__.py | docs/bugs/generated/__init__.py.md |  | 27.84 |  |
| 2026-04-05T11:31:04+00:00 | ok | keisei/sl/__init__.py | docs/bugs/generated/sl/__init__.py.md |  | 56.22 |  |
| 2026-04-05T11:31:17+00:00 | ok | keisei/sl/dataset.py | docs/bugs/generated/sl/dataset.py.md |  | 69.53 | evidence_gate=1 |
| 2026-04-05T11:31:25+00:00 | ok | keisei/sl/prepare.py | docs/bugs/generated/sl/prepare.py.md |  | 77.64 |  |
| 2026-04-05T11:32:02+00:00 | ok | keisei/sl/parsers.py | docs/bugs/generated/sl/parsers.py.md |  | 114.36 |  |
| 2026-04-05T11:32:04+00:00 | ok | keisei/server/app.py | docs/bugs/generated/server/app.py.md |  | 116.33 |  |
| 2026-04-05T11:32:07+00:00 | ok | keisei/sl/trainer.py | docs/bugs/generated/sl/trainer.py.md |  | 119.69 |  |
| 2026-04-05T11:32:20+00:00 | ok | keisei/config.py | docs/bugs/generated/config.py.md |  | 132.03 |  |
| 2026-04-05T11:32:33+00:00 | ok | keisei/db.py | docs/bugs/generated/db.py.md |  | 144.97 |  |
| 2026-04-05T11:33:03+00:00 | ok | keisei/training/__init__.py | docs/bugs/generated/training/__init__.py.md |  | 30.58 |  |
| 2026-04-05T11:33:38+00:00 | ok | keisei/training/gae.py | docs/bugs/generated/training/gae.py.md |  | 65.75 |  |
| 2026-04-05T11:33:56+00:00 | ok | keisei/training/frontier_promoter.py | docs/bugs/generated/training/frontier_promoter.py.md |  | 83.60 |  |
| 2026-04-05T11:33:58+00:00 | ok | keisei/training/checkpoint.py | docs/bugs/generated/training/checkpoint.py.md |  | 85.29 |  |
| 2026-04-05T11:34:11+00:00 | ok | keisei/training/algorithm_registry.py | docs/bugs/generated/training/algorithm_registry.py.md |  | 98.01 |  |
| 2026-04-05T11:34:20+00:00 | ok | keisei/training/distributed.py | docs/bugs/generated/training/distributed.py.md |  | 107.41 |  |
| 2026-04-05T11:34:27+00:00 | ok | keisei/training/dynamic_trainer.py | docs/bugs/generated/training/dynamic_trainer.py.md |  | 114.28 |  |
| 2026-04-05T11:34:30+00:00 | ok | keisei/training/concurrent_matches.py | docs/bugs/generated/training/concurrent_matches.py.md |  | 117.88 |  |
| 2026-04-05T11:35:15+00:00 | ok | keisei/training/evaluate.py | docs/bugs/generated/training/evaluate.py.md |  | 161.99 |  |
| 2026-04-05T11:35:21+00:00 | ok | keisei/training/demonstrator.py | docs/bugs/generated/training/demonstrator.py.md |  | 168.67 |  |
| 2026-04-05T11:35:53+00:00 | ok | keisei/training/models/__init__.py | docs/bugs/generated/training/models/__init__.py.md |  | 31.82 |  |
| 2026-04-05T11:36:25+00:00 | ok | keisei/training/models/base.py | docs/bugs/generated/training/models/base.py.md |  | 63.26 |  |
| 2026-04-05T11:36:45+00:00 | ok | keisei/training/match_scheduler.py | docs/bugs/generated/training/match_scheduler.py.md |  | 84.14 |  |
| 2026-04-05T11:36:46+00:00 | ok | keisei/training/historical_gauntlet.py | docs/bugs/generated/training/historical_gauntlet.py.md |  | 84.32 |  |
| 2026-04-05T11:36:46+00:00 | ok | keisei/training/models/katago_base.py | docs/bugs/generated/training/models/katago_base.py.md |  | 84.52 |  |
| 2026-04-05T11:37:02+00:00 | ok | keisei/training/katago_ppo.py | docs/bugs/generated/training/katago_ppo.py.md |  | 100.90 |  |
| 2026-04-05T11:37:06+00:00 | ok | keisei/training/historical_library.py | docs/bugs/generated/training/historical_library.py.md |  | 104.72 |  |
| 2026-04-05T11:37:24+00:00 | ok | keisei/training/match_utils.py | docs/bugs/generated/training/match_utils.py.md |  | 122.94 |  |
| 2026-04-05T11:37:47+00:00 | ok | keisei/training/katago_loop.py | docs/bugs/generated/training/katago_loop.py.md |  | 145.55 |  |
| 2026-04-05T11:38:02+00:00 | ok | keisei/training/model_registry.py | docs/bugs/generated/training/model_registry.py.md |  | 161.21 |  |
| 2026-04-05T11:39:14+00:00 | ok | keisei/training/priority_scorer.py | docs/bugs/generated/training/priority_scorer.py.md |  | 71.81 |  |
| 2026-04-05T11:39:25+00:00 | ok | keisei/training/models/resnet.py | docs/bugs/generated/training/models/resnet.py.md |  | 82.22 |  |
| 2026-04-05T11:39:34+00:00 | ok | keisei/training/models/se_resnet.py | docs/bugs/generated/training/models/se_resnet.py.md |  | 91.98 |  |
| 2026-04-05T11:39:36+00:00 | ok | keisei/training/models/mlp.py | docs/bugs/generated/training/models/mlp.py.md |  | 93.73 |  |
| 2026-04-05T11:39:49+00:00 | ok | keisei/training/role_elo.py | docs/bugs/generated/training/role_elo.py.md |  | 106.29 |  |
| 2026-04-05T11:39:49+00:00 | ok | keisei/training/models/transformer.py | docs/bugs/generated/training/models/transformer.py.md |  | 106.54 |  |
| 2026-04-05T11:39:58+00:00 | ok | keisei/training/tiered_pool.py | docs/bugs/generated/training/tiered_pool.py.md |  | 115.05 |  |
| 2026-04-05T11:40:02+00:00 | ok | keisei/training/tournament.py | docs/bugs/generated/training/tournament.py.md |  | 119.65 |  |
| 2026-04-05T11:40:56+00:00 | ok | keisei/training/tier_managers.py | docs/bugs/generated/training/tier_managers.py.md |  | 173.49 |  |
| 2026-04-05T11:41:23+00:00 | ok | keisei/training/opponent_store.py | docs/bugs/generated/training/opponent_store.py.md |  | 200.68 |  |
| 2026-04-05T11:42:57+00:00 | ok | keisei/training/value_adapter.py | docs/bugs/generated/training/value_adapter.py.md |  | 94.08 |  |
| 2026-04-05T11:43:20+00:00 | ok | keisei/training/transition.py | docs/bugs/generated/training/transition.py.md |  | 116.94 |  |
| 2026-04-07T23:32:33+00:00 | ok | keisei/db.py | docs/bugs/generated/db.py.md |  | 27.12 | evidence_gate=1 |
| 2026-04-07T23:32:38+00:00 | ok | keisei/__init__.py | docs/bugs/generated/__init__.py.md |  | 32.59 | evidence_gate=1 |
| 2026-04-07T23:32:50+00:00 | ok | keisei/sl/prepare.py | docs/bugs/generated/sl/prepare.py.md |  | 44.07 | evidence_gate=1 |
| 2026-04-07T23:33:23+00:00 | ok | keisei/server/__init__.py | docs/bugs/generated/server/__init__.py.md |  | 77.95 |  |
| 2026-04-07T23:34:06+00:00 | ok | keisei/sl/__init__.py | docs/bugs/generated/sl/__init__.py.md |  | 120.64 | evidence_gate=1 |
| 2026-04-07T23:34:14+00:00 | ok | keisei/config.py | docs/bugs/generated/config.py.md |  | 128.96 | evidence_gate=2 |
| 2026-04-07T23:34:23+00:00 | ok | keisei/sl/dataset.py | docs/bugs/generated/sl/dataset.py.md |  | 137.17 | evidence_gate=1 |
| 2026-04-07T23:34:44+00:00 | ok | keisei/server/app.py | docs/bugs/generated/server/app.py.md |  | 158.45 | evidence_gate=1 |
| 2026-04-07T23:34:46+00:00 | ok | keisei/sl/trainer.py | docs/bugs/generated/sl/trainer.py.md |  | 160.98 | evidence_gate=1 |
| 2026-04-07T23:35:05+00:00 | ok | keisei/sl/parsers.py | docs/bugs/generated/sl/parsers.py.md |  | 179.57 |  |
| 2026-04-07T23:35:32+00:00 | ok | keisei/training/algorithm_registry.py | docs/bugs/generated/training/algorithm_registry.py.md |  | 26.57 | evidence_gate=1 |
| 2026-04-07T23:35:37+00:00 | ok | keisei/training/frontier_promoter.py | docs/bugs/generated/training/frontier_promoter.py.md |  | 32.40 | evidence_gate=1 |
| 2026-04-07T23:36:17+00:00 | ok | keisei/training/__init__.py | docs/bugs/generated/training/__init__.py.md |  | 72.28 |  |
| 2026-04-07T23:36:26+00:00 | ok | keisei/training/distributed.py | docs/bugs/generated/training/distributed.py.md |  | 81.10 | evidence_gate=1 |
| 2026-04-07T23:36:45+00:00 | ok | keisei/training/gae.py | docs/bugs/generated/training/gae.py.md |  | 99.62 | evidence_gate=1 |
| 2026-04-07T23:36:57+00:00 | ok | keisei/training/checkpoint.py | docs/bugs/generated/training/checkpoint.py.md |  | 111.86 | evidence_gate=1 |
| 2026-04-07T23:38:07+00:00 | ok | keisei/training/demonstrator.py | docs/bugs/generated/training/demonstrator.py.md |  | 181.62 |  |
| 2026-04-07T23:38:08+00:00 | ok | keisei/training/concurrent_matches.py | docs/bugs/generated/training/concurrent_matches.py.md |  | 183.17 |  |
| 2026-04-07T23:38:20+00:00 | ok | keisei/training/evaluate.py | docs/bugs/generated/training/evaluate.py.md |  | 195.20 |  |
| 2026-04-07T23:38:55+00:00 | ok | keisei/training/dynamic_trainer.py | docs/bugs/generated/training/dynamic_trainer.py.md |  | 230.09 | evidence_gate=1 |
| 2026-04-07T23:40:10+00:00 | ok | keisei/training/historical_gauntlet.py | docs/bugs/generated/training/historical_gauntlet.py.md |  | 74.87 | evidence_gate=1 |
| 2026-04-07T23:40:22+00:00 | ok | keisei/training/models/__init__.py | docs/bugs/generated/training/models/__init__.py.md |  | 86.47 |  |
| 2026-04-07T23:40:24+00:00 | ok | keisei/training/models/base.py | docs/bugs/generated/training/models/base.py.md |  | 88.41 |  |
| 2026-04-07T23:41:13+00:00 | ok | keisei/training/model_registry.py | docs/bugs/generated/training/model_registry.py.md |  | 137.37 |  |
| 2026-04-07T23:41:18+00:00 | ok | keisei/training/match_utils.py | docs/bugs/generated/training/match_utils.py.md |  | 143.11 |  |
| 2026-04-07T23:41:28+00:00 | ok | keisei/training/match_scheduler.py | docs/bugs/generated/training/match_scheduler.py.md |  | 152.93 |  |
| 2026-04-07T23:41:29+00:00 | ok | keisei/training/katago_loop.py | docs/bugs/generated/training/katago_loop.py.md |  | 154.30 | evidence_gate=1 |
| 2026-04-07T23:41:32+00:00 | ok | keisei/training/historical_library.py | docs/bugs/generated/training/historical_library.py.md |  | 156.91 | evidence_gate=1 |
| 2026-04-07T23:41:51+00:00 | ok | keisei/training/game_feature_tracker.py | docs/bugs/generated/training/game_feature_tracker.py.md |  | 175.60 |  |
| 2026-04-07T23:41:52+00:00 | ok | keisei/training/katago_ppo.py | docs/bugs/generated/training/katago_ppo.py.md |  | 176.93 | evidence_gate=1 |
| 2026-04-07T23:42:19+00:00 | ok | keisei/training/style_profiler.py | docs/bugs/generated/training/style_profiler.py.md |  | 26.97 | evidence_gate=1 |
| 2026-04-07T23:43:43+00:00 | ok | keisei/training/models/mlp.py | docs/bugs/generated/training/models/mlp.py.md |  | 110.97 | evidence_gate=1 |
| 2026-04-07T23:43:43+00:00 | ok | keisei/training/priority_scorer.py | docs/bugs/generated/training/priority_scorer.py.md |  | 111.33 |  |
| 2026-04-07T23:44:10+00:00 | ok | keisei/training/models/katago_base.py | docs/bugs/generated/training/models/katago_base.py.md |  | 138.18 |  |
| 2026-04-07T23:44:18+00:00 | ok | keisei/training/tier_managers.py | docs/bugs/generated/training/tier_managers.py.md |  | 146.35 | evidence_gate=1 |
| 2026-04-07T23:44:31+00:00 | ok | keisei/training/role_elo.py | docs/bugs/generated/training/role_elo.py.md |  | 158.56 | evidence_gate=1 |
| 2026-04-07T23:44:40+00:00 | ok | keisei/training/opponent_store.py | docs/bugs/generated/training/opponent_store.py.md |  | 168.29 | evidence_gate=1 |
| 2026-04-07T23:44:42+00:00 | ok | keisei/training/models/se_resnet.py | docs/bugs/generated/training/models/se_resnet.py.md |  | 170.33 | evidence_gate=1 |
| 2026-04-07T23:44:50+00:00 | ok | keisei/training/models/resnet.py | docs/bugs/generated/training/models/resnet.py.md |  | 178.01 |  |
| 2026-04-07T23:45:41+00:00 | ok | keisei/training/models/transformer.py | docs/bugs/generated/training/models/transformer.py.md |  | 229.26 |  |
| 2026-04-07T23:46:14+00:00 | ok | keisei/training/tournament.py | docs/bugs/generated/training/tournament.py.md |  | 32.87 | evidence_gate=1 |
| 2026-04-07T23:48:33+00:00 | ok | keisei/training/value_adapter.py | docs/bugs/generated/training/value_adapter.py.md |  | 171.46 | evidence_gate=1 |
| 2026-04-07T23:48:51+00:00 | ok | keisei/training/tiered_pool.py | docs/bugs/generated/training/tiered_pool.py.md |  | 189.76 |  |
| 2026-04-07T23:49:31+00:00 | ok | keisei/training/transition.py | docs/bugs/generated/training/transition.py.md |  | 229.31 |  |
| 2026-05-02T16:33:18+00:00 | ok | keisei/showcase/__init__.py | docs/bugs/generated/showcase/__init__.py.md |  | 89.66 |  |
| 2026-05-02T16:33:35+00:00 | ok | keisei/server/__init__.py | docs/bugs/generated/server/__init__.py.md |  | 106.00 |  |
| 2026-05-02T16:33:35+00:00 | ok | keisei/__init__.py | docs/bugs/generated/__init__.py.md |  | 106.16 |  |
| 2026-05-02T16:33:54+00:00 | ok | keisei/showcase/__main__.py | docs/bugs/generated/showcase/__main__.py.md |  | 125.70 |  |
| 2026-05-02T16:35:14+00:00 | ok | keisei/showcase/runner.py | docs/bugs/generated/showcase/runner.py.md |  | 204.97 |  |
| 2026-05-02T16:35:56+00:00 | ok | keisei/db.py | docs/bugs/generated/db.py.md |  | 247.45 |  |
| 2026-05-02T16:36:00+00:00 | ok | keisei/showcase/inference.py | docs/bugs/generated/showcase/inference.py.md |  | 251.51 |  |
| 2026-05-02T16:36:06+00:00 | ok | keisei/showcase/db_ops.py | docs/bugs/generated/showcase/db_ops.py.md |  | 257.79 |  |
| 2026-05-02T16:36:17+00:00 | ok | keisei/server/app.py | docs/bugs/generated/server/app.py.md |  | 268.32 |  |
| 2026-05-02T16:36:51+00:00 | ok | keisei/config.py | docs/bugs/generated/config.py.md |  | 302.84 |  |
| 2026-05-02T16:38:37+00:00 | ok | keisei/training/__init__.py | docs/bugs/generated/training/__init__.py.md |  | 105.47 |  |
| 2026-05-02T16:39:08+00:00 | ok | keisei/sl/__init__.py | docs/bugs/generated/sl/__init__.py.md |  | 136.73 |  |
| 2026-05-02T16:39:09+00:00 | ok | keisei/training/algorithm_registry.py | docs/bugs/generated/training/algorithm_registry.py.md |  | 137.59 |  |
| 2026-05-02T16:39:50+00:00 | ok | keisei/sl/prepare.py | docs/bugs/generated/sl/prepare.py.md |  | 179.03 |  |
| 2026-05-02T16:40:08+00:00 | ok | keisei/sl/dataset.py | docs/bugs/generated/sl/dataset.py.md |  | 196.94 |  |
| 2026-05-02T16:40:33+00:00 | ok | keisei/training/checkpoint.py | docs/bugs/generated/training/checkpoint.py.md |  | 221.43 |  |
| 2026-05-02T16:40:45+00:00 | ok | keisei/sl/trainer.py | docs/bugs/generated/sl/trainer.py.md |  | 233.48 |  |
| 2026-05-02T16:41:21+00:00 | ok | keisei/training/demonstrator.py | docs/bugs/generated/training/demonstrator.py.md |  | 269.16 |  |
| 2026-05-02T16:42:25+00:00 | ok | keisei/training/concurrent_matches.py | docs/bugs/generated/training/concurrent_matches.py.md |  | 333.42 |  |
| 2026-05-02T16:42:53+00:00 | ok | keisei/sl/parsers.py | docs/bugs/generated/sl/parsers.py.md |  | 361.27 |  |
| 2026-05-02T16:45:20+00:00 | ok | keisei/training/evaluate.py | docs/bugs/generated/training/evaluate.py.md |  | 146.99 |  |
| 2026-05-02T16:45:34+00:00 | ok | keisei/training/historical_gauntlet.py | docs/bugs/generated/training/historical_gauntlet.py.md |  | 161.40 |  |
| 2026-05-02T16:46:20+00:00 | ok | keisei/training/game_feature_tracker.py | docs/bugs/generated/training/game_feature_tracker.py.md |  | 207.47 |  |
| 2026-05-02T16:46:37+00:00 | ok | keisei/training/frontier_promoter.py | docs/bugs/generated/training/frontier_promoter.py.md |  | 224.24 |  |
| 2026-05-02T16:46:40+00:00 | ok | keisei/training/distributed.py | docs/bugs/generated/training/distributed.py.md |  | 227.69 |  |
| 2026-05-02T16:46:53+00:00 | ok | keisei/training/gae.py | docs/bugs/generated/training/gae.py.md |  | 240.41 |  |
| 2026-05-02T16:47:18+00:00 | ok | keisei/training/historical_library.py | docs/bugs/generated/training/historical_library.py.md |  | 264.94 |  |
| 2026-05-02T16:47:50+00:00 | ok | keisei/training/dynamic_trainer.py | docs/bugs/generated/training/dynamic_trainer.py.md |  | 297.65 |  |
| 2026-05-02T16:48:09+00:00 | ok | keisei/training/katago_loop.py | docs/bugs/generated/training/katago_loop.py.md |  | 316.44 |  |
| 2026-05-02T16:48:14+00:00 | ok | keisei/training/katago_ppo.py | docs/bugs/generated/training/katago_ppo.py.md |  | 320.97 |  |
| 2026-05-02T16:50:06+00:00 | ok | keisei/training/models/__init__.py | docs/bugs/generated/training/models/__init__.py.md |  | 112.71 |  |
| 2026-05-02T16:50:24+00:00 | ok | keisei/training/models/base.py | docs/bugs/generated/training/models/base.py.md |  | 130.78 |  |
| 2026-05-02T16:51:11+00:00 | ok | keisei/training/models/katago_base.py | docs/bugs/generated/training/models/katago_base.py.md |  | 177.31 |  |
| 2026-05-02T16:51:24+00:00 | ok | keisei/training/models/se_resnet.py | docs/bugs/generated/training/models/se_resnet.py.md |  | 190.80 |  |
| 2026-05-02T16:51:55+00:00 | ok | keisei/training/model_registry.py | docs/bugs/generated/training/model_registry.py.md |  | 221.75 |  |
| 2026-05-02T16:51:57+00:00 | ok | keisei/training/match_scheduler.py | docs/bugs/generated/training/match_scheduler.py.md |  | 223.23 |  |
| 2026-05-02T16:52:18+00:00 | ok | keisei/training/models/resnet.py | docs/bugs/generated/training/models/resnet.py.md |  | 244.56 |  |
| 2026-05-02T16:52:19+00:00 | ok | keisei/training/models/mlp.py | docs/bugs/generated/training/models/mlp.py.md |  | 245.33 |  |
| 2026-05-02T16:52:28+00:00 | ok | keisei/training/models/transformer.py | docs/bugs/generated/training/models/transformer.py.md |  | 254.17 |  |
| 2026-05-02T16:53:06+00:00 | ok | keisei/training/match_utils.py | docs/bugs/generated/training/match_utils.py.md |  | 292.26 |  |
| 2026-05-02T16:56:15+00:00 | ok | keisei/training/priority_scorer.py | docs/bugs/generated/training/priority_scorer.py.md |  | 189.09 |  |
| 2026-05-02T16:56:30+00:00 | ok | keisei/training/tournament_dispatcher.py | docs/bugs/generated/training/tournament_dispatcher.py.md |  | 203.91 |  |
| 2026-05-02T16:56:48+00:00 | ok | keisei/training/role_elo.py | docs/bugs/generated/training/role_elo.py.md |  | 222.02 |  |
| 2026-05-02T16:56:50+00:00 | ok | keisei/training/style_profiler.py | docs/bugs/generated/training/style_profiler.py.md |  | 224.22 |  |
