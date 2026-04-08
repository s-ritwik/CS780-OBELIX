# Difficulty 3 Wall Search Run Tracker

Target: wall mean reward >= -300 on `evaluate.py --difficulty 3 --wall_obstacles --max_steps 2000`.

## Active Runs

| Run | Checkpoint | Method | Status |
| --- | --- | --- | --- |
| s701 | `wall_d3_v3base_cont_s701.pth` | Continue deployed wall actor, balanced shaping, `768` envs | stopped; trainer eval flat at `-1136.7` |
| s702 | `wall_d3_v3base_cont_s702_pushheavy.pth` | Continue deployed wall actor, push-heavy shaping, `512` envs | stopped after macro target hit |
| s703 | `wall_d3_v3base_cont_s703_explore.pth` | Continue deployed wall actor, higher entropy/exploration, `384` envs | stopped after macro target hit |
| s704 | `wall_d3_rich_warm_s704.pth` | Rich obs features from scratch, warm-started from final switch agent, `256` envs | stopped after macro target hit |
| s705 | `wall_d3_recurrent_gru_s705.pth` | GRU recurrent asym PPO, privileged-teacher warm start/loss, no teacher action takeover, `128` envs | stopped after macro target hit |
| s706 | `wall_d3_recurrent_lstm_s706.pth` | LSTM recurrent asym PPO, privileged-teacher warm start/loss, no teacher action takeover, `128` envs | stopped after macro target hit |

## Completed Runs

| Candidate | Artifact | Wall mean | No-wall mean | Weighted | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| macro_51 | `submission_d3_wall_macro_51/submission_d3_wall_macro_51.zip` | 51.6 | 1252.9 | 532.1 | Inlined macro probes from legal obs/time; one agent file and one weight file |
