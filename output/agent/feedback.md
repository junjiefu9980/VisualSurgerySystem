# Agent Feedback

Generated: 2026-04-10T23:07:04

## Overall

- Total cases: `17`
- Policy counts: `tri=9`, `monster=3`, `2d=3`, `None=2`
- Gate buckets: `tri pass=9`, `monster pass=3`, `fallback 2d=3`, `bad cases=2`
- Mean metrics: `tri disp_p95=34.696`, `monster disp_p95=58.419`, `chosen kf gain=52.982`
- Feedback: `tri-dominant, monster-fallback-active, bad-cases-remain`
- Note: TRI stays the main route (9 cases). MONSTER is used as fallback in 3 cases, 2D fallback appears in 3 cases, and bad cases=2.

## Selected Cases

## Case 33 | left

- Final mode: `None`
- Chosen method: `2d`
- Scenario: `c1_2d_fail_none`
- Anchor kpt: `L1`
- Route reason: C1 2d fail: reason_2d=f1<0.85, 2d_f1=0.625719, policy_mode=None
- 2D: frames=1001, valid=1.000, disp_p95=181.081, red_s=0.17, events=1
- Trajectory figure: `output/agent/figures/33_left_2d_traj.png`
- Timeline figure: `output/agent/figures/33_left_2d_timeline.png`
- Feedback: `bad-case, mild-red, visual-fail`
- Note: Visual tracking is weak on this side.

## Case 33 | right

- Final mode: `None`
- Chosen method: `2d`
- Scenario: `c1_2d_fail_none`
- Anchor kpt: `R1`
- Route reason: C1 2d fail: reason_2d=f1<0.85, 2d_f1=0.625719, policy_mode=None
- 2D: frames=1001, valid=1.000, disp_p95=345.273, red_s=0.23, events=1
- Trajectory figure: `output/agent/figures/33_right_2d_traj.png`
- Timeline figure: `output/agent/figures/33_right_2d_timeline.png`
- Feedback: `bad-case, mild-red, visual-fail`
- Note: Visual tracking is weak on this side.

## Case 31 | left

- Final mode: `2d`
- Chosen method: `2d`
- Scenario: `c2_fallback_2d`
- Anchor kpt: `L2`
- Route reason: C2 fallback 2d: tri fail (valid_all<0.80), monster fail (disp_p95>90.0), 2d_f1=0.885807
- 2D: frames=1001, valid=1.000, disp_p95=249.580, red_s=0.00, events=0
- Trajectory figure: `output/agent/figures/31_left_2d_traj.png`
- Timeline figure: `output/agent/figures/31_left_2d_timeline.png`
- Feedback: `fallback-2d, clean`
- Note: 2D fallback is selected for this side.

## Case 31 | right

- Final mode: `2d`
- Chosen method: `2d`
- Scenario: `c2_fallback_2d`
- Anchor kpt: `R1`
- Route reason: C2 fallback 2d: tri fail (valid_all<0.80), monster fail (disp_p95>90.0), 2d_f1=0.885807
- 2D: frames=1001, valid=1.000, disp_p95=12.241, red_s=0.00, events=0
- Trajectory figure: `output/agent/figures/31_right_2d_traj.png`
- Timeline figure: `output/agent/figures/31_right_2d_timeline.png`
- Feedback: `fallback-2d, clean`
- Note: 2D fallback is selected for this side.

## Case 24 | left

- Final mode: `tri`
- Chosen method: `tri`
- Scenario: `c3_tri_pass`
- Anchor kpt: `L1`
- Route reason: C3 tri pass: tri_valid_all=0.978771, tri_disp_p95=26.893363, 2d_f1=0.984596
- TRI: frames=1001, valid=1.000, disp_p95=18.575, red_s=0.00, events=0
- Trajectory figure: `output/agent/figures/24_left_tri_traj.png`
- Timeline figure: `output/agent/figures/24_left_tri_timeline.png`
- Feedback: `tri-ready, clean`
- Note: TRI looks usable on this side.

## Case 24 | right

- Final mode: `tri`
- Chosen method: `tri`
- Scenario: `c3_tri_pass`
- Anchor kpt: `R1`
- Route reason: C3 tri pass: tri_valid_all=0.978771, tri_disp_p95=26.893363, 2d_f1=0.984596
- TRI: frames=1001, valid=1.000, disp_p95=20.654, red_s=0.00, events=0
- Trajectory figure: `output/agent/figures/24_right_tri_traj.png`
- Timeline figure: `output/agent/figures/24_right_tri_timeline.png`
- Feedback: `tri-ready, clean`
- Note: TRI looks usable on this side.

## Case 20 | left

- Final mode: `monster`
- Chosen method: `monster`
- Scenario: `c4_monster_rescue`
- Anchor kpt: `L1`
- Route reason: C4 monster rescue: tri fail (valid_all<0.80); tri_disp_p95=35.805868 -> monster_disp_p95=39.528091
- MONSTER: frames=1001, valid=1.000, disp_p95=22.480, red_s=0.00, events=0
- Trajectory figure: `output/agent/figures/20_left_monster_traj.png`
- Timeline figure: `output/agent/figures/20_left_monster_timeline.png`
- Feedback: `monster-needed, clean`
- Note: MONSTER is the safer 3D choice on this side.

## Case 20 | right

- Final mode: `monster`
- Chosen method: `monster`
- Scenario: `c4_monster_rescue`
- Anchor kpt: `R2`
- Route reason: C4 monster rescue: tri fail (valid_all<0.80); tri_disp_p95=35.805868 -> monster_disp_p95=39.528091
- MONSTER: frames=1001, valid=1.000, disp_p95=2.711, red_s=0.00, events=0
- Trajectory figure: `output/agent/figures/20_right_monster_traj.png`
- Timeline figure: `output/agent/figures/20_right_monster_timeline.png`
- Feedback: `monster-needed, clean`
- Note: MONSTER is the safer 3D choice on this side.
