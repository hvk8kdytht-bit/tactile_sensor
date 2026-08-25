"""验证: 姿态自动恢复(最后自愈手段)
暴力甩臂(tilt>>20°)后, 2s内无指令应自动复位, 倾角归零。
"""
import numpy as np
import mujoco
from kinova_sensor_sim import SimSensor

s = SimSensor()
dt = s.model.opt.timestep
np.set_printoptions(precision=3, suppress=True)

TOOL = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, "bracelet_link")


def step(n):
    for _ in range(n):
        s.update_interaction()
        mujoco.mj_step(s.model, s.data)
        f = np.array(s.get_contact_forces())
        s.raw_fx, s.raw_fy, s.raw_fz = f
        s.fz_filt += 0.2 * (f[2] - s.fz_filt)
        s.tilt_now = s.tilt_deg()


for mag, d in [(2000.0, (0, 1, 0)), (1500.0, (1, 0, 1)), (3000.0, (0, 0, -1))]:
    s.mode = "touch"
    s.reset_scene()
    s.data.xfrc_applied[TOOL, :3] = np.array(d, dtype=float) / np.linalg.norm(d) * mag
    step(1500)
    s.data.xfrc_applied[:] = 0
    print(f"释放瞬间: tilt={s.tilt_deg():.1f}deg")
    # 释放后逐步观察: 自动恢复应在 ~2s(1000步)后触发
    for phase in range(4):
        step(1000)
        print(f"  +{(phase+1)}s: tilt={s.tilt_deg():6.1f}deg "
              f"jerr_max={np.abs(s.data.qpos[:7] - s.home_qpos[:7]).max():.3f}rad")
