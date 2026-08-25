"""无头复现4: 视窗暴力拖拽(扰动 力可达数千N)
mujoco.viewer 拖拽力随鼠标位移线性增大且无上限, 用户快速大距离拖拽
可施加 >> 伺服上限(±52/105N·m)的力, 把相邻连杆凸包压入互相穿透,
释放后接触约束(刚性, 数千N)锁死关节, 伺服永远拉不回 home。
扫力幅+方向, 找出锁定阈值和锁定接触对。
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


def lock_contacts():
    out = []
    for i in range(s.data.ncon):
        c = s.data.contact[i]
        b1 = s.model.geom_bodyid[c.geom1]
        b2 = s.model.geom_bodyid[c.geom2]
        if b1 == 0 or b2 == 0 or b1 == s.touch_body or b2 == s.touch_body:
            continue
        g1 = mujoco.mj_id2name(s.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or "?"
        g2 = mujoco.mj_id2name(s.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or "?"
        f = np.zeros(6)
        mujoco.mj_contactForce(s.model, s.data, i, f)
        if abs(f[0]) > 5:
            out.append((f"{g1}|{g2}", abs(f[0])))
    return sorted(out, key=lambda x: -x[1])[:4]


DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
for mag in (500.0, 1000.0, 2000.0):
    for d in DIRS:
        s.mode = "touch"
        s.reset_scene()
        s.data.xfrc_applied[TOOL, :3] = np.array(d) * mag
        step(1500)
        s.data.xfrc_applied[:] = 0
        step(4000)
        err = s.data.qpos[:7] - s.home_qpos[:7]
        worst = int(np.argmax(np.abs(err)))
        stuck = np.abs(err).max() > 0.05
        tag = " <== 卡死" if stuck else ""
        print(f"{mag:5.0f}N {d}: tilt={s.tilt_deg():6.1f}deg "
              f"j{worst+1}err={err[worst]:+6.2f}rad{tag}")
        if stuck:
            for pair, fn in lock_contacts():
                print(f"      锁定: {pair}  {fn:.0f}N")
