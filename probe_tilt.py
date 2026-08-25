"""无头探针: 夹爪姿态保持/恢复诊断
每个场景结束后测量: 传感器垫相对 home 的倾角, 关节误差, 以及机械臂上所有残留接触力
找出 "动了一下就回不到水平" 的根源
"""
import numpy as np
import mujoco
from kinova_sensor_sim import SimSensor

s = SimSensor()
dt = s.model.opt.timestep
home = s.home_qpos.copy()
np.set_printoptions(precision=3, suppress=True)

ARM_BODY_IDS = set()
for b in range(s.model.nbody):
    name = mujoco.mj_id2name(s.model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    if any(k in name for k in ("link", "base", "grasping", "tool", "sensor", "finger", "2f85")):
        ARM_BODY_IDS.add(b)
ARM_BODY_IDS.discard(0)

GRASP_QADR = s.model.jnt_qposadr[s.grasp_jnt]


def tilt_deg():
    R = s.data.geom_xmat[s.sensor_geom_id].reshape(3, 3)
    Rel = R @ s.arm_R0.T
    w = np.empty(4)
    mujoco.mju_mat2Quat(w, Rel.reshape(9))
    return float(np.degrees(2 * np.arcsin(min(1.0, np.linalg.norm(w[1:])))))


def arm_contacts():
    """机械臂本体上的所有接触: (geom对, 法向力N)"""
    out = []
    for i in range(s.data.ncon):
        c = s.data.contact[i]
        b1 = s.model.geom_bodyid[c.geom1]
        b2 = s.model.geom_bodyid[c.geom2]
        if b1 in ARM_BODY_IDS or b2 in ARM_BODY_IDS:
            f = np.zeros(6)
            mujoco.mj_contactForce(s.model, s.data, i, f)
            g1 = mujoco.mj_id2name(s.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            g2 = mujoco.mj_id2name(s.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            out.append((f"{g1}|{g2}", float(np.linalg.norm(f[:3]))))
    return out


def step(n):
    for _ in range(n):
        s.update_interaction()
        mujoco.mj_step(s.model, s.data)
        f = np.array(s.get_contact_forces())
        s.raw_fx, s.raw_fy, s.raw_fz = f
        s.fz_filt += 0.2 * (f[2] - s.fz_filt)
        out = s.signal.process(f, dt)
        s.fx, s.fy, s.fz = out


def report(tag):
    je = np.abs(s.data.qpos[:7] - home[:7])
    fje = np.abs(s.data.qpos[7:10] - home[7:10])
    cons = arm_contacts()
    big = [c for c in cons if c[1] > 0.5]
    box = s.data.qpos[GRASP_QADR:GRASP_QADR + 3]
    print(f"[{tag}] tilt={tilt_deg():6.2f}deg  maxArmErr={je.max():.4f}rad  "
          f"fingerErr={fje.max():.4f}rad  rawFz={s.raw_fz:+6.2f}N  graspBox=({box[0]:.2f},{box[1]:.2f},{box[2]:.2f})")
    if big:
        for g, f in big:
            print(f"         !! 残留接触: {g}  F={f:.2f}N")
    return tilt_deg()


print("===== 基线 =====")
s.mode = "touch"
s.reset_scene()
step(2000)
report("home静止2s")

print("\n===== A. touch 6N 按压 -> 释放 =====")
s.reset_scene()
with s.lock:
    s.touch_force, s.touch_sx, s.touch_sy = 6.0, 0.0, 0.0
step(6000)
report("A1 加载中")
with s.lock:
    s.touch_force = 0.0
step(6000)
report("A2 释放后")

print("\n===== B. touch 6N + 大剪切拖拽 -> 释放 =====")
s.reset_scene()
with s.lock:
    s.touch_force, s.touch_sx, s.touch_sy = 6.0, 0.7, -0.7
step(6000)
report("B1 拖拽中")
with s.lock:
    s.touch_force, s.touch_sx, s.touch_sy = 0.0, 0.0, 0.0
step(6000)
report("B2 释放后")

print("\n===== C. grasp 70% 夹取 -> 松开 =====")
s.mode = "grasp"
s.reset_scene()
s.grasp_pct = 70.0
step(6000)
report("C1 夹紧中")
s.grasp_pct = 0.0
step(6000)
report("C2 松开后")

print("\n===== D. arm 5N 臂压 -> 撤力 =====")
s.mode = "arm"
s.reset_scene()
s.arm_force = 5.0
step(8000)
report("D1 臂压中")
s.arm_force = 0.0
step(8000)
report("D2 撤力后")

print("\n===== E. 腕部外力扰动(模拟鼠标拖拽) -> 松手 =====")
s.mode = "touch"
s.reset_scene()
wrist = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, "spherical_wrist2_link")
if wrist < 0:
    wrist = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, "tactile_sensor")
print(f"    扰动体: {mujoco.mj_id2name(s.model, mujoco.mjtObj.mjOBJ_BODY, wrist)}")
for _ in range(1000):
    s.data.xfrc_applied[wrist, 3:6] = [0.0, 2.0, 1.0]
    s.update_interaction()
    mujoco.mj_step(s.model, s.data)
s.data.xfrc_applied[wrist, :] = 0.0
report("E1 扰动结束瞬间")
step(6000)
report("E2 松手6s后")

print("\n===== F. 连续模式切换 touch->arm->grasp->touch =====")
s.mode = "touch"
s.reset_scene()
with s.lock:
    s.touch_force = 6.0
step(4000)
report("F1 touch 6N")
s.mode = "arm"
s.reset_scene()
s.arm_force = 5.0
step(6000)
report("F2 arm 5N")
s.mode = "grasp"
s.reset_scene()
s.grasp_pct = 70.0
step(5000)
report("F3 grasp 70%")
s.mode = "touch"
s.reset_scene()
step(4000)
report("F4 回touch静止")
with s.lock:
    s.touch_force = 6.0
step(5000)
report("F5 再按压")

print("\n===== G. touch 满量程 50N -> 释放 =====")
s.mode = "touch"
s.reset_scene()
with s.lock:
    s.touch_force = 50.0
step(6000)
report("G1 50N加载中")
with s.lock:
    s.touch_force = 0.0
step(6000)
report("G2 释放后")

print("\n完成")
