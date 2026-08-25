"""无头探针2: 复现touch模式joint_7冻结 + 验证arm模式standby几何
A. 按线上日志重建用户时变剪切力交互曲线 -> 观察joint_7
B. arm模式: force=0时 pad面-目标面 的带符号距离(standby应是间隙不是压入)
C. arm模式: 力标定曲线(新目标位置的实际N/mm)
"""
import numpy as np
import mujoco
from kinova_sensor_sim import SimSensor

s = SimSensor()
dt = s.model.opt.timestep
home = s.home_qpos.copy()
np.set_printoptions(precision=3, suppress=True)


def step(n):
    for _ in range(n):
        s.update_interaction()
        mujoco.mj_step(s.model, s.data)
        f = np.array(s.get_contact_forces())
        s.raw_fx, s.raw_fy, s.raw_fz = f
        s.fz_filt += 0.2 * (f[2] - s.fz_filt)
        out = s.signal.process(f, dt)
        s.fx, s.fy, s.fz = out


def q7():
    return float(s.data.qpos[6])


def tilt():
    R = s.data.geom_xmat[s.sensor_geom_id].reshape(3, 3)
    Rel = R @ s.arm_R0.T
    w = np.empty(4)
    mujoco.mju_mat2Quat(w, Rel.reshape(9))
    return float(np.degrees(2 * np.arcsin(min(1.0, np.linalg.norm(w[1:])))))


def pad_face_gap_to_target():
    """pad感测面到目标近表面的带符号距离: >0=间隙, <0=压入深度"""
    R, c = s._sensor_frame()
    face = c + R @ np.array([0, -0.0026, 0])
    n = R @ np.array([0, -1, 0])          # pad面外法线(指向目标)
    tc = s.data.geom_xpos[s.press_geom_id].copy()
    # 目标沿n方向的半厚度
    half = 0.012
    near_face = tc - n * half              # 面向pad的一侧
    return float(n @ (near_face - face))   # face沿n到near_face的距离


print("===== A. 时变剪切(重建用户操作) =====")
s.mode = "touch"
s.reset_scene()
# 阶段1: 法向力0->6N (0.5s), 剪切sx 0->0.7 (1s)
for i in range(500):
    with s.lock:
        s.touch_force = min(6.0, i / 250 * 6.0)
    step(1)
for i in range(500):
    with s.lock:
        s.touch_sx = min(0.7, i / 250 * 0.7)
    step(1)
print(f"  A1 建立剪切: q7={q7():+.4f} tilt={tilt():.2f}deg rawFz={s.raw_fz:+.2f} rawFx={s.raw_fx:+.2f}")
# 阶段2: 剪切保持2s后释放(sx->0), 法向保持
with s.lock:
    s.touch_sx = 0.7
step(1000)
print(f"  A2 剪切保持: q7={q7():+.4f} tilt={tilt():.2f}deg rawFz={s.raw_fz:+.2f} rawFx={s.raw_fx:+.2f}")
for i in range(1500):
    with s.lock:
        s.touch_sx = max(0.0, 0.7 - i / 500 * 0.7)
    step(1)
print(f"  A3 剪切释放: q7={q7():+.4f} tilt={tilt():.2f}deg rawFz={s.raw_fz:+.2f}")
# 阶段3: 法向释放
for i in range(500):
    with s.lock:
        s.touch_force = max(0.0, 6.0 - i / 250 * 6.0)
    step(1)
step(5000)
print(f"  A4 全释放后10s: q7={q7():+.4f} tilt={tilt():.2f}deg rawFz={s.raw_fz:+.2f}")

print("\n===== A2. 剪切快速摆动(用户拖拽甩动) =====")
s.reset_scene()
with s.lock:
    s.touch_force = 6.0
step(2000)
for i in range(3000):
    with s.lock:
        s.touch_sx = 0.9 * np.sin(i / 150.0 * 2 * np.pi * 3)
        s.touch_sy = 0.9 * np.cos(i / 200.0 * 2 * np.pi * 3)
    step(1)
print(f"  B1 甩动中: q7={q7():+.4f} tilt={tilt():.2f}deg rawFz={s.raw_fz:+.2f} rawFx={s.raw_fx:+.2f}")
with s.lock:
    s.touch_force = 0.0
    s.touch_sx = 0.0
    s.touch_sy = 0.0
step(10000)
print(f"  B2 释放后20s: q7={q7():+.4f} tilt={tilt():.2f}deg rawFz={s.raw_fz:+.2f}")

print("\n===== C. arm模式几何验证 =====")
s.mode = "arm"
s.reset_scene()
step(1000)
print(f"  C1 force=0: gap={pad_face_gap_to_target()*1000:+.2f}mm (应为正值=间隙) q7={q7():+.4f} tilt={tilt():.2f}deg")
for f_cmd in (2.0, 5.0, 8.0):
    s.arm_force = f_cmd
    s.arm_q_goal = None  # 强制重解IK
    step(4000)
    print(f"  C2 force={f_cmd}N: gap={pad_face_gap_to_target()*1000:+.2f}mm rawFz={s.raw_fz:+.2f}N "
          f"tilt={tilt():.2f}deg q7={q7():+.4f}")
s.arm_force = 0.0
s.arm_q_goal = None
step(8000)
print(f"  C3 撤力后16s: gap={pad_face_gap_to_target()*1000:+.2f}mm rawFz={s.raw_fz:+.2f}N tilt={tilt():.2f}deg q7={q7():+.4f}")

print("\n完成")
