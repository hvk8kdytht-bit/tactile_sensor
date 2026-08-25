"""验证: 臂动模式右指垫不再穿模红色目标块
1) 接入+5N按压+撤力全程, 右指垫中心不得进入目标块真实体积(穿模特征)
2) 5N力闭环仍精确, 倾角正常
3) 目标块被瞬移5cm后由PD弹簧拉回原位(自由体不卡死机制)
"""
import numpy as np
import mujoco
from kinova_sensor_sim import SimSensor

s = SimSensor()
m, d = s.model, s.data
TB = s.press_body_id
HALF = np.array([0.014, 0.012, 0.014])
np.set_printoptions(precision=3, suppress=True)


def pad_in_target():
    c = d.geom_xpos[s.right_pad_main]
    loc = d.xmat[TB].reshape(3, 3).T @ (c - d.xpos[TB])
    return float((HALF - np.abs(loc)).min())


def pad_target_contact_pen():
    worst = 0.0
    for i in range(d.ncon):
        con = d.contact[i]
        ids = {int(con.geom1), int(con.geom2)}
        if s.press_geom_id in ids and ids & s.right_pad_geoms:
            worst = max(worst, -float(con.dist))
    return worst


def step(n):
    for _ in range(n):
        s.update_interaction()
        mujoco.mj_step(m, d)
        f = np.array(s.get_contact_forces())
        s.raw_fx, s.raw_fy, s.raw_fz = f
        s.fz_filt += 0.2 * (f[2] - s.fz_filt)


print("===== 1. 臂动接入+按压全程穿模检测 =====")
s.mode = "arm"
s.reset_scene()
ghost = 0.0
pen = 0.0
lat_done = None
max_fz = 0.0
max_tilt = 0.0
for i in range(5000):
    step(1)
    ghost = max(ghost, pad_in_target())
    pen = max(pen, pad_target_contact_pen())
    max_fz = max(max_fz, abs(s.raw_fz))
    max_tilt = max(max_tilt, abs(s.tilt_deg()))
    if lat_done is None and s.arm_phase == "press":
        lat_done = i
    if i == lat_done + 1200 if lat_done else (i == 2500):
        s.arm_force = 5.0
check = lambda ok, msg: print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
check(ghost < 0.0005, f"右垫中心不进目标体积 (最大侵入={ghost*1000:.2f}mm, 穿模时>5mm)")
check(pen < 0.003, f"右垫-目标接触穿透<3mm (最大={pen*1000:.2f}mm)")
check(max_fz < 15.0, f"全程无力爆 (峰值|rawFz|={max_fz:.2f}N, 爆炸时>100N)")
check(max_tilt < 5.0, f"全程倾角<5° (峰值={max_tilt:.2f}deg, 甩动时>50°)")
check(4.5 < s.raw_fz < 5.5 and s.tilt_deg() < 3,
      f"5N力闭环 rawFz={s.raw_fz:+.2f}N tilt={s.tilt_deg():.2f}deg")
print(f"  侧滑接入耗时: {lat_done}步({lat_done*2/1000:.1f}s仿真时间)")

print("===== 2. 撤力悬停 =====")
s.arm_force = 0.0
step(2500)
check(abs(s.raw_fz) < 0.5 and s.tilt_deg() < 3,
      f"撤力归零 rawFz={s.raw_fz:+.2f}N tilt={s.tilt_deg():.2f}deg")

print("===== 3. 目标被撞飞后弹回 =====")
ja = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "press_target_joint")
qa = m.jnt_qposadr[ja]
d.qpos[qa:qa + 3] += [0.05, 0.0, 0.03]
mujoco.mj_forward(m, d)
s.press_prev = None
step(1500)
res = np.linalg.norm(d.xpos[TB] - s.press_p0) * 1000
qerr = np.linalg.norm(d.xquat[TB] - s.press_q0)
check(res < 2.0, f"弹回原位 残差={res:.2f}mm 姿态误差={qerr:.4f}")
check(abs(s.raw_fz) < 0.5 and s.tilt_deg() < 3,
      f"弹回后无残余力 rawFz={s.raw_fz:+.2f}N tilt={s.tilt_deg():.2f}deg")
