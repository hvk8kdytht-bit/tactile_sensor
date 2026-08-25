"""隔离测试: 红色目标块PD姿态保持的数值稳定性(无臂动/无接触)
1) 初始姿态扰动30° -> 应平滑回到0°
2) 初始角速度扰动 -> 应衰减
3) 打印 w_est(四元数差分) vs d.qvel角速度 vs 施加力矩 -> 查坐标系/符号
"""
import numpy as np
import mujoco
from kinova_sensor_sim import SimSensor

s = SimSensor()
m, d = s.model, s.data
TB = s.press_body_id
ja = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "press_target_joint")
qa = m.jnt_qposadr[ja]
qd = m.jnt_dofadr[ja]
np.set_printoptions(precision=3, suppress=True)
print("body_inertia =", m.body_inertia[TB].tolist(), " integrator =", int(m.opt.integrator),
      " dt =", m.opt.timestep)


def blk_ang():
    q = d.qpos[qa + 3:qa + 7]
    return np.degrees(2 * np.arcsin(min(1.0, float(np.linalg.norm(q[1:])))))


print("===== 1. 姿态扰动30°回正 =====")
ang = np.radians(30)
d.qpos[qa + 3:qa + 7] = [np.cos(ang / 2), np.sin(ang / 2), 0, 0]
mujoco.mj_forward(m, d)
s.press_prev = None
for i in range(300):
    s._hold_press_target()
    if i in (0, 1, 2, 5, 10, 20, 50, 100, 200, 299):
        print(f"i={i:3d} ang={blk_ang():7.2f}deg qvel_w={d.qvel[qd+3:qd+6]} "
              f"tau={d.xfrc_applied[TB, 3:]}")
    mujoco.mj_step(m, d)

print("===== 2. 角速度扰动(3rad/s)衰减 =====")
mujoco.mj_forward(m, d)
s.press_prev = None
d.qvel[qd + 3:qd + 6] = [3.0, 0, 0]
for i in range(60):
    s._hold_press_target()
    if i in (0, 1, 2, 3, 5, 10, 20, 40, 59):
        print(f"i={i:3d} ang={blk_ang():7.2f}deg qvel_w={d.qvel[qd+3:qd+6]} "
              f"tau={d.xfrc_applied[TB, 3:]}")
    mujoco.mj_step(m, d)

print("===== 3. 小扰动(0.3rad/s)长跑发散检测 =====")
mujoco.mj_forward(m, d)
s.press_prev = None
d.qvel[qd + 3:qd + 6] = [0.3, 0, 0]
hist = []
for i in range(2000):
    s._hold_press_target()
    mujoco.mj_step(m, d)
    hist.append(blk_ang())
hist = np.array(hist)
print(f"前100步最大ang={hist[:100].max():.3f}deg  末100步最大ang={hist[-100:].max():.3f}deg "
      f"全程最大={hist.max():.3f}deg")
print("发散" if hist[-100:].max() > 5.0 else "稳定")
