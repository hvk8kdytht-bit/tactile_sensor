"""快速几何探针: home位姿下各物体相对位置 + 执行器增益"""
import numpy as np
import mujoco
from kinova_sensor_sim import SimSensor

s = SimSensor()
d = s.data
d.qpos[:] = s.home_qpos
d.ctrl[:] = s.home_ctrl
mujoco.mj_forward(s.model, d)

R = d.geom_xmat[s.sensor_geom_id].reshape(3, 3)
pad = d.geom_xpos[s.sensor_geom_id]
face = pad + R @ np.array([0, -0.0026, 0])
n = R @ np.array([0, -1, 0])
tgt = d.geom_xpos[s.press_geom_id]

print(f"传感器垫中心  = {np.round(pad, 4)}")
print(f"垫表面(face)  = {np.round(face, 4)}")
print(f"垫法线 n      = {np.round(n, 4)}")
print(f"红色目标中心  = {np.round(tgt, 4)}")
print(f"目标-垫面矢量 = {np.round(tgt - face, 4)}   沿n投影 = {np.dot(tgt - face, n) * 1000:.1f} mm")
print(f"arm_tp(臂压起始点) = {np.round(s.arm_tp, 4)}")
print(f"木箱停靠点 BOX_PARK = {np.round([0.45, 0.35, 0.15], 3)}")
print(f"grasp模式木箱放置点 = face + [0,0.0355,0] = {np.round(face + np.array([0, 0.0355, 0]), 4)}")
print()

# 执行器增益
print("===== 执行器增益 =====")
for i in range(s.model.nu):
    name = mujoco.mj_id2name(s.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    kp = s.model.actuator_gainprm[i][0]
    kv = s.model.actuator_biasprm[i][2] if s.model.actuator_biastype[i] == 1 else 0
    print(f"  {name:22s} kp={kp:10.1f}  kv={kv:8.2f}  ctrlrange={np.round(s.model.actuator_ctrlrange[i], 2)}")

print()
print(f"home_ctrl[finger_act] = {s.home_ctrl[s.finger_act]:.3f}")
print(f"重力 = {s.model.opt.gravity}")
print(f"home qpos[7:10] (手指) = {np.round(s.home_qpos[7:10], 4)}")

# 手指body
for b in range(s.model.nbody):
    name = mujoco.mj_id2name(s.model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    if "finger" in name.lower() or "2f85" in name.lower() or "tool" in name.lower():
        print(f"  body[{b}] {name}")
