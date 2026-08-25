"""探针: touch 模式 6N 指令下的接触力细节"""
import os
import sys

import numpy as np
import mujoco

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from kinova_sensor_sim import SimSensor

s = SimSensor()
dt = s.model.opt.timestep

s.mode = "touch"
s.reset_scene()
with s.lock:
    s.touch_force, s.touch_sx, s.touch_sy = 6.0, 0.0, 0.0

print(f"dt={dt}  指令: touch_force=6N -> 命令压深={6.0/96000*1e6:.1f}um")
for i in range(2600):
    s.update_interaction()
    mujoco.mj_step(s.model, s.data)
    if i >= 2000 and (i - 2000) % 50 == 0:
        f = np.array(s.get_contact_forces())
        # 计算touch box与传感器面的实际穿透
        R = s.data.geom_xmat[s.sensor_geom_id].reshape(3, 3)
        c = s.data.geom_xpos[s.sensor_geom_id]
        box_p = s.data.mocap_pos[s.touch_mocap]
        box_c = box_p - (c + R @ np.array([0, -0.0026, 0]))
        depth_local = R.T @ box_c
        ncon_s = sum(1 for j in range(s.data.ncon)
                     if s.sensor_geom_id in (s.data.contact[j].geom1, s.data.contact[j].geom2))
        print(f"step {i}: Fz={f[2]:7.2f} Fx={f[0]:6.2f} Fy={f[1]:6.2f} "
              f"接触点={ncon_s} box局部坐标(y,z)={depth_local[1]*1e6:7.1f},{depth_local[2]*1e6:7.1f}um")

# 统计最后500步
vals = []
for i in range(500):
    s.update_interaction()
    mujoco.mj_step(s.model, s.data)
    vals.append(s.get_contact_forces()[2])
v = np.array(vals)
print(f"\n最后500步 Fz: mean={v.mean():.2f} std={v.std():.2f} min={v.min():.2f} max={v.max():.2f}")
