"""长探针: touch 模式 6N 指令 60s, 看力是否收敛 + 关节漂移"""
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

q0 = s.data.qpos[:7].copy()
finger_names = ["joint_finger_driver", "joint_finger_follower"]
print(f"指令6N -> 命令压深 {6.0/96000*1e6:.1f}um, 运行60s")
for i in range(30000):
    s.update_interaction()
    mujoco.mj_step(s.model, s.data)
    if i % 2500 == 0:
        f = np.array(s.get_contact_forces())
        dq = s.data.qpos[:7] - q0
        R = s.data.geom_xmat[s.sensor_geom_id].reshape(3, 3)
        c = s.data.geom_xpos[s.sensor_geom_id]
        box_c = R.T @ (s.data.mocap_pos[s.touch_mocap] - (c + R @ np.array([0, -0.0026, 0])))
        print(f"t={i*dt:5.1f}s Fz={f[2]:7.2f} Fx={f[0]:6.2f} Fy={f[1]:6.2f} "
              f"臂关节最大漂移={np.abs(dq).max()*1000:6.2f}mrad box_y={box_c[1]*1e6:8.1f}um")
