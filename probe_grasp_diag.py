"""grasp薄板滑脱诊断: 跟踪薄板位置和传感器力随闭合度变化"""
import numpy as np
import mujoco
from kinova_sensor_sim import SimSensor

s = SimSensor()
dt = s.model.opt.timestep
GQ = s.model.jnt_qposadr[s.grasp_jnt]

s.mode = "grasp"
s.reset_scene()
mujoco.mj_forward(s.model, s.data)
p0 = s.data.qpos[GQ:GQ + 3].copy()
print(f"初始板位置: {p0}  pad面法线方向接触前")

for phase, pct in (("闭合到40%", 40.0), ("闭合到70%", 70.0)):
    s.grasp_pct = pct
    for i in range(4000):
        s.update_interaction()
        mujoco.mj_step(s.model, s.data)
        f = np.array(s.get_contact_forces())
        s.raw_fx, s.raw_fy, s.raw_fz = f
        s.fz_filt += 0.2 * (f[2] - s.fz_filt)
        if i % 1000 == 0:
            p = s.data.qpos[GQ:GQ + 3]
            dp = np.linalg.norm(p - p0)
            print(f"  [{phase} t={i*dt:.1f}s] 板位移={dp*1000:6.1f}mm  pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})  rawFz={s.raw_fz:+7.2f}N")
    p = s.data.qpos[GQ:GQ + 3]
    print(f"  [{phase} 末] 板pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) rawFz={s.raw_fz:+.2f}N fz_filt={s.fz_filt:+.2f}")
print("完成")
