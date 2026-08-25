"""grasp死锁诊断2: 跟踪薄板/右指垫接触/指间距/手指关节"""
import numpy as np
import mujoco
from kinova_sensor_sim import SimSensor

s = SimSensor()
dt = s.model.opt.timestep
GQ = s.model.jnt_qposadr[s.grasp_jnt]
LQ = s.model.jnt_qposadr[mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_JOINT, "left_driver_joint")]
RQ = s.model.jnt_qposadr[mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint")]
LG = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_GEOM, "left_pad1")
RG = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_GEOM, "right_pad1")

s.mode = "grasp"
s.reset_scene()
mujoco.mj_forward(s.model, s.data)
print(f"geom check: grasp={s.grasp_geom_id} rightpads={s.right_pad_geoms}")
print(f"初始: 板=({s.data.qpos[GQ]:.3f},{s.data.qpos[GQ+1]:.3f},{s.data.qpos[GQ+2]:.3f}) "
      f"Lpad={s.data.geom_xpos[LG]} Rpad={s.data.geom_xpos[RG]}")
s.grasp_pct = 70.0
for i in range(8000):
    s.update_interaction()
    mujoco.mj_step(s.model, s.data)
    f = np.array(s.get_contact_forces())
    s.raw_fx, s.raw_fy, s.raw_fz = f
    s.fz_filt += 0.2 * (f[2] - s.fz_filt)
    if i % 1000 == 0:
        p = s.data.qpos[GQ:GQ + 3]
        touch = s._right_pad_touches_plate()
        gap_pad = np.linalg.norm(s.data.geom_xpos[LG] - s.data.geom_xpos[RG]) * 1000
        print(f"t={i*dt:4.1f}s 板=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) 右垫碰板={touch} "
              f"rawFz={s.raw_fz:+6.2f} 指间距={gap_pad:5.1f}mm Ldrv={s.data.qpos[LQ]:+.3f} Rdrv={s.data.qpos[RQ]:+.3f}")
print("完成")
