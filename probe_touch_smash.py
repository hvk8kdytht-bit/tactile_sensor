"""无头复现: touch盒子往返路径是否撞翻机械臂
模拟用户在网页按住触控板(5N)再松开, 观察倾角和臂上接触
"""
import numpy as np
import mujoco
from kinova_sensor_sim import SimSensor

s = SimSensor()
dt = s.model.opt.timestep


def arm_contact_fz():
    tot = 0.0
    for i in range(s.data.ncon):
        c = s.data.contact[i]
        b1 = s.model.geom_bodyid[c.geom1]
        b2 = s.model.geom_bodyid[c.geom2]
        if (b1 != 0 and b2 != 0) and not (b1 == s.touch_body or b2 == s.touch_body):
            f = np.zeros(6)
            mujoco.mj_contactForce(s.model, s.data, i, f)
            tot += abs(f[0])
    return tot


def step(n):
    for _ in range(n):
        s.update_interaction()
        mujoco.mj_step(s.model, s.data)
        f = np.array(s.get_contact_forces())
        s.raw_fx, s.raw_fy, s.raw_fz = f
        s.fz_filt += 0.2 * (f[2] - s.fz_filt)
        s.tilt_now = s.tilt_deg()


step(1000)
print(f"启动后: tilt={s.tilt_deg():.2f}deg")

with s.lock:
    s.touch_force = 5.0
    s.touch_sx = 0.0
    s.touch_sy = 0.0

for phase in range(6):
    step(1000)
    print(f"按压{phase+1}k步: tilt={s.tilt_deg():.2f}deg rawFz={s.raw_fz:+.2f}N "
          f"box@{np.round(s.data.mocap_pos[s.touch_mocap], 3)}")

with s.lock:
    s.touch_force = 0.0

for phase in range(4):
    step(1000)
    print(f"松开{phase+1}k步: tilt={s.tilt_deg():.2f}deg rawFz={s.raw_fz:+.2f}N")

step(3000)
print(f"最终: tilt={s.tilt_deg():.2f}deg")
