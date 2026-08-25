"""最终验证: 臂动力闭环+撤力间隙+倾角恢复 + grasp薄板夹取 + touch回归"""
import numpy as np
import mujoco
from kinova_sensor_sim import SimSensor

s = SimSensor()
dt = s.model.opt.timestep
GQ = s.model.jnt_qposadr[s.grasp_jnt]
np.set_printoptions(precision=3, suppress=True)
ok = []


def step(n):
    for _ in range(n):
        s.update_interaction()
        mujoco.mj_step(s.model, s.data)
        f = np.array(s.get_contact_forces())
        s.raw_fx, s.raw_fy, s.raw_fz = f
        s.fz_filt += 0.2 * (f[2] - s.fz_filt)
        s.tilt_now = s.tilt_deg()
        out = s.signal.process(f, dt)
        s.fx, s.fy, s.fz = out


def check(name, cond, detail):
    ok.append(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")


print("===== 1. arm 力闭环 =====")
s.mode = "arm"
s.reset_scene()
step(2000)
R0, c0 = s._sensor_frame()
face0 = c0 + R0 @ np.array([0, -0.0026, 0])
gap_mm = float(s.arm_n_world @ (s.arm_tp - face0)) * 1000
check("撤力悬停真间隙", 3.0 < gap_mm < 7.0, f"间隙={gap_mm:.1f}mm rawFz={s.raw_fz:+.2f}N")
res = {}
for f_cmd in (2.0, 5.0, 8.0):
    s.arm_force = f_cmd
    step(6000)
    res[f_cmd] = s.raw_fz
    check(f"臂压{f_cmd}N闭环", abs(s.raw_fz - f_cmd) < 0.3, f"rawFz={s.raw_fz:+.2f}N tilt={s.tilt_now:.2f}deg")
s.arm_force = 0.0
step(6000)
check("撤力完全回位", abs(s.raw_fz) < 0.1 and s.tilt_now < 1.0,
      f"rawFz={s.raw_fz:+.2f}N tilt={s.tilt_now:.2f}deg")

print("===== 2. grasp 薄板夹取 =====")
s.mode = "grasp"
s.reset_scene()
p0 = s.data.qpos[GQ:GQ + 3].copy()
# 薄板仅15mm厚且贴传感器侧, ~76%闭合才接触, 用100%全闭合验证夹持
s.grasp_pct = 100.0
step(6000)
p1 = s.data.qpos[GQ:GQ + 3].copy()
check("薄板被稳定夹住", s.fz_filt > 5.0, f"rawFz={s.raw_fz:+.2f}N 板位移={np.linalg.norm(p1-p0)*1000:.1f}mm tilt={s.tilt_now:.2f}deg")
fz70 = s.raw_fz
step(3000)
check("夹持力保持", abs(s.raw_fz - fz70) < 1.0, f"rawFz={s.raw_fz:+.2f}N(3s前{fz70:+.2f})")
s.grasp_pct = 0.0
step(4000)
check("松开后力归零", abs(s.raw_fz) < 0.5, f"rawFz={s.raw_fz:+.2f}N tilt={s.tilt_now:.2f}deg")

print("===== 3. touch 回归 =====")
s.mode = "touch"
s.reset_scene()
with s.lock:
    s.touch_force = 6.0
step(6000)
check("touch 6N", abs(s.raw_fz - 6.0) < 0.3, f"rawFz={s.raw_fz:+.2f}N tilt={s.tilt_now:.2f}deg")
with s.lock:
    s.touch_force = 0.0
step(6000)
check("touch释放回水平", abs(s.raw_fz) < 0.1 and s.tilt_now < 1.0, f"rawFz={s.raw_fz:+.2f}N tilt={s.tilt_now:.2f}deg")

print("===== 4. 模式轮换后水平 =====")
s.mode = "touch"
s.reset_scene()
step(2000)
check("最终水平保持", s.tilt_now < 0.5 and abs(s.data.qpos[6] - 1.5708) < 0.01,
      f"tilt={s.tilt_now:.2f}deg q7={s.data.qpos[6]:+.4f}")

print(f"\n总计: {sum(ok)}/{len(ok)} 通过")
