"""探针: 收紧后的touch力闭环 - 瞬移启动 与 从停靠点飞行启动 两种场景"""
import os
import sys

import numpy as np
import mujoco

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from kinova_sensor_sim import SimSensor, PARK_POS

s = SimSensor()
dt = s.model.opt.timestep


def run(label, flight, n=9000):
    s.mode = "touch"
    s.reset_scene()
    with s.lock:
        s.touch_force, s.touch_sx, s.touch_sy = 6.0, 0.0, 0.0
    if flight:
        s.touch_pos = PARK_POS.copy()  # 模拟木块从停靠点飞来
    peak = 0.0
    hist = []
    for i in range(n):
        s.update_interaction()
        mujoco.mj_step(s.model, s.data)
        f = np.array(s.get_contact_forces())
        s.raw_fx, s.raw_fy, s.raw_fz = f
        s.fz_filt += 0.2 * (f[2] - s.fz_filt)
        peak = max(peak, f[2])
        hist.append(f[2])
    tail = np.array(hist[-1000:])
    print(f"[{label}] 峰值Fz={peak:6.2f}N  末段mean={tail.mean():6.3f}N "
          f"std={tail.std():.3f}N  depth={s.touch_depth*1e6:.1f}um")
    for t in (1, 2, 3, 5, 8, 12, 16):
        if t * 500 - 1 < len(hist):
            print(f"   t={t:2d}s: Fz={hist[t*500-1]:7.3f}N")


print("指令6N:")
run("瞬移启动", flight=False)
run("飞行启动(从停靠点26cm外)", flight=True)
