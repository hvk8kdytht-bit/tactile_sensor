"""逐帧追踪i=780..900爆炸瞬间: 面位置/目标位置/所有接触对/关节加速度"""
import numpy as np
import mujoco
from kinova_sensor_sim import SimSensor

s = SimSensor()
m, d = s.model, s.data
TB = s.press_body_id
np.set_printoptions(precision=3, suppress=True)

gm = {g: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}") for g in range(m.ngeom)}

s.mode = "arm"
s.reset_scene()
s.arm_force = 5.0

for i in range(910):
    s.update_interaction()
    mujoco.mj_step(m, d)
    f = np.array(s.get_contact_forces())
    s.raw_fx, s.raw_fy, s.raw_fz = f
    s.fz_filt += 0.2 * (f[2] - s.fz_filt)

    if i >= 780:
        R, c = s._sensor_frame()
        face = c + R @ np.array([0, -0.0026, 0])
        cons = []
        fbuf = np.zeros(6)
        for k in range(d.ncon):
            con = d.contact[k]
            mujoco.mj_contactForce(m, d, k, fbuf)
            cons.append((gm[int(con.geom1)], gm[int(con.geom2)],
                         round(float(np.linalg.norm(fbuf[:3])), 1),
                         round(-float(con.dist) * 1000, 2)))
        qacc = float(np.abs(d.qacc[:7]).max())
        print(f"i={i:3d} dep={s.arm_depth*1000:+7.3f} face_y={face[1]:.5f} "
              f"tgt_y={d.xpos[TB][1]:.5f} tgtΔ={np.linalg.norm(d.xpos[TB]-s.press_p0)*1000:6.1f}mm "
              f"tilt={s.tilt_deg():5.1f} qerr={float(np.abs(d.qpos[:7]-s.arm_q_ref).max()):.3f} "
              f"qacc={qacc:8.1f} fzF={s.fz_filt:+7.2f}")
        for c1, c2, fn, pen in cons:
            if fn > 0.1 or pen > 0.05:
                print(f"      {c1} <-> {c2}: {fn}N pen={pen}mm")
