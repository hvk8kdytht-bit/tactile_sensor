"""逐帧追踪第一次碰撞: i=40..320, 记录垫面Y位置/目标位移/接触对"""
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

for i in range(330):
    s.update_interaction()
    mujoco.mj_step(m, d)
    f = np.array(s.get_contact_forces())
    s.raw_fx, s.raw_fy, s.raw_fz = f
    s.fz_filt += 0.2 * (f[2] - s.fz_filt)

    if i < 40 or i % 10 == 0 or d.ncon > 0:
        R, c = s._sensor_frame()
        face = c + R @ np.array([0, -0.004, 0])
        # 垫面相对目标近面(-0.0181)的间隙: 负=穿透
        gap = float((face - d.xpos[TB]) @ np.array([0, 1, 0]) - 0.012) * 1000
        tdis = d.xpos[TB] - s.press_p0
        cons = []
        fbuf = np.zeros(6)
        for k in range(d.ncon):
            con = d.contact[k]
            mujoco.mj_contactForce(m, d, k, fbuf)
            cons.append((gm[int(con.geom1)], gm[int(con.geom2)],
                         round(float(np.linalg.norm(fbuf[:3])), 1)))
        ph = s.arm_phase
        if i < 40 or i % 10 == 0 or cons:
            print(f"i={i:3d} ph={ph:7s} gap={gap:+7.2f}mm tgtD={np.round(tdis*1000,1)} "
                  f"fzF={s.fz_filt:+7.1f} tilt={s.tilt_deg():5.1f} qerr={float(np.abs(d.qpos[:7]-s.arm_q_ref).max()):.3f}")
            for c1, c2, fn in cons:
                if fn > 0.5:
                    print(f"      {c1} <-> {c2}: {fn}N")
