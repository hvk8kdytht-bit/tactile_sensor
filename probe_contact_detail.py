"""详细接触诊断 i=850..875: 接触点位置/法向/穿透 + 目标块姿态角速度 + 垫角点"""
import numpy as np
import mujoco
from kinova_sensor_sim import SimSensor

s = SimSensor()
m, d = s.model, s.data
TB = s.press_body_id
np.set_printoptions(precision=4, suppress=True)

ja = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "press_target_joint")
qa = m.jnt_qposadr[ja]
qd = m.jnt_dofadr[ja]

s.mode = "arm"
s.reset_scene()
s.arm_force = 5.0

for i in range(876):
    s.update_interaction()
    mujoco.mj_step(m, d)
    f = np.array(s.get_contact_forces())
    s.raw_fx, s.raw_fy, s.raw_fz = f
    s.fz_filt += 0.2 * (f[2] - s.fz_filt)

    if i >= 852:
        q = d.qpos[qa + 3:qa + 7]
        ang = float(2 * np.arcsin(min(1.0, float(np.linalg.norm(q[1:])))))
        w = d.qvel[qd + 3:qd + 6]
        v = d.qvel[qd:qd + 3]
        R, c = s._sensor_frame()
        face = c + R @ np.array([0, -0.0026, 0])
        print(f"i={i} dep={s.arm_depth*1000:+.3f}mm ang={np.degrees(ang):.2f}deg "
              f"w={w} v={v}")
        print(f"      face={face} tgt={d.xpos[TB]}")
        fbuf = np.zeros(6)
        for k in range(d.ncon):
            con = d.contact[k]
            mujoco.mj_contactForce(m, d, k, fbuf)
            g1, g2 = int(con.geom1), int(con.geom2)
            ids = {g1, g2}
            if s.press_geom_id in ids:
                print(f"      [{k}] pos={con.pos} dist={con.dist*1000:+.3f}mm "
                      f"n={con.frame[:3]} F={fbuf[:3]}")
