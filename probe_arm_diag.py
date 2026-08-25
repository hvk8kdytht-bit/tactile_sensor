"""臂动模式甩动时序诊断: 逐段记录倾角/力/关节误差/接触对, 定位发散起点"""
import numpy as np
import mujoco
from kinova_sensor_sim import SimSensor

s = SimSensor()
m, d = s.model, s.data
TB = s.press_body_id
np.set_printoptions(precision=3, suppress=True)

# 0) 走廊终点与锚点表收敛性核查
mujoco.mj_forward(m, s.ik_data)
s.ik_data.qpos[:] = s.home_qpos
s.ik_data.qpos[:7] = s.arm_corridor_q[-1] if s.arm_corridor_q else s.arm_home_q
mujoco.mj_forward(m, s.ik_data)
R = s.ik_data.geom_xmat[s.sensor_geom_id].reshape(3, 3)
face = s.ik_data.geom_xpos[s.sensor_geom_id] + R @ np.array([0, -0.0026, 0])
p_align = s.arm_tp - s.arm_n_world * s.arm_home_clear
print("IK收敛核查: 走廊终点面位置 =", np.round(face, 4), " 期望 =", np.round(p_align, 4))
print("  残差 =", round(float(np.linalg.norm(face - p_align)) * 1000, 2), "mm")
print("  走廊路点数:", len(s.arm_corridor_q), " home_clear =",
      round(s.arm_home_clear * 1000, 1), "mm")
print("  arm_home_q =", np.round(s.arm_home_q, 3))

gm = {}  # geom id -> name
for g in range(m.ngeom):
    gm[g] = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"


def contacts_summary():
    """返回(涉及目标块的最大接触力, 涉及右垫的接触描述, 总接触力最大值)"""
    ft = 0.0
    fr = 0.0
    pairs_t = []
    pairs_r = []
    fbuf = np.zeros(6)
    for i in range(d.ncon):
        con = d.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        mujoco.mj_contactForce(m, d, i, fbuf)
        fn = float(np.linalg.norm(fbuf[:3]))
        ids = {g1, g2}
        if s.press_geom_id in ids:
            ft = max(ft, fn)
            other = g2 if g1 == s.press_geom_id else g1
            pairs_t.append((gm[other], round(fn, 1)))
        if ids & s.right_pad_geoms:
            fr = max(fr, fn)
            other = next(g for g in ids if g not in s.right_pad_geoms)
            pairs_r.append((gm[other], round(fn, 1)))
    return ft, fr, pairs_t[:3], pairs_r[:3]


def step(n):
    for _ in range(n):
        s.update_interaction()
        mujoco.mj_step(m, d)
        f = np.array(s.get_contact_forces())
        s.raw_fx, s.raw_fy, s.raw_fz = f
        s.fz_filt += 0.2 * (f[2] - s.fz_filt)


print("\n===== 臂动模式时序诊断 (每100步采样) =====")
s.mode = "arm"
s.reset_scene()
s.arm_force = 5.0  # 一开始就给指令力, 观察接入全程

bad_t = None
for i in range(4000):
    step(1)
    if i % 100 == 0 or (bad_t is None and s.tilt_deg() > 30):
        if bad_t is None and s.tilt_deg() > 30:
            bad_t = i
            print(f"--- 倾角超30° 首次出现于步{i} ---")
        qerr = float(np.abs(d.qpos[:7] - s.arm_q_ref).max())
        tdis = float(np.linalg.norm(d.xpos[TB] - s.press_p0)) * 1000
        ft, fr, pt, pr = contacts_summary()
        nan = np.any(~np.isfinite(d.qacc))
        print(f"i={i:4d} ph={s.arm_phase:7s} tilt={s.tilt_deg():6.1f} rawFz={s.raw_fz:+6.2f} "
              f"fzF={s.fz_filt:+6.2f} dep={s.arm_depth*1000:6.3f}mm qerr={qerr:5.3f} "
              f"tgtΔ={tdis:5.1f}mm F_tgt={ft:6.1f} F_rpad={fr:6.1f} nan={nan}")
        if pt:
            print(f"        目标接触: {pt}")
        if pr and fr > 1.0:
            print(f"        右垫接触: {pr}")

print(f"\n首现倾角>30°的步数: {bad_t}")
print(f"最终: tilt={s.tilt_deg():.1f}deg rawFz={s.raw_fz:+.2f}N fzF={s.fz_filt:+.2f}N")
print(f"最终关节误差: {np.round(d.qpos[:7] - s.arm_q_ref, 3)}")
print(f"最终臂角: {np.round(d.qpos[:7], 3)}")
print(f"参考臂角: {np.round(s.arm_q_ref, 3)}")
