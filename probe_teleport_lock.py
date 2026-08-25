"""无头复现3: 最坏情形
A) touch mocap盒被用户在3D视窗拖进机械臂内部(mocap体可被视窗直接移动),
   与代码每步覆写mocap_pos竞争 -> 盒子嵌入臂内 -> 求解器巨力弹射
B) 关节直接瞬移到大偏角(视窗ctrl+drag旋转扰动等效), 看伺服能否拉回
"""
import numpy as np
import mujoco
from kinova_sensor_sim import SimSensor

s = SimSensor()
dt = s.model.opt.timestep
np.set_printoptions(precision=3, suppress=True)


def step(n):
    for _ in range(n):
        s.update_interaction()
        mujoco.mj_step(s.model, s.data)
        f = np.array(s.get_contact_forces())
        s.raw_fx, s.raw_fy, s.raw_fz = f
        s.fz_filt += 0.2 * (f[2] - s.fz_filt)
        s.tilt_now = s.tilt_deg()


def lock_contacts():
    out = []
    for i in range(s.data.ncon):
        c = s.data.contact[i]
        b1 = s.model.geom_bodyid[c.geom1]
        b2 = s.model.geom_bodyid[c.geom2]
        if b1 == 0 or b2 == 0 or b1 == s.touch_body or b2 == s.touch_body:
            continue
        g1 = mujoco.mj_id2name(s.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or "?"
        g2 = mujoco.mj_id2name(s.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or "?"
        f = np.zeros(6)
        mujoco.mj_contactForce(s.model, s.data, i, f)
        if abs(f[0]) > 5:
            out.append((f"{g1}|{g2}", abs(f[0])))
    return sorted(out, key=lambda x: -x[1])[:4]


print("===== A) mocap盒瞬移进臂内部 =====")
s.mode = "touch"
s.reset_scene()
step(500)
# 用户在视窗把橙盒拖到腕部中心(模拟视窗直接搬mocap)
s.data.mocap_pos[s.touch_mocap] = np.array([0.55, 0.0, 0.45])
with s.lock:
    s.touch_force = 5.0
step(2000)  # 代码会把touch_pos从PARK限速飞过来, 期间盒子在臂内
with s.lock:
    s.touch_force = 0.0
step(3000)
err = s.data.qpos[:7] - s.home_qpos[:7]
worst = int(np.argmax(np.abs(err)))
print(f"A结果: tilt={s.tilt_deg():.1f}deg j{worst+1}err={err[worst]:+.2f}rad "
      f"{'卡死' if np.abs(err).max() > 0.05 else '恢复'}")
for pair, fn in lock_contacts():
    print(f"    锁定: {pair}  {fn:.0f}N")

print("===== B) 关节大偏角瞬移 =====")
for jname, delta in [("joint_2", 1.2), ("joint_5", 1.5), ("joint_6", 1.5),
                     ("joint_7", 2.0), ("joint_4", 1.5)]:
    s.mode = "touch"
    s.reset_scene()
    jid = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    qadr = s.model.jnt_qposadr[jid]
    rng = s.model.jnt_range[jid]
    target = np.clip(s.home_qpos[qadr] + delta, rng[0] + 0.05, rng[1] - 0.05)
    s.data.qpos[qadr] = target
    s.data.qvel[:] = 0
    step(3000)
    err = s.data.qpos[:7] - s.home_qpos[:7]
    worst = int(np.argmax(np.abs(err)))
    stuck = np.abs(err).max() > 0.05
    print(f"{jname} 瞬移{delta:+.1f}rad: tilt={s.tilt_deg():6.1f}deg "
          f"j{worst+1}err={err[worst]:+6.2f}rad {'卡死' if stuck else '恢复'}")
    if stuck:
        for pair, fn in lock_contacts():
            print(f"    锁定: {pair}  {fn:.0f}N")
