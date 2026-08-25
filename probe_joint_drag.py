"""无头复现: 视窗拖拽关节后臂/夹爪能否回home
模拟 mujoco.viewer 的关节拖拽: 对关节施加外力矩一段时间再释放,
之后伺服应把关节拉回home。若被自碰撞锁死则永远回不去。
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


def arm_lock_contacts():
    """臂/夹爪本体上的接触对及法向力(排除外部物体)"""
    out = []
    for i in range(s.data.ncon):
        c = s.data.contact[i]
        b1 = s.model.geom_bodyid[c.geom1]
        b2 = s.model.geom_bodyid[c.geom2]
        if b1 == 0 or b2 == 0:
            continue
        if b1 == s.touch_body or b2 == s.touch_body:
            continue
        g1 = mujoco.mj_id2name(s.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or "?"
        g2 = mujoco.mj_id2name(s.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or "?"
        f = np.zeros(6)
        mujoco.mj_contactForce(s.model, s.data, i, f)
        out.append((f"{g1}|{g2}", abs(f[0])))
    return sorted(out, key=lambda x: -x[1])[:3]


TESTS = [
    ("joint_1", 30.0), ("joint_2", 15.0), ("joint_3", 15.0), ("joint_4", 10.0),
    ("joint_5", 8.0), ("joint_6", 5.0), ("joint_7", 3.0),
    ("right_driver_joint", 2.0), ("left_driver_joint", 2.0),
    ("right_follower_joint", 1.0), ("left_follower_joint", 1.0),
]

for jname, torque in TESTS:
    s.mode = "touch"
    s.reset_scene()
    jid = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    if jid < 0:
        print(f"{jname}: 不存在"); continue
    dof = s.model.jnt_dofadr[jid]
    qadr = s.model.jnt_qposadr[jid]
    home_q = s.home_qpos[qadr]
    # 拖拽阶段: 施加力矩2000步(约4s仿真), 模拟用户按住关节拖
    s.data.qfrc_applied[dof] = torque
    step(2000)
    dragged = s.data.qpos[qadr] - home_q
    # 释放阶段: 撤掉外力, 看伺服能否拉回home
    s.data.qfrc_applied[:] = 0
    step(4000)
    err = s.data.qpos[qadr] - home_q
    cons = arm_lock_contacts()
    lock = " <== 卡死!" if abs(err) > 0.05 else ""
    print(f"{jname:22s} 拖到{dragged:+6.2f}rad 释放后误差{err:+6.2f}rad "
          f"tilt={s.tilt_deg():6.1f}deg{lock}")
    if abs(err) > 0.05 and cons:
        for pair, fn in cons:
            print(f"    锁定接触: {pair}  {fn:.0f}N")
