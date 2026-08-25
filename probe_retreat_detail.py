"""撤力回退时序 + 瞬移弹回姿态解码:
1) 5N按压稳态 -> 撤力, 每100步打印倾角/压深/块姿态角 -> 判定5.2°倾角是
   瞬跳激励的衰减振荡还是持续漂移
2) 目标块瞬移5cm弹回, 打印四元数w符号/rvec/角速度 -> 判定姿态误差1.45是
   真旋转90°(楔死)还是四元数符号翻转(PD误判无误差)
"""
import numpy as np
import mujoco
from kinova_sensor_sim import SimSensor

s = SimSensor()
m, d = s.model, s.data
TB = s.press_body_id
ja = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "press_target_joint")
qa = m.jnt_qposadr[ja]
qd = m.jnt_dofadr[ja]
np.set_printoptions(precision=3, suppress=True)


def step(n):
    for _ in range(n):
        s.update_interaction()
        mujoco.mj_step(m, d)
        f = np.array(s.get_contact_forces())
        s.raw_fx, s.raw_fy, s.raw_fz = f
        s.fz_filt += 0.2 * (f[2] - s.fz_filt)


def blk_ang():
    q = d.qpos[qa + 3:qa + 7]
    return np.degrees(2 * np.arcsin(min(1.0, float(np.linalg.norm(q[1:])))))


print("===== 1. 5N按压至稳态 =====")
s.mode = "arm"
s.reset_scene()
s.arm_force = 5.0
step(5000)
print(f"稳态: rawFz={s.raw_fz:+.2f}N tilt={s.tilt_deg():.2f}deg dep={s.arm_depth*1000:+.3f}mm")

print("===== 2. 撤力回退时序 =====")
s.arm_force = 0.0
for i in range(2500):
    step(1)
    if i % 100 == 0 or i < 20:
        print(f"i={i:4d} tilt={s.tilt_deg():6.2f}dep={s.arm_depth*1000:+7.3f}mm "
              f"blk_ang={blk_ang():6.2f}deg rawFz={s.raw_fz:+6.2f}N")

print("===== 3. 瞬移5cm弹回时序 =====")
d.qpos[qa:qa + 3] += [0.05, 0.0, 0.03]
mujoco.mj_forward(m, d)
s.press_prev = None
for i in range(1500):
    step(1)
    if i % 100 == 0 or i < 30:
        q = d.qpos[qa + 3:qa + 7]
        w = d.qvel[qd + 3:qd + 6]
        res = np.linalg.norm(d.xpos[TB] - s.press_p0) * 1000
        print(f"i={i:4d} quat={q} w={w} res={res:6.2f}mm tilt={s.tilt_deg():6.2f}deg")
