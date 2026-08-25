"""手指连杆加固参数探针
测试: 1) 不同腱预紧下指垫的保持刚度与扰动恢复速度
      2) 弹簧刚度提高后夹取闭合是否仍然可用
目标: 找到"指垫稳固+闭合正常"的参数组合
"""
import time
import numpy as np
import mujoco
from kinova_sensor_sim import SimSensor

s = SimSensor()
dt = s.model.opt.timestep

FINGER_JOINTS = ["right_driver_joint", "right_coupler_joint", "right_spring_link_joint",
                 "right_follower_joint", "left_driver_joint", "left_coupler_joint",
                 "left_spring_link_joint", "left_follower_joint"]
JID = {n: mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in FINGER_JOINTS}
QADR = {n: s.model.jnt_qposadr[JID[n]] for n in FINGER_JOINTS}
HOME_F = {n: s.home_qpos[QADR[n]] for n in FINGER_JOINTS}
SPRING_J = [JID["left_spring_link_joint"], JID["right_spring_link_joint"]]
print("home手指关节角:", {n.split('_')[0] + "_" + n.split('_')[1][:4]: round(HOME_F[n], 3) for n in FINGER_JOINTS})

sensor_body = s.sensor_body_id


def tilt_deg():
    R = s.data.geom_xmat[s.sensor_geom_id].reshape(3, 3)
    Rel = R @ s.arm_R0.T
    w = np.empty(4)
    mujoco.mju_mat2Quat(w, Rel.reshape(9))
    return float(np.degrees(2 * np.arcsin(min(1.0, np.linalg.norm(w[1:])))))


def pad_gap():
    """左右指垫间距 mm: 用两个silicone pad上的点位"""
    lp = s.data.geom_xpos[s.sensor_geom_id]
    rid = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_GEOM, "right_pad1")
    rp = s.data.geom_xpos[rid]
    return float(np.linalg.norm(rp - lp) * 1000)


def step(n, finger_ctrl=None):
    for _ in range(n):
        if finger_ctrl is not None:
            s.data.ctrl[s.finger_act] = finger_ctrl
        s.update_interaction()
        mujoco.mj_step(s.model, s.data)


def disturb_recover(tag, torque=0.8, hold_steps=500):
    """施加0.5s腕部扰动 -> 测恢复时间"""
    s.data.xfrc_applied[sensor_body, 3:6] = [0.0, torque, torque * 0.5]
    for _ in range(hold_steps):
        mujoco.mj_step(s.model, s.data)
    s.data.xfrc_applied[sensor_body, :] = 0.0
    t0 = tilt_deg()
    rec = None
    for i in range(4000):
        mujoco.mj_step(s.model, s.data)
        if tilt_deg() < 1.0:
            rec = i * dt
            break
    print(f"  [{tag}] 扰动中tilt={t0:6.1f}deg -> 回到<1deg耗时 {rec if rec else '>8s':>8} s(仿真)")
    return rec


print("\n===== 1. 基线: 腱ctrl=0 (当前touch模式状态) =====")
s.reset_scene()
step(2000)
print(f"  静止: tilt={tilt_deg():.2f}deg padGap={pad_gap():.1f}mm")
disturb_recover("baseline")

print("\n===== 2. 腱预紧测试 (touch模式下指垫保持) =====")
for ctrl in (0.0, 2.0, 5.0, 10.0, 20.0):
    s.reset_scene()
    step(1500, finger_ctrl=ctrl)
    gap = pad_gap()
    tilt0 = tilt_deg()
    rec = disturb_recover(f"ctrl={ctrl:5.1f}")
    print(f"        预紧后 padGap={gap:5.1f}mm tilt={tilt0:.2f}deg")

print("\n===== 3. 弹簧刚度提高 x 预紧 (闭合能力+恢复) =====")
orig_k = [s.model.jnt_stiffness[j] for j in SPRING_J]
orig_d = [s.model.dof_damping[s.model.jnt_dofadr[j]] for j in SPRING_J]
for k in (0.05, 0.15, 0.30):
    for j in SPRING_J:
        s.model.jnt_stiffness[j] = k
        dof = s.model.jnt_dofadr[j]
        s.model.dof_damping[dof] = 0.05
    # 闭合能力: grasp 70% 是否仍产生夹紧力
    s.mode = "grasp"
    s.reset_scene()
    s.grasp_pct = 70.0
    step(5000)
    fz70 = s.raw_fz
    gap70 = pad_gap()
    # 扰动恢复(touch模式+预紧10)
    s.mode = "touch"
    s.reset_scene()
    step(1500, finger_ctrl=10.0)
    rec = disturb_recover(f"k={k:.2f}")
    print(f"        grasp70%: rawFz={fz70:6.2f}N padGap={gap70:5.1f}mm")
    for j, k0, d0 in zip(SPRING_J, orig_k, orig_d):
        s.model.jnt_stiffness[j] = k0
        s.model.dof_damping[s.model.jnt_dofadr[j]] = d0

print("\n===== 4. follower关节阻尼提高 (0.1 -> 0.5) + 预紧10 =====")
for n in ("left_follower_joint", "right_follower_joint"):
    dof = s.model.jnt_dofadr[JID[n]]
    s.model.dof_damping[dof] = 0.5
s.reset_scene()
step(1500, finger_ctrl=10.0)
disturb_recover("follower阻尼0.5")
print("完成")
