"""快速验证: arm模式IK解的姿态质量 + 各深度下的IK收敛情况"""
import numpy as np
import mujoco
from kinova_sensor_sim import SimSensor

s = SimSensor()
np.set_printoptions(precision=3, suppress=True)


def pose_tilt(q):
    d2 = s.ik_data
    d2.qpos[:] = s.home_qpos
    d2.qpos[:7] = q
    mujoco.mj_forward(s.model, d2)
    R = d2.geom_xmat[s.sensor_geom_id].reshape(3, 3)
    Rel = R @ s.arm_R0.T
    w = np.empty(4)
    mujoco.mju_mat2Quat(w, Rel.reshape(9))
    t = float(np.degrees(2 * np.arcsin(min(1.0, np.linalg.norm(w[1:])))))
    face = d2.geom_xpos[s.sensor_geom_id] + R @ np.array([0, -0.0026, 0])
    return t, face


# IK目标: 不同深度 (standby间隙 -5mm 到压入 8.58mm)
for depth_mm in (-5.0, -2.0, 0.0, 2.19, 5.0, 7.19, 8.58):
    tp = s.arm_tp + s.arm_n_world * (depth_mm / 1000.0)
    q = s._ik(s.arm_home_q, tp, s.arm_R0)
    t, face = pose_tilt(q)
    err = np.linalg.norm(face - tp) * 1000
    print(f"depth={depth_mm:+6.2f}mm -> IK解 tilt={t:6.2f}deg  位置误差={err:6.2f}mm  q7={q[6]:+.3f}")

# home本身的pad面位置
t0, face0 = pose_tilt(s.arm_home_q)
print(f"\nhome: pad面={face0}  目标近面={s.arm_tp}")
print(f"home时pad面到目标近面距离 = {float(s.arm_n_world @ (s.arm_tp - face0))*1000:+.2f}mm")
