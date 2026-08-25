import mujoco
import numpy as np

MODEL = r"C:\Users\cuizi\AppData\Roaming\TRAE SOLO CN\ModularData\ai-agent\work-mode-projects\6a8bb58b84243539c48c3256\mujoco_menagerie\kinova_gen3\gen3_sensor.xml"
m = mujoco.MjModel.from_xml_path(MODEL)
d = mujoco.MjData(m)
key = m.key("home")
d.qpos[:] = key.qpos
d.ctrl[:] = key.ctrl
mujoco.mj_forward(m, d)
bias = d.qfrc_bias.copy()
acts = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint_{i}") for i in range(1, 8)]
home_ctrl = key.ctrl.copy()
for aid in acts:
    jid = m.actuator_trnid[aid][0]
    dof = m.jnt_dofadr[jid]
    kp = m.actuator_gainprm[aid][0]
    home_ctrl[aid] += bias[dof] / kp
d.ctrl[:] = home_ctrl

gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "sensor_pad")
wid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "contact_wall")

R = d.geom_xmat[gid].reshape(3, 3)
z_w = R[:, 2]
pad_c = d.geom_xpos[gid].copy()
outer = pad_c + 0.0026 * z_w
print(f"pad center {np.round(pad_c,4)}  z_world {np.round(z_w,2)}  outer_face {np.round(outer,4)}")
print(f"wall face x = {d.geom_xpos[wid][0] + m.geom_size[wid][0]:.4f}")

for _ in range(1500):
    mujoco.mj_step(m, d)

R = d.geom_xmat[gid].reshape(3, 3)
z_w = R[:, 2]
pad_c = d.geom_xpos[gid].copy()
outer = pad_c + 0.0026 * z_w
print(f"settled: pad center {np.round(pad_c,4)} outer_face {np.round(outer,4)}")
print(f"qpos err: {np.round(d.qpos[:7]-key.qpos[:7],5)}")
print(f"ncon={d.ncon}")
for i in range(d.ncon):
    c = d.contact[i]
    n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
    n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
    f = np.zeros(6)
    mujoco.mj_contactForce(m, d, i, f)
    print(f"  {n1} | {n2} dist={c.dist:.5f} pos={np.round(c.pos,4)} f={np.round(f[:3],2)}")
