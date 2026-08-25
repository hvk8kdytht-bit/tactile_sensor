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
home_ctrl = key.ctrl.copy()
acts = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint_{i}") for i in range(1, 8)]
for aid in acts:
    jid = m.actuator_trnid[aid][0]
    dof = m.jnt_dofadr[jid]
    kp = m.actuator_gainprm[aid][0]
    home_ctrl[aid] += bias[dof] / kp
for _ in range(2):
    d.qpos[:] = key.qpos
    d.qvel[:] = 0
    d.ctrl[:] = home_ctrl
    for _ in range(1500):
        mujoco.mj_step(m, d)
    err = d.qpos[:7] - key.qpos[:7]
    for k, aid in enumerate(acts):
        home_ctrl[aid] += err[k]
d.qpos[:] = key.qpos
d.qvel[:] = 0
d.ctrl[:] = home_ctrl
mujoco.mj_forward(m, d)

gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "sensor_pad")
wid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "contact_wall")

def step_for(press, sx=0, sy=0, steps=1500):
    depth = press / 5000.0
    b = -depth / 0.354
    d.ctrl[:] = home_ctrl
    d.ctrl[acts[1]] += 0.777 * b
    d.ctrl[acts[3]] += b
    d.ctrl[acts[0]] += -(np.sin(np.radians(sy)) * 0.0015) / 0.473
    d.ctrl[acts[5]] += (np.sin(np.radians(sx)) * 0.0015) / 0.183
    for _ in range(steps):
        mujoco.mj_step(m, d)

def forces():
    fw = np.zeros(3)
    for i in range(d.ncon):
        c = d.contact[i]
        if {c.geom1, c.geom2} == {gid, wid}:
            f = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f)
            fw += c.frame.reshape(3, 3).T @ f[:3]
    R = d.geom_xmat[gid].reshape(3, 3)
    fl = R.T @ fw
    return fl[0], fl[1], -fl[2], d.ncon

for press in [0, 3, 5, 8, 10]:
    step_for(press)
    fx, fy, fz, n = forces()
    print(f"press={press:5.1f}  contacts={n}  fx={fx:+.3f} fy={fy:+.3f} fz={fz:+.3f}")

step_for(6, sx=30, sy=0)
print("press=6 sx=30:", ["%+.3f" % v for v in forces()[:3]])
step_for(6, sy=30)
print("press=6 sy=30:", ["%+.3f" % v for v in forces()[:3]])
