import mujoco
import numpy as np

MODEL = r"C:\Users\cuizi\AppData\Roaming\TRAE SOLO CN\ModularData\ai-agent\work-mode-projects\6a8bb58b84243539c48c3256\mujoco_menagerie\kinova_gen3\gen3_2f85_sensor_demo.xml"
m = mujoco.MjModel.from_xml_path(MODEL)
d = mujoco.MjData(m)
m.opt.gravity[:] = 0

key = m.key("home")
d.qpos[:] = key.qpos
d.ctrl[:] = key.ctrl
mujoco.mj_forward(m, d)
acts = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint_{i}") for i in range(1, 8)]
hc = key.ctrl.copy()
bias = d.qfrc_bias.copy()
for aid in acts:
    jid = m.actuator_trnid[aid][0]
    dof = m.jnt_dofadr[jid]
    hc[aid] += bias[dof] / m.actuator_gainprm[aid][0]
for _ in range(2):
    d.qpos[:] = key.qpos
    d.qvel[:] = 0
    d.ctrl[:] = hc
    for _ in range(1500):
        mujoco.mj_step(m, d)
    err = d.qpos[:7] - key.qpos[:7]
    for k, aid in enumerate(acts):
        hc[aid] += err[k]
print("calibrated arm err:", np.abs(d.qpos[:7] - key.qpos[:7]).max())

gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "sensor_pad")
bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "grasp_box_geom")
mid = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "grasp_box")]
pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pinch")
fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
print("nq", m.nq, "nu", m.nu, "fingers act id", fid)

d.qpos[:] = key.qpos
d.qvel[:] = 0
d.ctrl[:] = hc
mujoco.mj_forward(m, d)
sp = d.geom_xpos[gid]
R = d.geom_xmat[gid].reshape(3, 3)
print("sensor xpos", sp)
print("sensor R (local axes in world):\n", np.round(R, 3))
print("pinch site", d.site_xpos[pid])
print("sensor local y (outward?):", np.round(R @ np.array([0, 1, 0]), 3), " -y:", np.round(R @ np.array([0, -1, 0]), 3))


def read_sensor():
    f = np.zeros(3)
    n = 0
    for i in range(d.ncon):
        c = d.contact[i]
        if gid in (c.geom1, c.geom2):
            f6 = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f6)
            f += c.frame.reshape(3, 3).T @ f6[:3]
            n += 1
    fl = R_current.T @ f
    return n, fl


def settle(steps=1000):
    for _ in range(steps):
        mujoco.mj_step(m, d)


R_current = R.copy()

print("\n--- A: close gripper on box at pinch ---")
d.qpos[:] = key.qpos
d.qvel[:] = 0
d.ctrl[:] = hc
d.ctrl[fid] = 255
mujoco.mj_forward(m, d)
d.mocap_pos[mid] = d.site_xpos[pid] + np.array([0, 0, 0.003])
settle(1500)
for cmd in [255, 200, 150, 100, 60, 30, 0]:
    d.ctrl[fid] = cmd
    settle(800)
    R_current = d.geom_xmat[gid].reshape(3, 3)
    n, fl = read_sensor()
    print(f"cmd={cmd:3d} ncon_sensor={n} local=({fl[0]:+.3f},{fl[1]:+.3f},{fl[2]:+.3f})")

print("\n--- B: mocap box pressed onto sensor face along local axes ---")
d.qpos[:] = key.qpos
d.qvel[:] = 0
d.ctrl[:] = hc
d.ctrl[fid] = 255
mujoco.mj_forward(m, d)
R_current = d.geom_xmat[gid].reshape(3, 3)
sp = d.geom_xpos[gid]
box_half = 0.015
for axis, name in [([0, -1, 0], "-y"), ([0, 1, 0], "+y"), ([0, 0, 1], "+z"), ([0, 0, -1], "-z"), ([1, 0, 0], "+x")]:
    ax = np.array(axis, dtype=float)
    surf = sp + R_current @ (ax * 0.0026)
    target = surf + R_current @ (ax * box_half)
    d.mocap_pos[mid] = target - R_current @ (ax * 0.0008)
    d.mocap_quat[mid] = np.array([1, 0, 0, 0])
    settle(800)
    R_current = d.geom_xmat[gid].reshape(3, 3)
    n, fl = read_sensor()
    print(f"press along local {name}: ncon={n} local=({fl[0]:+.3f},{fl[1]:+.3f},{fl[2]:+.3f})")
