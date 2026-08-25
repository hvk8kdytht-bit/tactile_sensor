import mujoco
import numpy as np
import os

BASE = r"C:\Users\cuizi\AppData\Roaming\TRAE SOLO CN\ModularData\ai-agent\work-mode-projects\6a8bb58b84243539c48c3256"
MODEL = os.path.join(BASE, "mujoco_menagerie", "kinova_gen3", "gen3_2f85_sensor.xml")

m = mujoco.MjModel.from_xml_path(MODEL)
d = mujoco.MjData(m)
print(f"load OK: nq={m.nq}, nu={m.nu}, nbody={m.nbody}, nmesh={m.nmesh}")

d.qpos[:] = m.key("home").qpos
mujoco.mj_forward(m, d)

gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "sensor_pad")
sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "sensor_site")
sp = d.site_xpos[sid].copy()
R = d.geom_xmat[gid].reshape(3, 3)
sz = m.geom_size[gid]
print(f"sensor size: {sz[0]*2000:.1f} x {sz[1]*2000:.1f} x {sz[2]*2000:.1f} mm")
print(f"sensor site world: {np.round(sp, 4)}")
print(f"sensor face normal (world): {np.round(R[:,1], 3)}")

fa = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
print(f"actuators: joint_1..7 + fingers_actuator(id={fa}, ctrl 0-255, 0=open)")

# grasp test: freeze arm servos, close gripper on a test cube via mocap-free approach:
# simpler: set arm ctrl to home, close fingers fully (ctrl 255) and see pad-sensor self contact force is zero;
# real object grasp validated visually by contact between left_pad and sensor when closing
d.ctrl[:] = m.key("home").ctrl
d.ctrl[fa] = 255
for _ in range(3000):
    mujoco.mj_step(m, d)

n_sensor_contacts = 0
for i in range(d.ncon):
    c = d.contact[i]
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
             mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)}
    if "sensor_pad" in names:
        n_sensor_contacts += 1
        print(f"  sensor contact with: {names - {'sensor_pad'}}, dist={c.dist:.5f}")
print(f"fingers closed (ctrl=255), sensor contacts: {n_sensor_contacts}")
print(f"driver joint pos: right={d.qpos[7]:.3f} left={d.qpos[11]:.3f} (range 0-0.8)")
for _ in range(1000):
    mujoco.mj_step(m, d)
print(f"sim 4000 steps total OK, stable: {np.all(np.isfinite(d.qacc))}")
