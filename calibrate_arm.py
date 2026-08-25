import mujoco
import numpy as np

MODEL = r"C:\Users\cuizi\AppData\Roaming\TRAE SOLO CN\ModularData\ai-agent\work-mode-projects\6a8bb58b84243539c48c3256\mujoco_menagerie\kinova_gen3\gen3_sensor.xml"

m = mujoco.MjModel.from_xml_path(MODEL)
d = mujoco.MjData(m)
home = m.key("home").qpos.copy()
d.qpos[:] = home
mujoco.mj_forward(m, d)

sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "sensor_site")
gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "sensor_pad")
p0 = d.site_xpos[sid].copy()
xmat = d.geom_xmat[gid].reshape(3, 3)
z_axis = xmat[:, 2]

print(f"sensor world pos: x={p0[0]:.4f} y={p0[1]:.4f} z={p0[2]:.4f}")
print(f"sensor z-axis (normal dir): {np.round(z_axis, 3)}")
print()
print("joint  |  site dx      dy      dz  (for +0.05 rad)")
for j in range(7):
    q = home.copy()
    q[j] += 0.05
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    dp = d.site_xpos[sid] - p0
    print(f"j{j+1}    | {dp[0]:+.5f} {dp[1]:+.5f} {dp[2]:+.5f}")
