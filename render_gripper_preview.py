import mujoco
import numpy as np
import os

BASE = r"C:\Users\cuizi\AppData\Roaming\TRAE SOLO CN\ModularData\ai-agent\work-mode-projects\6a8bb58b84243539c48c3256"
SRC = os.path.join(BASE, "mujoco_menagerie", "kinova_gen3", "gen3_2f85_sensor.xml")
DEMO = os.path.join(BASE, "mujoco_menagerie", "kinova_gen3", "gen3_2f85_sensor_demo.xml")

xml = open(SRC, encoding="utf-8").read()
xml = xml.replace('<option integrator="implicitfast" cone="elliptic" impratio="10"/>',
                  '<option integrator="implicitfast" cone="elliptic" impratio="10" gravity="0 0 0"/>')
obj = '''<body name="grasp_box" pos="0 0 0" mocap="true">
      <geom name="grasp_box_geom" type="box" size="0.015 0.015 0.015" rgba="0.85 0.6 0.15 1" condim="4" friction="0.8 0.2 0.01"/>
    </body>'''
xml = xml.replace('</worldbody>', obj + '\n  </worldbody>')
open(DEMO, "w", encoding="utf-8").write(xml)

m = mujoco.MjModel.from_xml_path(DEMO)
d = mujoco.MjData(m)
fa = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "sensor_pad")
bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "grasp_box")

d.qpos[:] = m.key("home").qpos
d.ctrl[:] = m.key("home").ctrl
mujoco.mj_forward(m, d)

pinch = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pinch")
box_pos = d.site_xpos[pinch].copy()
mid = m.body_mocapid[bid]
d.mocap_pos[mid] = box_pos
print(f"box at pinch: {np.round(box_pos, 4)}")

r = mujoco.Renderer(m, 900, 1200)
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()

def shot(name, lookat, dist, az, el):
    cam.lookat[:] = lookat
    cam.distance = dist
    cam.azimuth = az
    cam.elevation = el
    r.update_scene(d, cam, opt)
    arr = r.render()
    from PIL import Image
    Image.fromarray(arr).save(os.path.join(BASE, name))
    print("saved:", name)

shot("gripper_full.png", [0.35, 0, 0.35], 1.6, 135, -12)

for _ in range(1000):
    mujoco.mj_step(m, d)
d.ctrl[fa] = 255
for _ in range(2500):
    mujoco.mj_step(m, d)

fw = np.zeros(3)
n_con = 0
for i in range(d.ncon):
    c = d.contact[i]
    g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
    g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
    if "sensor_pad" in (g1, g2):
        f = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f)
        fw += c.frame.reshape(3, 3).T @ f[:3]
        n_con += 1
R = d.geom_xmat[gid].reshape(3, 3)
fl = R.T @ fw
print(f"GRASP: {n_con} sensor contacts, world force {np.round(fw,3)}, "
      f"sensor-frame: fx={fl[0]:+.3f} fy={-fl[1]:+.3f} fz={fl[2]:+.3f} N")

opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
shot("gripper_grasp.png", [box_pos[0], 0, box_pos[2]], 0.35, 130, -15)

d.ctrl[fa] = 0
for _ in range(2000):
    mujoco.mj_step(m, d)
shot("gripper_open.png", [box_pos[0], 0, box_pos[2]], 0.22, 100, -10)

shot("sensor_face.png", d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "sensor_site")], 0.08, 90, -5)

r.close()
print("DONE")
