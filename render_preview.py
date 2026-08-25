import mujoco
import numpy as np
import os

BASE = r"C:\Users\cuizi\AppData\Roaming\TRAE SOLO CN\ModularData\ai-agent\work-mode-projects\6a8bb58b84243539c48c3256"
MODEL = os.path.join(BASE, "mujoco_menagerie", "kinova_gen3", "gen3_sensor.xml")

def save_img(arr, name):
    path = os.path.join(BASE, name)
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
    except ImportError:
        import matplotlib
        matplotlib.image.imsave(path, arr)
    print("saved:", name)

m = mujoco.MjModel.from_xml_path(MODEL)
d = mujoco.MjData(m)
d.qpos[:] = m.key("home").qpos
mujoco.mj_forward(m, d)

r = mujoco.Renderer(m, 900, 1200)
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()

cam.lookat[:] = [0.25, 0, 0.30]
cam.distance = 1.5
cam.azimuth = 135
cam.elevation = -12
r.update_scene(d, cam, opt)
save_img(r.render(), "preview_arm.png")

sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "sensor_site")
sp = d.site_xpos[sid].copy()
print("sensor world pos:", np.round(sp, 4))

cam.lookat[:] = sp
cam.distance = 0.12
cam.azimuth = 180
cam.elevation = -8
r.update_scene(d, cam, opt)
save_img(r.render(), "preview_sensor_front.png")

cam.azimuth = 225
cam.elevation = -25
cam.distance = 0.10
r.update_scene(d, cam, opt)
save_img(r.render(), "preview_sensor_iso.png")

xml = open(MODEL, encoding="utf-8").read()
wall = ('<geom name="contact_wall" type="box" pos="0.4580 0.0014 0.4337" size="0.01 0.08 0.08" '
        'rgba="0.55 0.55 0.60 1" contype="0" conaffinity="0"/>\n    <body name="base_link">')
xml = xml.replace('<body name="base_link">', wall, 1)
pair = ('<contact><pair geom1="contact_wall" geom2="sensor_pad" condim="4" '
        'solref="0.005 1" solimp="0.95 0.99 0.0005" friction="1.0 0.3 0.1 0.001 0.001"/></contact>\n\n  <actuator>')
xml = xml.replace('  <actuator>', pair, 1)
demo_path = os.path.join(BASE, "mujoco_menagerie", "kinova_gen3", "gen3_sensor_demo.xml")
open(demo_path, "w", encoding="utf-8").write(xml)

m2 = mujoco.MjModel.from_xml_path(demo_path)
d2 = mujoco.MjData(m2)
key = m2.key("home")
acts = [mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint_{i}") for i in range(1, 8)]
d2.qpos[:] = key.qpos
d2.ctrl[:] = key.ctrl
mujoco.mj_forward(m2, d2)
bias = d2.qfrc_bias.copy()
home_ctrl = key.ctrl.copy()
for aid in acts:
    jid = m2.actuator_trnid[aid][0]
    kp = m2.actuator_gainprm[aid][0]
    home_ctrl[aid] += bias[m2.jnt_dofadr[jid]] / kp
for _ in range(2):
    d2.qpos[:] = key.qpos
    d2.qvel[:] = 0
    d2.ctrl[:] = home_ctrl
    for _ in range(1500):
        mujoco.mj_step(m2, d2)
    err = d2.qpos[:7] - key.qpos[:7]
    for k, aid in enumerate(acts):
        home_ctrl[aid] += err[k]

d2.ctrl[:] = home_ctrl
depth = 5.0 / 5000.0
b = -depth / 0.354
d2.ctrl[acts[1]] += 0.777 * b
d2.ctrl[acts[3]] += b
for _ in range(1500):
    mujoco.mj_step(m2, d2)

gid = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_GEOM, "sensor_pad")
wid = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_GEOM, "contact_wall")
fw = np.zeros(3)
for i in range(d2.ncon):
    c = d2.contact[i]
    if {c.geom1, c.geom2} == {gid, wid}:
        f = np.zeros(6)
        mujoco.mj_contactForce(m2, d2, i, f)
        fw += c.frame.reshape(3, 3).T @ f[:3]
R = d2.geom_xmat[gid].reshape(3, 3)
fl = R.T @ fw
print(f"press demo force: fx={fl[0]:+.3f} fy={fl[1]:+.3f} fz={-fl[2]:+.3f} N")

r2 = mujoco.Renderer(m2, 900, 1200)
opt2 = mujoco.MjvOption()
opt2.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
opt2.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
cam2 = mujoco.MjvCamera()
cam2.lookat[:] = [0.463, 0.001, 0.434]
cam2.distance = 0.16
cam2.azimuth = 200
cam2.elevation = -18
r2.update_scene(d2, cam2, opt2)
save_img(r2.render(), "preview_press_contact.png")

r.close()
r2.close()
print("DONE")
