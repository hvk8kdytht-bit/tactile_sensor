"""全场景穿模检测: 对所有外部物体(红色目标/木箱/触控盒) + 机械臂自碰撞
逐个模式验证双侧指垫不穿模, 以及臂连杆不自穿。"""
import numpy as np
import mujoco
from kinova_sensor_sim import SimSensor

s = SimSensor()
m, d = s.model, s.data
np.set_printoptions(precision=4, suppress=True)

PRESS_G = s.press_geom_id
GRASP_G = s.grasp_geom_id
TOUCH_G = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "touch_finger_geom")
LEFT_PADS = s.left_pad_geoms
RIGHT_PADS = s.right_pad_geoms
ALL_PADS = LEFT_PADS | RIGHT_PADS

# 臂连杆 geom: 排除手/指部分, 只看 base..forearm 的大连杆
ARM_BODIES = set()
for name in ("base_link", "shoulder_link", "half_arm_1_link", "half_arm_2_link",
             "forearm_link", "spherical_wrist_1_link", "spherical_wrist_2_link",
             "bracelet_link"):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid >= 0:
        ARM_BODIES.add(bid)


def max_penetration(target_geom, pad_geoms):
    """指定目标geom与指垫之间的最大穿透深度(mm), 0表示无接触/无穿透"""
    worst = 0.0
    for i in range(d.ncon):
        con = d.contact[i]
        ids = {int(con.geom1), int(con.geom2)}
        if target_geom in ids and ids & pad_geoms:
            worst = max(worst, -float(con.dist))
    return worst * 1000


def pad_inside_target_volume(target_geom, pad_geom, half_size):
    """pad_geom中心进入target_geom体积内部的深度(mm), 穿模时>0"""
    tg = target_geom
    pg = pad_geom
    tbid = m.geom_bodyid[tg]
    R = d.xmat[tbid].reshape(3, 3).T
    loc = R @ (d.geom_xpos[pg] - d.geom_xpos[tg])
    return float((half_size - np.abs(loc)).min()) * 1000


def arm_self_penetration():
    """机械臂连杆之间的最大自穿透(mm), 相邻连杆已被碰撞过滤排除, 正常应为0"""
    worst = 0.0
    arm_geoms = set()
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in ARM_BODIES:
            arm_geoms.add(g)
    for i in range(d.ncon):
        con = d.contact[i]
        ids = {int(con.geom1), int(con.geom2)}
        if ids <= arm_geoms:  # 两个geom都属于臂连杆
            worst = max(worst, -float(con.dist))
    return worst * 1000


def step(n):
    for _ in range(n):
        s.update_interaction()
        mujoco.mj_step(m, d)
        f = np.array(s.get_contact_forces())
        s.raw_fx, s.raw_fy, s.raw_fz = f
        s.fz_filt += 0.2 * (f[2] - s.fz_filt)


def check(ok, msg):
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")


print("=" * 65)
print("  全场景穿模检测 (双侧指垫 + 臂自碰撞)")
print("=" * 65)

# ========== 1. arm模式: 红色按压目标 ==========
print("\n===== 1. ARM模式: 红色按压目标 =====")
s.mode = "arm"
s.reset_scene()
s.arm_force = 0.0
max_press_left_pen = 0.0
max_press_right_pen = 0.0
max_press_left_vol = 0.0
max_press_right_vol = 0.0
press_half = np.array([0.014, 0.012, 0.014])
for i in range(2500):
    step(1)
    if i == 500:
        s.arm_force = 5.0
    if i == 1800:
        s.arm_force = 0.0
    lp = max_penetration(PRESS_G, LEFT_PADS)
    rp = max_penetration(PRESS_G, RIGHT_PADS)
    max_press_left_pen = max(max_press_left_pen, lp)
    max_press_right_pen = max(max_press_right_pen, rp)
    # 体积侵入只看右垫主geom (左垫有传感器贴在目标面, 正常贴紧不算穿模)
    rv = pad_inside_target_volume(PRESS_G, s.right_pad_main, press_half)
    max_press_right_vol = max(max_press_right_vol, rv)

check(max_press_right_vol < 0.5,
      f"右垫不进入目标体积 (最大侵入={max_press_right_vol:.2f}mm, 穿模>5mm)")
check(max_press_right_pen < 3.0,
      f"右垫-目标接触穿透<3mm (最大={max_press_right_pen:.2f}mm)")
check(max_press_left_pen < 3.0,
      f"左垫-目标接触穿透<3mm (最大={max_press_left_pen:.2f}mm)")
check(abs(s.raw_fz) < 0.5 and s.tilt_deg() < 3,
      f"撤力归零 rawFz={s.raw_fz:+.2f}N tilt={s.tilt_deg():.2f}deg")

# ========== 2. grasp模式: 木箱薄板 ==========
print("\n===== 2. GRASP模式: 木箱薄板 =====")
s.mode = "grasp"
s.reset_scene()
s.grasp_pct = 0.0
max_grasp_left_pen = 0.0
max_grasp_right_pen = 0.0
grasp_half = np.array([0.012, 0.0075, 0.036])
max_grasp_left_vol = 0.0
max_grasp_right_vol = 0.0
for i in range(2000):
    step(1)
    if i == 200:
        s.grasp_pct = 100.0
    if i == 1500:
        s.grasp_pct = 0.0
    lp = max_penetration(GRASP_G, LEFT_PADS)
    rp = max_penetration(GRASP_G, RIGHT_PADS)
    max_grasp_left_pen = max(max_grasp_left_pen, lp)
    max_grasp_right_pen = max(max_grasp_right_pen, rp)
    # 体积侵入: 两侧都看
    for pg in LEFT_PADS:
        v = pad_inside_target_volume(GRASP_G, pg, grasp_half)
        max_grasp_left_vol = max(max_grasp_left_vol, v)
    for pg in RIGHT_PADS:
        v = pad_inside_target_volume(GRASP_G, pg, grasp_half)
        max_grasp_right_vol = max(max_grasp_right_vol, v)

check(max_grasp_right_vol < 0.5,
      f"右垫不进入木箱体积 (最大侵入={max_grasp_right_vol:.2f}mm)")
check(max_grasp_left_vol < 0.5,
      f"左垫不进入木箱体积 (最大侵入={max_grasp_left_vol:.2f}mm)")
check(max_grasp_right_pen < 3.0,
      f"右垫-木箱接触穿透<3mm (最大={max_grasp_right_pen:.2f}mm)")
check(max_grasp_left_pen < 3.0,
      f"左垫-木箱接触穿透<3mm (最大={max_grasp_left_pen:.2f}mm)")
check(abs(s.raw_fz) < 0.5 and s.tilt_deg() < 3,
      f"松开后归零 rawFz={s.raw_fz:+.2f}N tilt={s.tilt_deg():.2f}deg")

# ========== 3. touch模式: 触控盒 ==========
print("\n===== 3. TOUCH模式: 触控盒 =====")
s.mode = "touch"
s.reset_scene()
s.touch_force = 0.0
max_touch_left_pen = 0.0
max_touch_right_pen = 0.0
touch_half = np.array([0.012, 0.012, 0.012])
max_touch_left_vol = 0.0
max_touch_right_vol = 0.0
for i in range(2000):
    step(1)
    if i == 200:
        s.touch_force = 6.0
    if i == 1500:
        s.touch_force = 0.0
    lp = max_penetration(TOUCH_G, LEFT_PADS)
    rp = max_penetration(TOUCH_G, RIGHT_PADS)
    max_touch_left_pen = max(max_touch_left_pen, lp)
    max_touch_right_pen = max(max_touch_right_pen, rp)
    for pg in LEFT_PADS:
        v = pad_inside_target_volume(TOUCH_G, pg, touch_half)
        max_touch_left_vol = max(max_touch_left_vol, v)
    for pg in RIGHT_PADS:
        v = pad_inside_target_volume(TOUCH_G, pg, touch_half)
        max_touch_right_vol = max(max_touch_right_vol, v)

check(max_touch_right_vol < 0.5,
      f"右垫不进入触控盒体积 (最大侵入={max_touch_right_vol:.2f}mm)")
check(max_touch_left_vol < 0.5,
      f"左垫不进入触控盒体积 (最大侵入={max_touch_left_vol:.2f}mm)")
check(max_touch_right_pen < 3.0,
      f"右垫-触控盒接触穿透<3mm (最大={max_touch_right_pen:.2f}mm)")
check(max_touch_left_pen < 3.0,
      f"左垫-触控盒接触穿透<3mm (最大={max_touch_left_pen:.2f}mm)")
check(abs(s.raw_fz) < 0.5 and s.tilt_deg() < 3,
      f"释放后归零 rawFz={s.raw_fz:+.2f}N tilt={s.tilt_deg():.2f}deg")

# ========== 4. 机械臂自碰撞 ==========
print("\n===== 4. 机械臂自碰撞 (三模式轮换后) =====")
s.mode = "arm"
s.reset_scene()
s.arm_force = 8.0
step(1000)
s.arm_force = 0.0
step(500)
s.mode = "grasp"
s.reset_scene()
s.grasp_pct = 100
step(500)
s.grasp_pct = 0
step(500)
s.mode = "touch"
s.reset_scene()
s.touch_force = 6.0
step(500)
s.touch_force = 0.0
step(500)

self_pen = arm_self_penetration()
check(self_pen < 0.1,
      f"臂连杆无自碰撞穿透 (最大={self_pen:.3f}mm, 相邻连杆已过滤)")
check(abs(s.tilt_deg()) < 0.1,
      f"最终水平保持 tilt={s.tilt_deg():.3f}deg")

print("\n" + "=" * 65)
print("  检测完成")
print("=" * 65)
