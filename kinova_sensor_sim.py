"""
Kinova Gen3 + Robotiq 2F-85 + 触觉传感器 仿真
模型: mujoco_menagerie/kinova_gen3/gen3_2f85_sensor_scene.xml
三种交互模式:
  touch  - 网页触控板拖拽,模拟手指按压传感器表面(法向力+滑动剪切力)
  grasp  - 夹爪闭合抓取方块,右指垫将物体压向左指垫上的传感器
  arm    - 机械臂IK运动,主动伸过去用传感器按压场景中的红色目标物
力提取: sensor_pad geom 上所有接触 -> 传感器局部坐标系
  Fz(法向) = -local_y, Fx = local_z, Fy = local_x   (实测标定)
臂动按压: 力闭环(压深按指令力-实测力积分), 撤力后退到目标面外5mm真间隙
信号特征层 SignalEmulator: 在物理真值力之上叠加真实传感器信号特性
  串扰矩阵 -> 迟滞(回隙) -> 蠕变 -> 带宽低通 -> 零漂OU -> 白噪声 -> 死区 -> ADC量化限幅 -> 采样保持
  参数在 signal_config.json, 可用 fit_signal_params.py 从真实录制数据自动拟合后注入
"""
import mujoco
import numpy as np
import time
import threading
import json
import http.server
import socketserver
import os
from contextlib import nullcontext

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mujoco_menagerie", "kinova_gen3", "gen3_2f85_sensor_scene.xml")
PORT = 8771

SENSOR_HALF_Y = 0.0026
TOUCH_BOX_HALF = 0.012
GRASP_BOX_HALF_Y = 0.0075   # 夹取薄板沿垫法向的半厚度(m)
GRASP_TENDON_MAX = 15.0     # 腱力上限(N, 内存中放开, 原模型±5N夹取力不足)
GRASP_FAST_RATE = 150.0     # 闭合快速段 ctrl速率/s (≈50mm/s)
GRASP_SLOW_RATE = 20.0      # 距薄板6mm内蠕速 ctrl/s (≈7mm/s, 接触链~1MN/m冲击会弹出薄板)
GRASP_FZ_CAP = 35.0         # 夹持力上限(N): 传感器量程±50N; 全闭合时连杆近奇异,
                            # 腱力15N经放大可达125N且持续爬升, 超限即冻结闭合
ARM_STANDBY = 0.005         # 臂动模式撤力后指垫离目标面的悬停间隙(m)
ARM_KI = 0.0006            # 臂动力闭环积分增益 m/(N·s): 压深随(指令力-实测力)积分
ARM_MAX_DEPTH = 0.006      # 臂动力闭环压深上限 (m)
TOUCH_KI = 0.0006          # 力闭环积分增益 m/(N·s): 压深随(指令力-实测力)积分
TOUCH_MAX_DEPTH = 0.00005  # 力闭环压深上限 (m), 接触链刚度~1MN/m时对应~50N
PARK_POS = np.array([0.75, 0.3, 0.55])
BOX_PARK = np.array([0.45, 0.35, 0.15])
ARM_RATE = 0.0015
ARM_MAX_FORCE = 10.0

DEFAULT_SIGNAL_CONFIG = {
    "enabled": True,
    "sample_rate": 200.0,      # 传感器输出采样率 Hz
    "adc_bits": 12,            # ADC 位数
    "full_scale": 50.0,        # 每轴量程 ±FS (N)
    "noise_sigma": 0.02,       # 白噪声 RMS (N)
    "noise_relative": 0.002,   # 与载荷相关的相对噪声
    "drift_sigma": 0.05,       # 零漂 OU 稳态 σ (N)
    "drift_tau": 60.0,         # 零漂时间常数 (s)
    "bandwidth": 50.0,         # 一阶低通带宽 (Hz)
    "crosstalk": [[1.0, 0.02, -0.01],
                  [-0.015, 1.0, 0.02],
                  [0.01, -0.02, 1.0]],
    "hysteresis_frac": 0.02,   # 迟滞回隙 (相对近期载荷包络)
    "hysteresis_tau": 1.0,     # 载荷包络记忆衰减 (s), 兼作过载恢复时间
    "creep_gain": 0.02,        # 蠕变渐近增益 (恒载读数增量比例)
    "creep_tau": 8.0,          # 蠕变时间常数 (s)
    "deadband": 0.004,         # 死区 (N)
}


class SignalEmulator:
    """真实传感器信号特征层: 物理真值力 -> 带真实特性的传感器读数"""

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.reset()

    def reset(self):
        self.lpf = np.zeros(3)
        self.creep = np.zeros(3)
        self.hyst = np.zeros(3)
        self.env = np.zeros(3)
        self.drift = np.zeros(3)
        self.t_acc = 0.0
        self.held = np.zeros(3)

    def process(self, f_true, dt):
        c = self.cfg
        f_true = np.asarray(f_true, dtype=float).copy()
        if not c.get("enabled", True) or dt <= 0:
            return f_true

        # 1) 轴间串扰
        f = np.asarray(c.get("crosstalk"), dtype=float) @ f_true

        # 2) 迟滞 (回隙模型, 回隙宽度按近期载荷包络记忆, 卸载后残余缓慢消退)
        tau_env = max(c.get("hysteresis_tau", 5.0), 1e-3)
        self.env = np.maximum(np.abs(f), self.env * np.exp(-dt / tau_env))
        h = c.get("hysteresis_frac", 0.0) * self.env
        y = self.hyst.copy()
        for i in range(3):
            if f[i] >= y[i] + h[i]:
                y[i] = f[i] - h[i]
            elif f[i] <= y[i] - h[i]:
                y[i] = f[i] + h[i]
        self.hyst = y
        f = y

        # 3) 蠕变: 一阶趋近 creep_gain * 载荷
        tau = max(c.get("creep_tau", 1.0), 1e-3)
        self.creep += (c.get("creep_gain", 0.0) * f - self.creep) * min(1.0, dt / tau)
        f = f + self.creep

        # 4) 带宽一阶低通
        fc = max(c.get("bandwidth", 1.0), 0.1)
        alpha = min(1.0, dt * 2.0 * np.pi * fc)
        self.lpf += (f - self.lpf) * alpha
        f = self.lpf.copy()

        # 5) 零漂 (Ornstein-Uhlenbeck)
        tau_d = max(c.get("drift_tau", 60.0), 1e-2)
        sig_d = c.get("drift_sigma", 0.0)
        self.drift += -self.drift * min(1.0, dt / tau_d) \
            + np.random.randn(3) * sig_d * np.sqrt(2.0 * dt / tau_d)
        f = f + self.drift

        # 6) 白噪声 + 相对噪声
        n_abs = c.get("noise_sigma", 0.0)
        n_rel = c.get("noise_relative", 0.0)
        f = f + np.random.randn(3) * np.sqrt(n_abs ** 2 + (n_rel * np.abs(f)) ** 2)

        # 7) 死区
        db = c.get("deadband", 0.0)
        f[np.abs(f) < db] = 0.0

        # 8) ADC 量化 + 限幅
        bits = int(c.get("adc_bits", 0))
        fs_n = max(c.get("full_scale", 1.0), 1e-6)
        if bits > 0:
            lsb = 2.0 * fs_n / (2 ** bits)
            f = np.round(f / lsb) * lsb
        f = np.clip(f, -fs_n, fs_n)

        # 9) 采样保持 (零阶保持)
        self.t_acc += dt
        period = 1.0 / max(c.get("sample_rate", 1e9), 1.0)
        if self.t_acc >= period:
            self.t_acc = 0.0
            self.held = f.copy()
        return self.held.copy()


class SimSensor:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.model.opt.gravity[:] = 0

        # 交付模型 base_link/shoulder_link 凸包在 home 位姿互相穿透 12mm,
        # 产生 ~48kN 接触把 joint_1 抱死 -> 关闭 base 碰撞 geom 解除
        base_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == base_bid and \
                    (self.model.geom_contype[g] or self.model.geom_conaffinity[g]):
                self.model.geom_contype[g] = 0
                self.model.geom_conaffinity[g] = 0

        # 相邻连杆互碰排除 + 场景物体碰撞过滤(视窗拖拽防锁死), 见方法注释
        self._setup_collision_filters()

        # joint_1/3/5/7补关节限位: 无限位时视窗滑条只有±1 rad,
        # joint_7水平基准1.5708拖不回去(范围调不到大于1), 见方法注释
        self._setup_joint_limits()

        # 2F-85腱力上限±5N经连杆放大后夹取力不足, 内存中放宽(不改XML)
        fa = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
        if fa >= 0:
            self.model.actuator_forcerange[fa][0] = -GRASP_TENDON_MAX
            self.model.actuator_forcerange[fa][1] = GRASP_TENDON_MAX

        self.fx = 0.0
        self.fy = 0.0
        self.fz = 0.0
        self.peak_force = 0.0
        self.running = False
        self.start_time = 0.0
        self.sample_count = 0
        self.fps = 0

        self.tare_offset = np.zeros(3)
        self.raw_fx = 0.0
        self.raw_fy = 0.0
        self.raw_fz = 0.0

        self.mode = "touch"
        self.pending_mode = None
        self.reset_request = False
        self.touch_force = 0.0
        self.touch_sx = 0.0
        self.touch_sy = 0.0
        self.touch_depth = 0.0
        self.fz_filt = 0.0
        self.tilt_now = 0.0
        self.tilt_bad_t = 0.0
        self.grasp_pct = 0.0
        self.grasp_ctrl = 0.0
        self.auto_mode = False
        self.auto_t = 0.0
        self.touch_pos = None
        self.lock = threading.Lock()

        self.sensor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sensor_pad")
        self.sensor_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tactile_sensor")
        self.press_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "press_target_geom")
        self.touch_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "touch_finger")
        self.touch_mocap = self.model.body_mocapid[self.touch_body]
        self.grasp_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "grasp_box")
        self.grasp_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "grasp_box_joint")
        self.grasp_qadr = self.model.jnt_qposadr[self.grasp_jnt]
        self.grasp_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "grasp_box_geom")
        self.left_pad_geoms = {g for g in (
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in ("left_pad1", "left_pad2")) if g >= 0}
        self.right_pad_geoms = {g for g in (
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in ("right_pad1", "right_pad2")) if g >= 0}
        self.right_pad_main = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_pad1")
        self.finger_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
        self.arm_acts = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint_{i}")
                         for i in range(1, 8)]

        self._load_signal()
        self._calibrate_home()
        self._init_arm_ik()
        self.reset_scene()

    def _setup_joint_limits(self):
        """视窗关节滑条范围修复(内存中改, 不动XML):
        XML里joint_1/3/5/7未定义range(无限位关节), MuJoCo视窗对无限位关节的
        拖拽滑条只会给±1 rad, 而joint_7水平基准位是1.5708 rad -- 拖过之后
        永远拖不回水平, 即"范围根本调不到大于1"。真机Gen3的roll轴
        (J1/J3/J5/J7)限位±175°=±3.0543 rad, 但home姿态joint_3=π略超,
        统一取±3.2 rad留余量; 对应执行器ctrlrange同步放开, ctrl面板
        同样不再被钳在±1。
        """
        LIM = 3.2
        for name in ("joint_1", "joint_3", "joint_5", "joint_7"):
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j >= 0:
                self.model.jnt_range[j] = [-LIM, LIM]
                self.model.jnt_limited[j] = 1
            a = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if a >= 0:
                self.model.actuator_ctrlrange[a] = [-LIM, LIM]
                self.model.actuator_ctrllimited[a] = 1

    def _setup_collision_filters(self):
        """视窗拖拽防锁死的碰撞过滤(全部内存中改, 不动XML):
        1) 相邻连杆(parent-child)互碰排除: 真实机械臂相邻连杆以转轴相连本不互碰,
           凸包粗糙互相穿透时接触求解器产生数千N刚性约束, 远超伺服力矩上限
           (大joint±105/小joint±52 N·m), 视窗拖拽关节稍过位姿即永久卡死。
           用逐body独立碰撞位实现: contype=bit(b), conaffinity=bit0|其余非相邻body位,
           非相邻连杆仍正常互碰(真实自碰撞检测保留)。
        2) 触控mocap盒只与传感器垫/左指垫碰撞: 用户在3D视窗里把盒子拖进臂内
           也不会砸翻机械臂(mocap体接触等效静态几何, 求解器全力弹射臂)。
        3) 红色按压目标只与传感器垫/左指垫碰撞: 臂动模式按压链路不变,
           但臂本体被拖进目标时不再被静态几何卡死。
        """
        m = self.model
        BIT_TOUCH = 1 << 28   # 触控盒 <-> 传感器侧
        BIT_TARGET = 1 << 27  # 按压目标 <-> 传感器侧

        # 1) 相邻连杆排除: 所有 parent 非 world 的 body 参与位分配
        bodies = [b for b in range(1, m.nbody) if m.body_parentid[b] != 0]
        assert len(bodies) < 26, "body数超出可用碰撞位"
        bit = {b: 1 << (1 + i) for i, b in enumerate(bodies)}
        children = {}
        for b in bodies:
            children.setdefault(int(m.body_parentid[b]), []).append(b)

        def affinity_of(b, extra=0):
            excl = {b, int(m.body_parentid[b])} | set(children.get(b, []))
            allow = 0
            for x in bodies:
                if x not in excl:
                    allow |= bit[x]
            return 1 | allow | extra

        for b in bodies:
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == b and (m.geom_contype[g] or m.geom_conaffinity[g]):
                    m.geom_contype[g] = bit[b]
                    m.geom_conaffinity[g] = affinity_of(b)

        # 2)+3) 传感器侧body(tactile_sensor/left_pad)额外允许与触控盒碰撞
        ts_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tactile_sensor")
        lp_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_pad")
        for b in (ts_bid, lp_bid):
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == b and (m.geom_contype[g] or m.geom_conaffinity[g]):
                    m.geom_conaffinity[g] = affinity_of(b, BIT_TOUCH)

        tg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "touch_finger_geom")
        if tg >= 0:
            m.geom_contype[tg] = BIT_TOUCH
            m.geom_conaffinity[tg] = BIT_TOUCH
        pg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "press_target_geom")
        if pg >= 0:
            m.geom_contype[pg] = BIT_TARGET
            m.geom_conaffinity[pg] = 1 | bit[ts_bid] | bit[lp_bid]

    def _load_signal(self):
        cfg = dict(DEFAULT_SIGNAL_CONFIG)
        cfg_path = os.path.join(BASE_DIR, "signal_config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    cfg.update(loaded)
            except (OSError, ValueError):
                pass
        self.signal_cfg = cfg
        self.signal = SignalEmulator(cfg)

    def _init_arm_ik(self):
        d = self.data
        d.qpos[:] = self.home_qpos
        d.ctrl[:] = self.home_ctrl
        mujoco.mj_forward(self.model, d)
        R0 = d.geom_xmat[self.sensor_geom_id].reshape(3, 3)
        self.arm_R0 = R0.copy()
        self.arm_n_world = (R0 @ np.array([0, -1, 0])).copy()
        self.arm_tp = d.geom_xpos[self.press_geom_id] - self.arm_n_world * 0.012
        self.arm_home_q = self.home_qpos[:7].copy()
        self.ik_data = mujoco.MjData(self.model)
        self.ik_data.qpos[:] = self.home_qpos
        mujoco.mj_forward(self.model, self.ik_data)
        self.arm_force = 0.0
        self.arm_q_ref = self.arm_home_q.copy()
        self.arm_q_goal = None
        self.arm_depth = 0.0
        self.arm_ik_anchors = None

    def _arm_ik_q(self, depth):
        """按压深插值IK锚点解, 避免闭环每步全量解IK"""
        if self.arm_ik_anchors is None:
            depths = np.concatenate([[-ARM_STANDBY],
                                     np.arange(0.0, ARM_MAX_DEPTH * 1000 + 0.1, 1.0) / 1000.0])
            qs = np.array([self._ik(self.arm_home_q,
                                    self.arm_tp + self.arm_n_world * d,
                                    self.arm_R0) for d in depths])
            self.arm_ik_anchors = (depths, qs)
        depths, qs = self.arm_ik_anchors
        return np.array([np.interp(depth, depths, qs[:, k]) for k in range(7)])

    def _ik(self, q0, target_p, target_R, iters=300):
        d2 = self.ik_data
        q = q0.copy()
        for _ in range(iters):
            d2.qpos[:] = self.home_qpos
            d2.qpos[:7] = q
            mujoco.mj_forward(self.model, d2)
            R = d2.geom_xmat[self.sensor_geom_id].reshape(3, 3)
            p = d2.geom_xpos[self.sensor_geom_id] + R @ np.array([0, -SENSOR_HALF_Y, 0])
            dp = target_p - p
            Rerr = target_R @ R.T
            w = np.zeros(4)
            mujoco.mju_mat2Quat(w, Rerr.reshape(9))
            ang = 2 * np.arcsin(min(1, np.linalg.norm(w[1:])))
            xyz = np.zeros(3)
            if ang > 1e-6:
                xyz = w[1:] / np.linalg.norm(w[1:]) * ang
            if np.linalg.norm(dp) < 2e-5 and ang < 1e-4:
                break
            v = np.concatenate([dp, xyz])
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, d2, jacp, jacr, self.sensor_body_id)
            J = np.vstack([jacp[:, :7], jacr[:, :7]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(6), v)
            nrm = np.linalg.norm(dq)
            if nrm > 0.004:
                dq *= 0.004 / nrm
            q = q + dq
        return q

    def _calibrate_home(self):
        key = self.model.key("home")
        self.home_qpos = key.qpos.copy()
        d = self.data
        d.qpos[:] = self.home_qpos
        d.ctrl[:] = key.ctrl
        self.data.mocap_pos[self.touch_mocap] = PARK_POS
        self._place_grasp_box()
        mujoco.mj_forward(self.model, d)
        bias = d.qfrc_bias.copy()
        self.home_ctrl = key.ctrl.copy()
        for aid in self.arm_acts:
            jid = self.model.actuator_trnid[aid][0]
            dof = self.model.jnt_dofadr[jid]
            kp = self.model.actuator_gainprm[aid][0]
            self.home_ctrl[aid] += bias[dof] / kp
        for _ in range(2):
            d.qpos[:] = self.home_qpos
            d.qvel[:] = 0
            d.ctrl[:] = self.home_ctrl
            self._place_grasp_box()
            for _ in range(1500):
                mujoco.mj_step(self.model, d)
            err = d.qpos[:7] - self.home_qpos[:7]
            for k, aid in enumerate(self.arm_acts):
                self.home_ctrl[aid] += err[k]

    def reset_scene(self):
        d = self.data
        d.qpos[:] = self.home_qpos
        d.qvel[:] = 0
        d.ctrl[:] = self.home_ctrl
        mujoco.mj_forward(self.model, d)
        self.touch_pos = None
        self.touch_depth = 0.0
        self.fz_filt = 0.0
        self.grasp_ctrl = 0.0
        self.tilt_bad_t = 0.0
        self.signal.reset()
        self.tare_offset = np.zeros(3)
        self.arm_q_ref = self.arm_home_q.copy()
        self.arm_q_goal = None
        self.arm_depth = 0.0
        with self.lock:
            self.touch_force = 0.0
            self.touch_sx = 0.0
            self.touch_sy = 0.0
        self._place_grasp_box()

    def _sensor_frame(self):
        R = self.data.geom_xmat[self.sensor_geom_id].reshape(3, 3)
        c = self.data.geom_xpos[self.sensor_geom_id].copy()
        return R, c

    def _right_pad_touches_plate(self):
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            ids = {int(con.geom1), int(con.geom2)}
            if self.grasp_geom_id in ids and ids & self.right_pad_geoms:
                return True
        return False

    def _right_pad_plate_gap(self):
        """右指垫内侧面到薄板近侧面的间隙(m), 沿传感器垫法向(闭合轴)"""
        R, c = self._sensor_frame()
        axis = R @ np.array([0, 1, 0])
        pad_y = (self.data.geom_xpos[self.right_pad_main] - c) @ axis
        plate_y = (self.data.qpos[self.grasp_qadr:self.grasp_qadr + 3] - c) @ axis
        return float(pad_y - 0.004 - (plate_y + GRASP_BOX_HALF_Y))

    def _right_pad_hits_self(self):
        """右指垫撞上传感器/左指垫(薄板不在中间时全闭会自碰撞锁死腕部)"""
        for i in range(self.data.ncon):
            ids = {int(self.data.contact[i].geom1), int(self.data.contact[i].geom2)}
            if ids & self.right_pad_geoms and \
                    (self.sensor_geom_id in ids or ids & self.left_pad_geoms):
                return True
        return False

    def _place_grasp_box(self):
        R, c = self._sensor_frame()
        face = c + R @ np.array([0, -SENSOR_HALF_Y, 0])
        if self.mode == "grasp":
            # 薄板悬在垫面外0.3mm(零预压, 0.2mm重叠会被接触求解器横向弹出60mm),
            # 未夹住前由update_interaction每步放回, 夹住后交给物理
            center = face + R @ np.array([0, -(GRASP_BOX_HALF_Y + 0.0003), 0])
            quat = np.array([1.0, 0, 0, 0])
        else:
            center = BOX_PARK
            quat = np.array([1.0, 0, 0, 0])
        jadr = self.model.jnt_qposadr[self.grasp_jnt]
        self.data.qpos[jadr:jadr + 3] = center
        self.data.qpos[jadr + 3:jadr + 7] = quat
        jvadr = self.model.jnt_dofadr[self.grasp_jnt]
        self.data.qvel[jvadr:jvadr + 6] = 0

    def set_mode(self, mode):
        if mode not in ("touch", "grasp", "arm") or mode == self.mode:
            return
        self.mode = mode
        self.reset_scene()

    def update_interaction(self):
        if self.pending_mode is not None:
            m, self.pending_mode = self.pending_mode, None
            if m in ("touch", "grasp", "arm") and m != self.mode:
                self.mode = m
                self.reset_scene()
        if self.reset_request:
            self.reset_request = False
            self.reset_scene()
        # 姿态自动恢复: 夹爪倾角>20°持续2s且无任何指令 -> 复位场景。
        # 视窗拖拽扰动力无上限, 仍可能把臂甩进意外卡死姿态; 复位是最后的自愈手段
        if self.tilt_deg() > 20.0:
            self.tilt_bad_t += self.model.opt.timestep
            if self.tilt_bad_t > 2.0:
                with self.lock:
                    idle = (self.touch_force <= 0.01 and self.arm_force <= 0.05
                            and not self.auto_mode)
                if idle and self.grasp_pct < 1.0:
                    print(f"[自动恢复] 夹爪姿态异常 tilt={self.tilt_deg():.1f}deg, 已复位场景")
                    self.reset_scene()
                self.tilt_bad_t = 0.0
        else:
            self.tilt_bad_t = 0.0
        if self.auto_mode:
            self.auto_t += 0.004
            self.grasp_pct = (np.sin(self.auto_t) * 0.5 + 0.5) * 100
            self.touch_force = (np.sin(self.auto_t) * 0.5 + 0.5) * 6.0
            self.touch_sx = np.sin(self.auto_t * 0.6) * 0.7
            self.touch_sy = np.cos(self.auto_t * 0.45) * 0.7

        if self.mode == "arm":
            self.data.ctrl[self.finger_act] = self.home_ctrl[self.finger_act]
            self.data.mocap_pos[self.touch_mocap] = PARK_POS
            if self.arm_force <= 0.05:
                # 撤力: 指垫退到目标面外ARM_STANDBY真间隙, 压深积分清零
                self.arm_depth = 0.0
                self.arm_q_goal = self._arm_ik_q(-ARM_STANDBY)
            else:
                # 力闭环: 开环定深标定(N/mm)随臂姿变化失效(实测同一指令力
                # 在不同目标位置偏差4倍), 改为压深按(指令力-实测力)积分,
                # 接触后才积分, 稳态实测力≈指令力, 与touch模式同一套机理
                if self.fz_filt > 0.05:
                    err = self.arm_force - self.fz_filt
                    self.arm_depth = float(np.clip(
                        self.arm_depth + ARM_KI * err * self.model.opt.timestep,
                        0.0, ARM_MAX_DEPTH))
                elif self.arm_depth <= 0.0:
                    self.arm_depth = 2e-6
                self.arm_q_goal = self._arm_ik_q(self.arm_depth)
            self.arm_q_ref += np.clip(self.arm_q_goal - self.arm_q_ref, -ARM_RATE, ARM_RATE)
            for k, aid in enumerate(self.arm_acts):
                self.data.ctrl[aid] = self.home_ctrl[aid] + (self.arm_q_ref[k] - self.arm_home_q[k])
        elif self.mode == "grasp":
            # fingers_actuator 是腱长度伺服: F = 0.3137*ctrl - 100*L - 10*V,
            # 平衡腱长 L* = ctrl*0.3137/100, ctrl 0->255 恰好对应 L* 0->0.8(全开->全闭),
            # 滑动条0-100%线性映射闭合行程(夹到物体后由forcerange±15N限 squeeze 力)
            ctrl_target = self.grasp_pct / 100.0 * 255.0
            if ctrl_target > self.grasp_ctrl and (
                    self._right_pad_hits_self() or self.fz_filt > GRASP_FZ_CAP):
                pass  # 冻结: 右垫顶到传感器/左垫(薄板弹出防自碰撞) 或夹持力达上限
            elif ctrl_target < self.grasp_ctrl:
                rate = 600.0  # 张开不限速
                self.grasp_ctrl += float(np.clip(ctrl_target - self.grasp_ctrl,
                                                 -rate * self.model.opt.timestep, 0.0))
            else:
                # 距板6mm内蠕速逼近, 接触链~1MN/m, 快撞会把薄板弹出60mm
                rate = GRASP_FAST_RATE if self._right_pad_plate_gap() > 0.006 else GRASP_SLOW_RATE
                self.grasp_ctrl = float(np.clip(
                    self.grasp_ctrl + rate * self.model.opt.timestep,
                    self.grasp_ctrl, ctrl_target))
            self.data.ctrl[self.finger_act] = self.grasp_ctrl
            self.data.mocap_pos[self.touch_mocap] = PARK_POS
            # 薄板未被右指垫碰到且传感器无力: 每步放回垫面(模拟手扶板, 否则会被
            # 指间挤出或掉落); 右指垫一旦接触薄板立即交给物理, 由指垫把它压上传感器
            if self.fz_filt < 0.1 and not self._right_pad_touches_plate():
                self._place_grasp_box()
        else:
            self.data.ctrl[self.finger_act] = 0.0
            with self.lock:
                force, sx, sy = self.touch_force, self.touch_sx, self.touch_sy
            R, c = self._sensor_frame()
            face = c + R @ np.array([0, -SENSOR_HALF_Y, 0])
            n_in = R @ np.array([0, -1, 0])
            if force <= 0.01:
                self.touch_depth = 0.0
                target = PARK_POS
            else:
                # 力闭环: 木块若开环定深会与臂后退形成正反馈(力涨到66N不收敛),
                # 改为用上一步实测Fz(0.2低通)积分压深, 稳态实测力≈指令力
                lat = R @ np.array([sy * 0.008, 0, sx * 0.008])
                approach = face + n_in * TOUCH_BOX_HALF + lat  # 零压深目标(刚接触)
                if self.touch_depth <= 0.0:
                    self.touch_depth = 2e-6
                if self.fz_filt > 0.05:  # 接触后才积分压深, 接近途中不预充
                    err = force - self.fz_filt
                    self.touch_depth = float(np.clip(
                        self.touch_depth + TOUCH_KI * err * self.model.opt.timestep,
                        0.0, TOUCH_MAX_DEPTH))
                target = approach - n_in * self.touch_depth
            if self.touch_pos is None:
                self.touch_pos = target.copy()
            else:
                # 两段式限速: 远处150mm/s巡航, 距目标3mm内降至5mm/s,
                # 防止从停靠点高速撞入传感器产生数十牛冲击尖峰
                delta = (target - self.touch_pos) * 0.2
                dist = float(np.linalg.norm(target - self.touch_pos))
                vmax = (0.150 if dist > 0.003 else 0.005) * self.model.opt.timestep
                nrm = float(np.linalg.norm(delta))
                if nrm > vmax:
                    delta *= vmax / nrm
                self.touch_pos = self.touch_pos + delta
            self.data.mocap_pos[self.touch_mocap] = self.touch_pos
            if force > 0.01:
                quat = np.empty(4)
                mujoco.mju_mat2Quat(quat, R.reshape(9))
                self.data.mocap_quat[self.touch_mocap] = quat

    def tilt_deg(self):
        """传感器垫相对home基准姿态的倾角(度), 用于监控夹爪水平状态"""
        R = self.data.geom_xmat[self.sensor_geom_id].reshape(3, 3)
        Rel = R @ self.arm_R0.T
        w = np.empty(4)
        mujoco.mju_mat2Quat(w, Rel.reshape(9))
        return float(np.degrees(2 * np.arcsin(min(1.0, np.linalg.norm(w[1:])))))

    def get_contact_forces(self):
        f_world = np.zeros(3)
        R = self.data.geom_xmat[self.sensor_geom_id].reshape(3, 3)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if self.sensor_geom_id not in (con.geom1, con.geom2):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            f_world += con.frame.reshape(3, 3).T @ force[:3]
        f_local = R.T @ f_world
        return f_local[2], f_local[0], -f_local[1]

    def tare(self):
        self.tare_offset = np.array([self.fx, self.fy, self.fz])

    def get_data(self):
        fx = self.fx - self.tare_offset[0]
        fy = self.fy - self.tare_offset[1]
        fz = self.fz - self.tare_offset[2]
        mag = float(np.sqrt(fx * fx + fy * fy + fz * fz))
        if mag > self.peak_force:
            self.peak_force = mag
        if mag < 0.001:
            theta = phi = 0.0
        else:
            theta = float(np.degrees(np.arccos(np.clip(fz / mag, -1, 1))))
            phi = float(np.degrees(np.arctan2(fy, fx)))
        return {
            "fx": round(fx, 4), "fy": round(fy, 4), "fz": round(fz, 4),
            "mag": round(mag, 4), "theta": round(theta, 2), "phi": round(phi, 2),
            "fps": self.fps, "sampleCount": self.sample_count,
            "peakForce": round(self.peak_force, 4),
            "elapsed": round(time.time() - self.start_time, 1) if self.start_time > 0 else 0,
            "mode": self.mode, "graspPct": round(self.grasp_pct, 1),
            "tilt": round(self.tilt_now, 2),
            "touchForce": round(self.touch_force, 2),
            "armForce": round(self.arm_force, 2),
            "rawFx": round(self.raw_fx, 4), "rawFy": round(self.raw_fy, 4), "rawFz": round(self.raw_fz, 4),
            "signal": {
                "enabled": bool(self.signal.cfg.get("enabled", True)),
                "sampleRate": self.signal.cfg.get("sample_rate", 0),
                "adcBits": self.signal.cfg.get("adc_bits", 0),
                "fullScale": self.signal.cfg.get("full_scale", 0),
                "lsb": 2.0 * self.signal.cfg.get("full_scale", 1.0)
                       / max(1, 2 ** int(self.signal.cfg.get("adc_bits", 12))),
            },
            "modelName": "Gen3+2F85", "boardId": "SIM-KINOVA",
            "status": 0x01, "connected": True
        }


HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gen3+2F85 触觉传感器仿真 - 三维力实时显示</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0e1a;color:#e2e8f0;font-family:'Segoe UI',system-ui,sans-serif;overflow:hidden;height:100vh}
#app{display:grid;grid-template-columns:1.15fr 1fr 1fr;grid-template-rows:1fr 1fr;gap:8px;padding:8px;height:100vh}
.panel{background:rgba(30,41,59,.6);border:1px solid rgba(56,189,248,.15);border-radius:12px;padding:12px;backdrop-filter:blur(10px);overflow:hidden;display:flex;flex-direction:column}
.panel-title{font-size:11px;color:#38bdf8;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;display:flex;align-items:center;gap:6px}
.panel-title::before{content:'';width:6px;height:6px;background:#38bdf8;border-radius:50%;box-shadow:0 0 6px #38bdf8}
#chart3d{flex:1;border-radius:8px;overflow:hidden}
#timeChart{flex:1}
.bars{flex:1;display:flex;flex-direction:column;justify-content:center;gap:16px;padding:10px}
.bar-row{display:flex;align-items:center;gap:10px}
.bar-label{font-size:13px;font-weight:700;width:30px;text-align:right}
.bar-track{flex:1;height:28px;background:rgba(255,255,255,.05);border-radius:14px;position:relative;overflow:hidden;display:flex;justify-content:center}
.bar-fill{height:100%;border-radius:14px;transition:width .05s;position:absolute;top:0}
.bar-pos{right:50%;background:linear-gradient(90deg,transparent,#22d3ee);border-radius:14px 0 0 14px}
.bar-neg{left:50%;background:linear-gradient(270deg,transparent,#f87171);border-radius:0 14px 14px 0}
.bar-center{width:1px;height:100%;background:rgba(255,255,255,.2);position:absolute;left:50%;z-index:1}
.bar-val{font-size:12px;font-family:Consolas,monospace;width:70px;text-align:left;color:#94a3b8}
#touchpad{position:relative;flex:1;border-radius:8px;border:1px solid rgba(56,189,248,.15);background:radial-gradient(circle at 50% 50%,rgba(56,189,248,.08),rgba(10,14,26,.95));overflow:hidden;cursor:crosshair;touch-action:none;user-select:none;-webkit-user-select:none}
#touchpad:active{border-color:rgba(251,191,36,.45)}
.tp-grid{position:absolute;inset:0;background-image:linear-gradient(rgba(56,189,248,.08) 1px,transparent 1px),linear-gradient(90deg,rgba(56,189,248,.08) 1px,transparent 1px);background-size:12.5% 12.5%;pointer-events:none}
.tp-sensor{position:absolute;left:50%;top:50%;width:26%;aspect-ratio:1;transform:translate(-50%,-50%);background:rgba(34,197,94,.16);border:1px solid rgba(34,197,94,.55);border-radius:3px;box-shadow:0 0 14px rgba(34,197,94,.2);pointer-events:none}
.tp-sensor::after{content:'传感器';position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:10px;color:#4ade80}
.tp-cross-h{position:absolute;left:0;right:0;top:50%;height:1px;background:rgba(255,255,255,.12);pointer-events:none}
.tp-cross-v{position:absolute;top:0;bottom:0;left:50%;width:1px;background:rgba(255,255,255,.12);pointer-events:none}
.tp-ring{position:absolute;left:50%;top:50%;width:92%;aspect-ratio:1;transform:translate(-50%,-50%);border:1px dashed rgba(251,191,36,.25);border-radius:50%;pointer-events:none}
.tp-finger{position:absolute;left:50%;top:50%;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;background:radial-gradient(circle,rgba(251,191,36,.85),rgba(251,191,36,.15));border:2px solid #fbbf24;box-shadow:0 0 10px rgba(251,191,36,.45);pointer-events:none;transition:left .4s cubic-bezier(.2,1.7,.4,1),top .4s cubic-bezier(.2,1.7,.4,1),width .08s,height .08s,margin .08s,box-shadow .08s}
.tp-readout{position:absolute;top:6px;left:10px;font:700 13px Consolas,monospace;color:#fbbf24;pointer-events:none;text-shadow:0 0 8px rgba(0,0,0,.8)}
.tp-hint{position:absolute;bottom:6px;left:0;right:0;text-align:center;font-size:10px;color:#64748b;pointer-events:none}
.ctrl{display:flex;flex-direction:column;gap:8px;padding:4px;flex:1;overflow-y:auto}
.mode-tabs{display:flex;gap:6px}
.mode-tab{flex:1;text-align:center;padding:8px 0;border-radius:8px;border:1px solid rgba(56,189,248,.25);background:rgba(56,189,248,.08);color:#7dd3fc;font-size:13px;font-weight:700;cursor:pointer;transition:all .2s}
.mode-tab.active{background:rgba(56,189,248,.3);color:#fff;box-shadow:0 0 12px rgba(56,189,248,.3)}
.ctrl-row{display:flex;gap:8px;align-items:center}
.ctrl-btn{background:rgba(56,189,248,.15);border:1px solid rgba(56,189,248,.3);color:#7dd3fc;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:600;transition:all .2s}
.ctrl-btn:hover{background:rgba(56,189,248,.25)}
.ctrl-btn.tare{background:rgba(34,197,94,.15);border-color:rgba(34,197,94,.3);color:#4ade80}
.ctrl-btn.pause{background:rgba(251,191,36,.15);border-color:rgba(251,191,36,.3);color:#fbbf24}
.ctrl-btn.pause.paused{background:rgba(239,68,68,.15);border-color:rgba(239,68,68,.3);color:#f87171}
.ctrl-btn.auto{background:rgba(168,85,247,.15);border-color:rgba(168,85,247,.3);color:#c084fc;width:100%}
.ctrl-btn.auto.on{background:rgba(239,68,68,.15);border-color:rgba(239,68,68,.3);color:#f87171}
.slider-block{margin-top:2px}
.slider-head{display:flex;justify-content:space-between;font-size:11px;color:#94a3b8;margin-bottom:4px}
.sens-slider{width:100%;-webkit-appearance:none;height:6px;border-radius:3px;background:rgba(255,255,255,.1);outline:none}
.sens-slider::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:#fbbf24;cursor:pointer}
.sens-val{color:#fbbf24;font-weight:700;font-family:Consolas,monospace}
.info-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:8px}
.info-item{background:rgba(255,255,255,.03);padding:6px 10px;border-radius:6px;font-size:11px}
.info-item span{color:#64748b;font-size:10px;display:block}
.info-item b{color:#e2e8f0;font-family:Consolas,monospace;font-size:14px}
.status-bar{position:fixed;top:8px;right:12px;z-index:100;display:flex;gap:12px;align-items:center;font-size:11px}
.status-dot{width:8px;height:8px;border-radius:50%;background:#22c55e;box-shadow:0 0 8px #22c55e}
.sim-badge{background:rgba(168,85,247,.2);border:1px solid rgba(168,85,247,.4);color:#c084fc;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700}
#app .panel:nth-child(3){grid-row:1/3;grid-column:3}
</style>
</head>
<body>
<div class="status-bar">
  <span class="sim-badge">Gen3 + 2F-85 触觉仿真</span>
  <span class="status-dot"></span>
  <span id="statusText" style="color:#22c55e">仿真运行中</span>
  <span style="color:#64748b">| FPS: <b id="fpsVal" style="color:#38bdf8">0</b></span>
</div>
<div id="app">
  <div class="panel">
    <div class="panel-title">三维力向量 (Three.js)</div>
    <div id="chart3d"></div>
  </div>
  <div class="panel">
    <div class="panel-title">触控按压板 - 按住拖拽触碰传感器</div>
    <div id="touchpad">
      <div class="tp-grid"></div>
      <div class="tp-ring"></div>
      <div class="tp-cross-h"></div><div class="tp-cross-v"></div>
      <div class="tp-sensor"></div>
      <div class="tp-finger" id="tpFinger"></div>
      <div class="tp-readout" id="tpReadout">0.0 N</div>
      <div class="tp-hint">按住拖拽 = 手指按压 | 离中心越远压力越大 | 侧向拖动产生剪切力 | 松开=释放</div>
    </div>
  </div>
  <div class="panel">
    <div class="panel-title">控制面板</div>
    <div class="ctrl">
      <div class="mode-tabs">
        <div class="mode-tab active" id="tabTouch" onclick="setMode('touch')">触碰模式</div>
        <div class="mode-tab" id="tabGrasp" onclick="setMode('grasp')">抓取模式</div>
        <div class="mode-tab" id="tabArm" onclick="setMode('arm')">臂动触碰</div>
      </div>
      <div class="ctrl-row">
        <button class="ctrl-btn tare" onclick="doTare()">清零校准</button>
        <button class="ctrl-btn pause" id="pauseBtn" onclick="togglePause()">暂停</button>
      </div>
      <div class="slider-block">
        <div class="slider-head"><span>显示灵敏度</span><span class="sens-val" id="sensVal">3.0x</span></div>
        <input type="range" class="sens-slider" id="sensSlider" min="0.5" max="10" step="0.1" value="3" oninput="setSens(this.value)">
      </div>
      <div class="slider-block">
        <div class="slider-head"><span>白噪声 σ (信号层)</span><span class="sens-val" id="noiseVal" style="color:#a78bfa">0.020N</span></div>
        <input type="range" class="sens-slider" id="noiseSlider" min="0" max="0.1" step="0.002" value="0.02" oninput="setNoise(this.value)">
      </div>
      <div class="slider-block" style="margin-top:10px;border-top:1px solid rgba(244,114,182,.25);padding-top:8px">
        <div class="slider-head"><span style="color:#f472b6;font-weight:700">真实信号特征层</span><span id="sigState" style="color:#f472b6;font-weight:700">开启</span></div>
        <button class="ctrl-btn" id="sigToggle" onclick="toggleSignal()" style="width:100%;background:rgba(244,114,182,.15);border-color:rgba(244,114,182,.3);color:#f472b6">关闭信号层 (输出纯物理力)</button>
      </div>
      <div class="slider-block"><div class="slider-head"><span>采样率</span><span class="sens-val" id="sample_rateVal">200Hz</span></div>
        <input type="range" class="sens-slider" min="10" max="1000" step="10" value="200" oninput="setSig('sample_rate',this.value)"></div>
      <div class="slider-block"><div class="slider-head"><span>ADC 位数</span><span class="sens-val" id="adc_bitsVal">12bit</span></div>
        <input type="range" class="sens-slider" min="8" max="16" step="1" value="12" oninput="setSig('adc_bits',this.value)"></div>
      <div class="slider-block"><div class="slider-head"><span>传感器带宽</span><span class="sens-val" id="bandwidthVal">50Hz</span></div>
        <input type="range" class="sens-slider" min="1" max="500" step="1" value="50" oninput="setSig('bandwidth',this.value)"></div>
      <div class="slider-block"><div class="slider-head"><span>轴间串扰</span><span class="sens-val" id="crosstalk_pctVal">2%</span></div>
        <input type="range" class="sens-slider" min="0" max="10" step="0.5" value="2" oninput="setSig('crosstalk_pct',this.value)"></div>
      <div class="slider-block"><div class="slider-head"><span>蠕变 (恒载缓漂)</span><span class="sens-val" id="creep_gain_pctVal">2%</span></div>
        <input type="range" class="sens-slider" min="0" max="10" step="0.5" value="2" oninput="setSig('creep_gain_pct',this.value)"></div>
      <div class="slider-block"><div class="slider-head"><span>迟滞 (回隙)</span><span class="sens-val" id="hysteresis_pctVal">2%</span></div>
        <input type="range" class="sens-slider" min="0" max="10" step="0.5" value="2" oninput="setSig('hysteresis_pct',this.value)"></div>
      <div class="slider-block"><div class="slider-head"><span>零漂 σ</span><span class="sens-val" id="drift_sigmaVal">0.05N</span></div>
        <input type="range" class="sens-slider" min="0" max="0.5" step="0.01" value="0.05" oninput="setSig('drift_sigma',this.value)"></div>
      <div class="slider-block" id="graspBlock" style="display:none">
        <div class="slider-head"><span>夹爪闭合度 (255=夹紧)</span><span style="color:#c084fc;font-family:Consolas,monospace" id="graspVal">0%</span></div>
        <input type="range" class="sens-slider" id="graspSlider" min="0" max="100" step="1" value="0" oninput="setGrasp(this.value)" style="background:rgba(168,85,247,.15)">
      </div>
      <div class="slider-block" id="armBlock" style="display:none">
        <div class="slider-head"><span>机械臂按压力 (0=悬停接触面)</span><span style="color:#34d399;font-family:Consolas,monospace" id="armVal">0.0N</span></div>
        <input type="range" class="sens-slider" id="armSlider" min="0" max="10" step="0.1" value="0" oninput="setArmForce(this.value)" style="background:rgba(52,211,153,.15)">
      </div>
      <button class="ctrl-btn auto" id="autoBtn" onclick="toggleAuto()">自动演示模式</button>
      <button class="ctrl-btn" onclick="doReset()" style="width:100%">复位 (机械臂回原位)</button>
      <div class="info-grid">
        <div class="info-item"><span>合力 |F|</span><b id="magVal" style="color:#fbbf24">0.000</b></div>
        <div class="info-item"><span>俯仰角 θ</span><b id="thetaVal">0.0°</b></div>
        <div class="info-item"><span>偏航角 φ</span><b id="phiVal">0.0°</b></div>
        <div class="info-item"><span>峰值力</span><b id="peakVal">0.000</b></div>
        <div class="info-item"><span>采样数</span><b id="sampleVal">0</b></div>
        <div class="info-item"><span>运行时间</span><b id="timeVal">0.0s</b></div>
        <div class="info-item"><span>ADC分辨率</span><b id="lsbVal">0.0244N</b></div>
      </div>
      <div style="font-size:10px;color:#64748b;margin-top:6px;line-height:1.6">
        触碰模式: 在触控板上按住拖拽模拟手指压传感器<br>
        抓取模式: 蓝色方块置于指间, 拖动闭合度滑块夹紧<br>
        臂动触碰: 机械臂IK伸过去用传感器压红色目标物<br>
        信号特征层: 串扰/迟滞/蠕变/带宽/噪声/量化/采样保持/零漂<br>
        时序图虚线=物理真值 实线=传感器输出 | MuJoCo 3D视窗同步
      </div>
    </div>
  </div>
  <div class="panel">
    <div class="panel-title">力分量时序图</div>
    <canvas id="timeChart"></canvas>
  </div>
  <div class="panel">
    <div class="panel-title">分力条形图</div>
    <div class="bars">
      <div class="bar-row"><span class="bar-label" style="color:#22d3ee">Fx</span><div class="bar-track"><div class="bar-fill bar-pos" id="barFxP"></div><div class="bar-fill bar-neg" id="barFxN"></div><div class="bar-center"></div></div><span class="bar-val" id="barFxV">0.00N</span></div>
      <div class="bar-row"><span class="bar-label" style="color:#f87171">Fy</span><div class="bar-track"><div class="bar-fill bar-pos" id="barFyP"></div><div class="bar-fill bar-neg" id="barFyN"></div><div class="bar-center"></div></div><span class="bar-val" id="barFyV">0.00N</span></div>
      <div class="bar-row"><span class="bar-label" style="color:#a78bfa">Fz</span><div class="bar-track"><div class="bar-fill bar-pos" id="barFzP"></div><div class="bar-fill bar-neg" id="barFzN"></div><div class="bar-center"></div></div><span class="bar-val" id="barFzV">0.00N</span></div>
      <div class="bar-row"><span class="bar-label" style="color:#fbbf24">|F|</span><div class="bar-track"><div class="bar-fill bar-pos" id="barMagP" style="background:linear-gradient(90deg,transparent,#fbbf24)"></div><div class="bar-center"></div></div><span class="bar-val" id="barMagV" style="color:#fbbf24">0.00N</span></div>
    </div>
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
let sensitivity=3.0,paused=false,displayRange=2.0,mode='touch';
let scene,camera,renderer,forceArrow,xArrow,yArrow,zArrow,forceSphere;
let timeChart;
const HLEN=200;
let tData=[],fxData=[],fyData=[],fzData=[],magData=[],rFxData=[],rFyData=[],rFzData=[];
let startT=Date.now();

function init3D(){
  const c=document.getElementById('chart3d');
  scene=new THREE.Scene();
  camera=new THREE.PerspectiveCamera(45,c.clientWidth/c.clientHeight,0.1,100);
  camera.position.set(3,2.5,3);camera.lookAt(0,0,0);
  renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
  renderer.setSize(c.clientWidth,c.clientHeight);
  renderer.setClearColor(0x0a0e1a,0.3);
  c.appendChild(renderer.domElement);
  scene.add(new THREE.AmbientLight(0xffffff,0.5));
  const dl=new THREE.DirectionalLight(0xffffff,0.6);dl.position.set(5,5,5);scene.add(dl);
  const grid=new THREE.GridHelper(3,15,0x334155,0x1e293b);grid.rotation.x=Math.PI/2;scene.add(grid);
  const axes=new THREE.Group();
  [[1,0,0,0xff4444],[0,1,0,0x44ff44],[0,0,1,0x4488ff]].forEach(d=>{
    const a=new THREE.ArrowHelper(new THREE.Vector3(d[0],d[1],d[2]).normalize(),new THREE.Vector3(0,0,0),1.5,d[3],0.12,0.06);
    axes.add(a);
  });
  scene.add(axes);
  const sGeom=new THREE.BoxGeometry(1.7,0.52,1.7);
  const sMat=new THREE.MeshPhongMaterial({color:0x226633,transparent:true,opacity:0.4});
  const sMesh=new THREE.Mesh(sGeom,sMat);scene.add(sMesh);
  const edges=new THREE.EdgesGeometry(sGeom);
  const line=new THREE.LineSegments(edges,new THREE.LineBasicMaterial({color:0x22d3ee}));
  scene.add(line);
  forceArrow=new THREE.ArrowHelper(new THREE.Vector3(0,0,1),new THREE.Vector3(0,0,0),0.5,0xfbbf24,0.2,0.1);
  forceArrow.visible=false;scene.add(forceArrow);
  xArrow=new THREE.ArrowHelper(new THREE.Vector3(1,0,0),new THREE.Vector3(0,0,0),0.5,0x22d3ee,0.15,0.08);
  xArrow.visible=false;scene.add(xArrow);
  yArrow=new THREE.ArrowHelper(new THREE.Vector3(0,1,0),new THREE.Vector3(0,0,0),0.5,0xf87171,0.15,0.08);
  yArrow.visible=false;scene.add(yArrow);
  zArrow=new THREE.ArrowHelper(new THREE.Vector3(0,0,1),new THREE.Vector3(0,0,0),0.5,0xa78bfa,0.15,0.08);
  zArrow.visible=false;scene.add(zArrow);
  const sphGeom=new THREE.SphereGeometry(0.08,16,16);
  const sphMat=new THREE.MeshPhongMaterial({color:0xfbbf24,emissive:0xfbbf24,emissiveIntensity:0.5});
  forceSphere=new THREE.Mesh(sphGeom,sphMat);forceSphere.visible=false;scene.add(forceSphere);
  let dragging=false,px=0,py=0,az=120,el=25,dis=4;
  c.addEventListener('mousedown',e=>{dragging=true;px=e.clientX;py=e.clientY});
  c.addEventListener('mouseup',()=>dragging=false);
  c.addEventListener('mousemove',e=>{
    if(!dragging)return;
    az-=(e.clientX-px)*0.5;el+=(e.clientY-py)*0.5;el=Math.max(-89,Math.min(89,el));px=e.clientX;py=e.clientY;
    const r=dis*Math.cos(el*Math.PI/180);
    camera.position.set(r*Math.cos(az*Math.PI/180),r*Math.sin(az*Math.PI/180),dis*Math.sin(el*Math.PI/180));
    camera.lookAt(0,0,0);
  });
  c.addEventListener('wheel',e=>{e.preventDefault();dis=Math.max(1.5,Math.min(10,dis+e.deltaY*0.005));const r=dis*Math.cos(el*Math.PI/180);camera.position.set(r*Math.cos(az*Math.PI/180),r*Math.sin(az*Math.PI/180),dis*Math.sin(el*Math.PI/180));camera.lookAt(0,0,0)});
  animate3D();
}
function animate3D(){
  requestAnimationFrame(animate3D);
  scene.rotation.z+=0.002;
  renderer.render(scene,camera);
}
function upd3D(fx,fy,fz){
  const s=1.0/Math.max(displayRange,0.1)*sensitivity;
  if(Math.abs(fx)>0.001){xArrow.setDirection(new THREE.Vector3(Math.sign(fx),0,0));xArrow.setLength(Math.max(Math.abs(fx)*s,0.05),0.1,0.06);xArrow.visible=true}else xArrow.visible=false;
  if(Math.abs(fy)>0.001){yArrow.setDirection(new THREE.Vector3(0,Math.sign(fy),0));yArrow.setLength(Math.max(Math.abs(fy)*s,0.05),0.1,0.06);yArrow.visible=true}else yArrow.visible=false;
  if(Math.abs(fz)>0.001){zArrow.setDirection(new THREE.Vector3(0,0,Math.sign(fz)));zArrow.setLength(Math.max(Math.abs(fz)*s,0.05),0.1,0.06);zArrow.visible=true}else zArrow.visible=false;
  const m=Math.sqrt(fx*fx+fy*fy+fz*fz);
  if(m>0.001){const d=new THREE.Vector3(fx,fy,fz).normalize();forceArrow.setDirection(d);forceArrow.setLength(Math.max(m*s,0.1),0.15,0.08);forceArrow.visible=true;forceSphere.position.copy(d.clone().multiplyScalar(m*s));forceSphere.visible=true}else{forceArrow.visible=false;forceSphere.visible=false}
}
function initChart(){
  const ctx=document.getElementById('timeChart').getContext('2d');
  timeChart=new Chart(ctx,{type:'line',data:{labels:[],datasets:[
    {label:'Fx',data:[],borderColor:'#22d3ee',backgroundColor:'rgba(34,211,238,.1)',borderWidth:1.5,pointRadius:0,tension:0.3},
    {label:'Fy',data:[],borderColor:'#f87171',backgroundColor:'rgba(248,113,113,.1)',borderWidth:1.5,pointRadius:0,tension:0.3},
    {label:'Fz',data:[],borderColor:'#a78bfa',backgroundColor:'rgba(167,139,250,.1)',borderWidth:1.5,pointRadius:0,tension:0.3},
    {label:'|F|',data:[],borderColor:'#fbbf24',backgroundColor:'rgba(251,191,36,.1)',borderWidth:2,pointRadius:0,tension:0.3},
    {label:'raw Fx',data:[],borderColor:'rgba(34,211,238,.4)',borderDash:[4,3],borderWidth:1,pointRadius:0,tension:0.3},
    {label:'raw Fy',data:[],borderColor:'rgba(248,113,113,.4)',borderDash:[4,3],borderWidth:1,pointRadius:0,tension:0.3},
    {label:'raw Fz',data:[],borderColor:'rgba(167,139,250,.4)',borderDash:[4,3],borderWidth:1,pointRadius:0,tension:0.3}
  ]},options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{
    x:{display:false},
    y:{gridColor:'rgba(255,255,255,.05)',ticks:{color:'#64748b',font:{size:10}}}
  },plugins:{legend:{labels:{color:'#94a3b8',font:{size:10}}}}}});
}
function updBars(fx,fy,fz){
  const set=(id,v)=>{
    const r=Math.min(Math.abs(v)/Math.max(displayRange,0.1)*sensitivity,1)*50;
    const p=document.getElementById('bar'+id+'P');
    const n=document.getElementById('bar'+id+'N');
    const val=document.getElementById('bar'+id+'V');
    if(v>=0){p.style.width=r+'%';n.style.width='0%'}else{p.style.width='0%';n.style.width=r+'%'}
    val.textContent=v.toFixed(3)+'N';
  };
  set('Fx',fx);set('Fy',fy);set('Fz',fz);
  const mag=Math.sqrt(fx*fx+fy*fy+fz*fz);
  const r=Math.min(mag/Math.max(displayRange,0.1)*sensitivity,1)*50;
  document.getElementById('barMagP').style.width=r+'%';
  document.getElementById('barMagV').textContent=mag.toFixed(3)+'N';
}
async function api(path){try{await fetch(path)}catch(e){}}
async function getData(){try{const r=await fetch('/api/data');return await r.json()}catch(e){return null}}
function doTare(){api('/api/tare')}
function togglePause(){
  paused=!paused;
  const b=document.getElementById('pauseBtn');
  if(paused){b.textContent='继续';b.classList.add('paused')}else{b.textContent='暂停';b.classList.remove('paused')}
}
function setSens(v){sensitivity=parseFloat(v);document.getElementById('sensVal').textContent=sensitivity.toFixed(1)+'x'}
function setNoise(v){document.getElementById('noiseVal').textContent=parseFloat(v).toFixed(3)+'N';api('/api/noise?v='+v)}
const sigLabel={sample_rate:v=>v+'Hz',adc_bits:v=>v+'bit',bandwidth:v=>v+'Hz',noise_sigma:v=>parseFloat(v).toFixed(3)+'N',crosstalk_pct:v=>v+'%',creep_gain_pct:v=>v+'%',hysteresis_pct:v=>v+'%',drift_sigma:v=>parseFloat(v).toFixed(2)+'N'};
function setSig(k,v){document.getElementById(k+'Val').textContent=sigLabel[k](v);api('/api/signal?k='+k+'&v='+v)}
async function toggleSignal(){
  try{
    const r=await fetch('/api/signal?k=enabled&v=toggle');
    const d=await r.json();
    document.getElementById('sigState').textContent=d.enabled?'开启':'关闭';
    document.getElementById('sigToggle').textContent=d.enabled?'关闭信号层 (输出纯物理力)':'开启信号层 (真实传感器特性)';
  }catch(e){}
}
function syncModeUI(m){
  document.getElementById('tabTouch').classList.toggle('active',m==='touch');
  document.getElementById('tabGrasp').classList.toggle('active',m==='grasp');
  document.getElementById('tabArm').classList.toggle('active',m==='arm');
  document.getElementById('graspBlock').style.display=m==='grasp'?'block':'none';
  document.getElementById('armBlock').style.display=m==='arm'?'block':'none';
}
async function setMode(m){
  mode=m;
  syncModeUI(m);
  await api('/api/mode?v='+m);
}
function setArmForce(v){document.getElementById('armVal').textContent=parseFloat(v).toFixed(1)+'N';api('/api/arm?f='+v)}
function setGrasp(v){document.getElementById('graspVal').textContent=parseFloat(v).toFixed(0)+'%';api('/api/grasp?v='+v)}
async function toggleAuto(){
  try{
    const r=await fetch('/api/auto?v=toggle');
    const d=await r.json();
    const btn=document.getElementById('autoBtn');
    if(d.auto){btn.textContent='停止自动演示';btn.classList.add('on')}
    else{btn.textContent='自动演示模式';btn.classList.remove('on')}
  }catch(e){}
}
function doReset(){api('/api/reset')}
const tp=document.getElementById('touchpad');
const finger=document.getElementById('tpFinger');
const readout=document.getElementById('tpReadout');
let tpDown=false;
function tpUpdate(e){
  const rect=tp.getBoundingClientRect();
  let x=(e.clientX-rect.left)/rect.width*2-1;
  let y=(e.clientY-rect.top)/rect.height*2-1;
  const r=Math.min(1,Math.hypot(x,y));
  if(r>0.001){x/=Math.max(Math.hypot(x,y),1e-6);y/=Math.max(Math.hypot(x,y),1e-6);x*=r;y*=r}
  finger.style.left=(50+x*44)+'%';
  finger.style.top=(50+y*44)+'%';
  const size=22+r*14;
  finger.style.width=size+'px';finger.style.height=size+'px';
  finger.style.margin=(-size/2)+'px '+(size/2)+'px 0 0';
  const force=r*10;
  readout.textContent=force.toFixed(1)+' N';
  if(mode==='touch'&&!document.getElementById('autoBtn').classList.contains('on'))
    api('/api/touch?x='+x.toFixed(3)+'&y='+y.toFixed(3)+'&f='+force.toFixed(2));
}
tp.addEventListener('pointerdown',e=>{tpDown=true;tp.setPointerCapture(e.pointerId);tpUpdate(e)});
tp.addEventListener('pointermove',e=>{if(tpDown)tpUpdate(e)});
function tpRelease(){
  if(!tpDown)return;
  tpDown=false;
  finger.style.left='50%';finger.style.top='50%';
  finger.style.width='22px';finger.style.height='22px';finger.style.margin='-11px 0 0 -11px';
  readout.textContent='0.0 N';
  api('/api/touch?x=0&y=0&f=0');
}
tp.addEventListener('pointerup',tpRelease);
tp.addEventListener('pointercancel',tpRelease);
async function loop(){
  const d=await getData();
  if(d){
    document.getElementById('fpsVal').textContent=d.fps;
    document.getElementById('sampleVal').textContent=d.sampleCount;
    document.getElementById('timeVal').textContent=d.elapsed.toFixed(1)+'s';
    document.getElementById('peakVal').textContent=d.peakForce.toFixed(3);
    if(d.signal){
      document.getElementById('lsbVal').textContent=d.signal.lsb.toFixed(4)+'N';
      document.getElementById('sigState').textContent=d.signal.enabled?'开启':'关闭';
    }
    if(d.mode!==mode){mode=d.mode;syncModeUI(mode)}
    if(!paused){
      document.getElementById('magVal').textContent=d.mag.toFixed(3);
      document.getElementById('thetaVal').textContent=d.theta.toFixed(1)+'°';
      document.getElementById('phiVal').textContent=d.phi.toFixed(1)+'°';
      upd3D(d.fx,d.fy,d.fz);
      updBars(d.fx,d.fy,d.fz);
      const mx=Math.max(Math.abs(d.fx),Math.abs(d.fy),Math.abs(d.fz),d.mag);
      if(mx>displayRange*0.8)displayRange=mx*1.5;
      const t=(Date.now()-startT)/1000;
      tData.push(t);fxData.push(d.fx);fyData.push(d.fy);fzData.push(d.fz);magData.push(d.mag);
      rFxData.push(d.rawFx!==undefined?d.rawFx:d.fx);rFyData.push(d.rawFy!==undefined?d.rawFy:d.fy);rFzData.push(d.rawFz!==undefined?d.rawFz:d.fz);
      while(tData.length>HLEN){tData.shift();fxData.shift();fyData.shift();fzData.shift();magData.shift();rFxData.shift();rFyData.shift();rFzData.shift()}
      if(timeChart&&tData.length>2){
        timeChart.data.labels=tData;
        timeChart.data.datasets[0].data=fxData;
        timeChart.data.datasets[1].data=fyData;
        timeChart.data.datasets[2].data=fzData;
        timeChart.data.datasets[3].data=magData;
        timeChart.data.datasets[4].data=rFxData;
        timeChart.data.datasets[5].data=rFyData;
        timeChart.data.datasets[6].data=rFzData;
        timeChart.update('none');
      }
    }
  }
  setTimeout(loop,50);
}
window.addEventListener('resize',()=>{
  const c=document.getElementById('chart3d');
  if(renderer&&c){camera.aspect=c.clientWidth/c.clientHeight;camera.updateProjectionMatrix();renderer.setSize(c.clientWidth,c.clientHeight)}
});
init3D();initChart();loop();
</script>
</body>
</html>"""


class WebHandler(http.server.BaseHTTPRequestHandler):
    sensor = None

    def log_message(self, *args):
        pass

    def _json(self, obj):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode('utf-8'))

    def _params(self):
        params = self.path.split('?')[1] if '?' in self.path else ''
        return dict(p.split('=', 1) for p in params.split('&') if '=' in p)

    def do_GET(self):
        s = self.sensor
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        elif self.path == '/api/data':
            self._json(s.get_data())
        elif self.path.startswith('/api/mode'):
            v = self._params().get('v', 'touch')
            if v in ("touch", "grasp", "arm"):
                s.pending_mode = v
            self._json({"ok": True, "mode": s.mode})
        elif self.path.startswith('/api/touch'):
            kv = self._params()
            with s.lock:
                s.touch_sx = float(kv.get('x', 0))
                s.touch_sy = float(kv.get('y', 0))
                s.touch_force = float(kv.get('f', 0))
            self._json({"ok": True})
        elif self.path.startswith('/api/grasp'):
            s.grasp_pct = float(self._params().get('v', 0))
            self._json({"ok": True})
        elif self.path.startswith('/api/arm'):
            try:
                s.arm_force = max(0.0, min(ARM_MAX_FORCE, float(self._params().get('f', 0))))
            except ValueError:
                pass
            self._json({"ok": True})
        elif self.path.startswith('/api/tare'):
            s.tare()
            self._json({"ok": True})
        elif self.path.startswith('/api/noise'):
            try:
                s.signal.cfg["noise_sigma"] = max(0.0, float(self._params().get('v', 0.02)))
            except ValueError:
                pass
            self._json({"ok": True})
        elif self.path.startswith('/api/signal'):
            kv = self._params()
            k = kv.get('k', '')
            v = kv.get('v', '')
            cfg = s.signal.cfg
            if k == '__reload__':
                s._load_signal()
                self._json({"ok": True, "cfg": {kk: vv for kk, vv in s.signal.cfg.items()
                                                if isinstance(vv, (int, float, bool, str))}})
                return
            if k == 'enabled':
                if v == 'toggle':
                    cfg['enabled'] = not cfg.get('enabled', True)
                else:
                    cfg['enabled'] = v not in ('0', 'false', 'False')
                s.signal.reset()
                self._json({"ok": True, "enabled": bool(cfg['enabled'])})
                return
            try:
                if k == 'crosstalk_pct':
                    pct = max(0.0, min(10.0, float(v))) / 100.0
                    base = np.array([[0.0, 1.0, -0.5], [-0.75, 0.0, 1.0], [0.5, -1.0, 0.0]])
                    cfg['crosstalk'] = (np.eye(3) + pct * base).tolist()
                elif k == 'creep_gain_pct':
                    cfg['creep_gain'] = max(0.0, min(10.0, float(v))) / 100.0
                elif k == 'hysteresis_pct':
                    cfg['hysteresis_frac'] = max(0.0, min(10.0, float(v))) / 100.0
                elif k == 'adc_bits':
                    cfg['adc_bits'] = max(1, min(24, int(float(v))))
                elif k == 'sample_rate':
                    cfg['sample_rate'] = max(1.0, float(v))
                elif k in ('bandwidth', 'noise_sigma', 'drift_sigma', 'full_scale',
                           'drift_tau', 'creep_tau', 'deadband', 'noise_relative'):
                    cfg[k] = max(0.0, float(v))
            except ValueError:
                pass
            self._json({"ok": True})
        elif self.path.startswith('/api/auto'):
            v = self._params().get('v', 'toggle')
            if v == 'toggle':
                s.auto_mode = not s.auto_mode
            else:
                s.auto_mode = v == '1'
            if s.auto_mode:
                s.auto_t = 0.0
            self._json({"ok": True, "auto": s.auto_mode})
        elif self.path.startswith('/api/debug'):
            s._dbg = True
            self._json({
                "qpos": [round(float(v), 4) for v in s.data.qpos[:7]],
                "qref": [round(float(v), 4) for v in s.arm_q_ref],
                "qgoal": None if s.arm_q_goal is None else [round(float(v), 4) for v in s.arm_q_goal],
                "home": [round(float(v), 4) for v in s.arm_home_q],
                "tp": [round(float(v), 4) for v in s.arm_tp],
                "n": [round(float(v), 4) for v in s.arm_n_world],
                "face": [round(float(v), 4) for v in (s.data.geom_xpos[s.sensor_geom_id])],
                "force": s.arm_force,
            })
        elif self.path.startswith('/api/reset'):
            s.reset_request = True
            self._json({"ok": True})
        else:
            self.send_response(404)
            self.end_headers()


def run_web_server():
    with socketserver.ThreadingTCPServer(("127.0.0.1", PORT), WebHandler) as httpd:
        httpd.daemon_threads = True
        print(f"[Web] http://127.0.0.1:{PORT}/")
        httpd.serve_forever()


def main():
    sensor = SimSensor()
    WebHandler.sensor = sensor

    viewer = None
    try:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(sensor.model, sensor.data)
    except Exception as e:
        print(f"[提示] MuJoCo 3D 视窗不可用({e}), 仅网页可视化")

    threading.Thread(target=run_web_server, daemon=True).start()

    sensor.running = True
    sensor.start_time = time.time()

    print("=" * 60)
    print("  Kinova Gen3 + Robotiq 2F-85 + 触觉传感器 仿真")
    print("=" * 60)
    print("  传感器: 左指垫 sensor_pad (17x17mm, 感测面朝夹爪中心)")
    print("  触碰模式: 网页触控板按住拖拽 -> 手指按压传感器")
    print("  抓取模式: 拖动夹爪闭合度滑块 -> 夹紧指间方块")
    print("  臂动触碰: 拖动按压力滑块 -> 机械臂IK伸过去压红色目标物")
    sc = sensor.signal.cfg
    lsb = 2.0 * sc.get("full_scale", 1.0) / max(1, 2 ** int(sc.get("adc_bits", 12)))
    print(f"  信号特征层: {'开启' if sc.get('enabled') else '关闭'}  "
          f"fs={sc.get('sample_rate')}Hz ADC={sc.get('adc_bits')}bit LSB={lsb:.4f}N "
          f"带宽={sc.get('bandwidth')}Hz (signal_config.json)")
    print()
    print(f"  网页可视化: http://127.0.0.1:{PORT}/")
    print(f"  MuJoCo 3D 视窗: {'已打开' if viewer is not None else '不可用'}")
    print("=" * 60)

    fps_count = 0
    step_i = 0
    last_fps_time = time.time()
    last_log = time.time()

    while sensor.running:
        step_start = time.time()

        # 原生视窗渲染线程会并发复制 mjData, 物理推进必须持同一把锁,
        # 否则 sync 时的 mj_copyDataVisual 会报 "stack is in use" 并中止进程
        with viewer.lock() if viewer is not None else nullcontext():
            sensor.update_interaction()
            mujoco.mj_step(sensor.model, sensor.data)

            fx, fy, fz = sensor.get_contact_forces()
            sensor.raw_fx, sensor.raw_fy, sensor.raw_fz = fx, fy, fz
            sensor.fz_filt += 0.2 * (fz - sensor.fz_filt)
            sensor.tilt_now = sensor.tilt_deg()
        fx, fy, fz = sensor.signal.process(np.array([fx, fy, fz]), sensor.model.opt.timestep)
        sensor.fx = fx
        sensor.fy = fy
        sensor.fz = fz
        sensor.sample_count += 1

        step_i += 1
        if viewer is not None and viewer.is_running() and step_i % 2 == 0:
            viewer.sync()

        fps_count += 1
        now = time.time()
        if now - last_fps_time >= 1.0:
            sensor.fps = fps_count
            fps_count = 0
            last_fps_time = now
        if now - last_log >= 5.0:
            mag = np.sqrt(fx * fx + fy * fy + fz * fz)
            print(f"[{sensor.mode:5s}] Fx={fx:+7.3f} Fy={fy:+7.3f} Fz={fz:+7.3f} |F|={mag:7.3f} N  "
                  f"grasp={sensor.grasp_pct:5.1f}%  tilt={sensor.tilt_now:5.2f}deg  fps={sensor.fps}")
            last_log = now

        elapsed = time.time() - step_start
        sleep_time = sensor.model.opt.timestep - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
