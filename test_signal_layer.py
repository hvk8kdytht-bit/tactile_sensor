"""无界面验证: 信号特征层 (SignalEmulator) 特性 + 三模式 (touch/grasp/arm) 集成回归
用法: python test_signal_layer.py
不启动视窗和网页服务器, 直接驱动 SimSensor 物理循环。
"""
import os
import sys
import json

import numpy as np
import mujoco

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from kinova_sensor_sim import SignalEmulator, SimSensor, DEFAULT_SIGNAL_CONFIG

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def load_effective_cfg():
    cfg = dict(DEFAULT_SIGNAL_CONFIG)
    path = os.path.join(BASE_DIR, "signal_config.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        if isinstance(loaded, dict):
            cfg.update(loaded)
    return cfg


def quiet_cfg(cfg, **over):
    """关闭与被测特性无关的环节"""
    c = dict(cfg)
    c.update(noise_sigma=0.0, noise_relative=0.0, drift_sigma=0.0, deadband=0.0)
    c.update(over)
    return c


def unit_tests(cfg):
    print("\n===== A. SignalEmulator 单元特性 =====")
    lsb = 2.0 * cfg["full_scale"] / (2 ** int(cfg["adc_bits"]))

    # A1 关闭回退: 输出必须等于物理真值
    c = quiet_cfg(cfg, enabled=False)
    em = SignalEmulator(c)
    f = np.array([1.237, -2.348, 3.451])
    out = em.process(f, 0.002)
    check("关闭信号层 -> 原样输出物理真值", np.allclose(out, f, atol=1e-12),
          f"out={np.round(out, 6)}")

    # A2 ADC量化 + 采样保持: 快变输入下输出仍落在LSB栅格且按fs零阶保持
    c = quiet_cfg(cfg)
    em = SignalEmulator(c)
    outs = []
    for i in range(2000):  # 4 s @ dt=2ms, 输入20Hz正弦(每更新周期变化≫LSB)
        t = i * 0.002
        f = np.array([5.0 + 0.5 * np.sin(2 * np.pi * 20 * t)] * 3)
        outs.append(em.process(f, 0.002))
    grid = all(np.max(np.abs(o / lsb - np.round(o / lsb))) < 1e-6 for o in outs)
    check("ADC量化: 全部输出落在LSB整数栅格", grid, f"LSB={lsb:.5f} N")
    idx = [i for i in range(1, len(outs)) if not np.allclose(outs[i], outs[i - 1])]
    n_change = len(idx)
    min_gap_ms = min(idx[i] - idx[i - 1] for i in range(1, len(idx))) * 2.0
    period_ms = 1000.0 / cfg["sample_rate"]
    expect = 2000 * 0.002 / max(3 * 0.002, period_ms / 1000.0)  # dt与周期非整数比时的有效更新率
    check("采样保持: 更新间隔不小于采样周期", min_gap_ms >= period_ms - 2.0,
          f"最小间隔={min_gap_ms:.1f}ms 周期={period_ms:.1f}ms")
    check("采样保持: 4s内更新次数≈有效更新率",
          0.7 * expect < n_change < 1.3 * expect,
          f"{n_change}次, 期望≈{expect:.0f}")

    # A3 串扰: 纯Fz载荷 -> Fx/Fy按矩阵泄漏, Fz含蠕变增量 (关闭迟滞以隔离特性)
    c = quiet_cfg(cfg, hysteresis_frac=0.0)
    em = SignalEmulator(c)
    f_in = np.array([0.0, 0.0, 10.0])
    for _ in range(20000):  # 40 s > 5*creep_tau
        out = em.process(f_in, 0.002)
    m = np.array(cfg["crosstalk"])
    expect = m @ f_in * (1.0 + cfg["creep_gain"])
    check("串扰矩阵: 稳态输出 = M·F·(1+蠕变增益)",
          np.allclose(out, expect, atol=0.05),
          f"out={np.round(out, 3)} expect={np.round(expect, 3)}")

    # A4 带宽: 10N阶跃 -> 63.2%时间常数 ≈ 1/(2π·fc)
    c = quiet_cfg(cfg, creep_gain=0.0, hysteresis_frac=0.0,
                  adc_bits=0, sample_rate=1e6, bandwidth=50.0)
    em = SignalEmulator(c)
    dt = 0.0002
    for _ in range(5000):
        em.process(np.zeros(3), dt)
    t63 = None
    for i in range(1, 2000):
        out = em.process(np.array([10.0, 0, 0]), dt)
        if out[0] >= 6.32:
            t63 = i * dt * 1000.0
            break
    expect_t63 = 1000.0 / (2 * np.pi * 50.0)
    check("带宽一阶低通: 阶跃t63≈1/(2π·fc)",
          t63 is not None and 0.6 * expect_t63 < t63 < 1.5 * expect_t63,
          f"t63={t63:.2f}ms 期望≈{expect_t63:.2f}ms")

    # A5 迟滞: 加载->卸载后的残余读数及其按tau的消退
    tau = cfg.get("hysteresis_tau", 5.0)
    frac = cfg.get("hysteresis_frac", 0.0)
    c = quiet_cfg(cfg, creep_gain=0.0, adc_bits=0, sample_rate=1e6,
                  bandwidth=100000.0)
    em = SignalEmulator(c)
    dt = 0.0005
    for i in range(1000):  # 0.5s 内 0 -> 10N 缓慢加载
        em.process(np.array([0.0, 10.0 * i / 1000.0, 0.0]), dt)
    resid0 = em.process(np.zeros(3), dt)[1]
    check("迟滞回隙: 卸载瞬间残余≈frac×峰值载荷",
          0.6 * frac * 10 < resid0 < 1.4 * frac * 10,
          f"残余={resid0:.3f}N 期望≈{frac * 10:.3f}N")
    for i in range(int(tau / dt)):
        r_tau = em.process(np.zeros(3), dt)[1]
    check("迟滞消退: t=τ时残余≈初始/e",
          0.5 * resid0 / np.e < r_tau < 1.5 * resid0 / np.e,
          f"{resid0:.3f} -> {r_tau:.4f}N (τ={tau}s)")
    for i in range(int(2 * tau / dt)):
        r3 = em.process(np.zeros(3), dt)[1]
    check("迟滞消退: t=3τ时残余基本归零",
          r3 < 0.06 * frac * 10 + 1e-4, f"残余={r3:.5f}N")

    # A6 蠕变: 恒载10N持续3τ -> 读数→F·(1+gain)
    c = quiet_cfg(cfg, hysteresis_frac=0.0, adc_bits=0, sample_rate=1e6,
                  bandwidth=100000.0)
    em = SignalEmulator(c)
    for i in range(int(3 * cfg["creep_tau"] / 0.002)):
        out = em.process(np.array([10.0, 0, 0]), 0.002)
    expect = 10.0 * (1.0 + cfg["creep_gain"])
    check("蠕变: 恒载3τ后读数≈F×(1+gain)",
          abs(out[0] - expect) < 0.01 * 10,
          f"读数={out[0]:.3f}N 期望≈{expect:.3f}N")

    # A7 白噪声: 零输入下输出标准差≈noise_sigma
    c = quiet_cfg(cfg, hysteresis_frac=0.0, creep_gain=0.0, adc_bits=0,
                  sample_rate=1e6, bandwidth=100000.0,
                  noise_sigma=cfg["noise_sigma"])
    em = SignalEmulator(c)
    samples = [em.process(np.zeros(3), 0.002)[0] for _ in range(8000)]
    std = float(np.std(samples))
    check("白噪声: σ≈noise_sigma", 0.7 * cfg["noise_sigma"] < std < 1.3 * cfg["noise_sigma"],
          f"实测σ={std:.4f}N 配置={cfg['noise_sigma']}N")

    # A8 零漂: OU增量统计 (短窗口内稳态σ无法直接估计, 用1s增量方差验证 σ 与 τ)
    c = quiet_cfg(cfg, hysteresis_frac=0.0, creep_gain=0.0, adc_bits=0,
                  sample_rate=1e6, bandwidth=100000.0,
                  drift_sigma=cfg["drift_sigma"])
    em = SignalEmulator(c)
    lag_steps = 500  # 1s
    xs = []
    for i in range(50000):  # 100s
        x = em.process(np.zeros(3), 0.002)[0]
        if i % lag_steps == 0:
            xs.append(x)
    inc = np.diff(np.array(xs))
    tau_d = cfg["drift_tau"]
    expect_std = cfg["drift_sigma"] * np.sqrt(2.0 * (1.0 - np.exp(-lag_steps * 0.002 / tau_d)))
    std = float(np.std(inc))
    check("零漂OU: 1s增量σ≈σ_d·√(2(1-e^(-1/τ)))",
          0.6 * expect_std < std < 1.4 * expect_std,
          f"实测Δσ={std:.4f}N 期望≈{expect_std:.4f}N")


def mode_tests():
    print("\n===== B. 三模式集成回归 (物理真值 vs 传感器输出) =====")
    s = SimSensor()
    dt = s.model.opt.timestep
    cfg = s.signal.cfg
    print(f"  模型步长 dt={dt*1000:.1f}ms  信号层: enabled={cfg.get('enabled')} "
          f"fs={cfg.get('sample_rate')}Hz hysteresis_tau={cfg.get('hysteresis_tau')}s "
          f"(signal_config.json)")
    lsb = 2.0 * cfg["full_scale"] / (2 ** int(cfg["adc_bits"]))

    def run(n, settle):
        raws, sens = [], []
        for i in range(n):
            s.update_interaction()
            mujoco.mj_step(s.model, s.data)
            f = np.array(s.get_contact_forces())
            s.raw_fx, s.raw_fy, s.raw_fz = f
            s.fz_filt += 0.2 * (f[2] - s.fz_filt)
            out = s.signal.process(f, dt)
            s.fx, s.fy, s.fz = out
            if i >= settle:
                raws.append(f.copy())
                sens.append(np.asarray(out).copy())
        return np.array(raws), np.array(sens)

    # B1 touch: 6N触控板按压 -> 力闭环实测Fz≈6N
    s.mode = "touch"
    s.reset_scene()
    s.signal.reset()
    with s.lock:
        s.touch_force, s.touch_sx, s.touch_sy = 6.0, 0.0, 0.0
    raws, sens = run(5000, 4000)  # 力闭环时间常数~0.7s, 预留8s收敛
    raw_fz, sen_fz = raws[:, 2].mean(), sens[:, 2].mean()
    check("touch模式: 6N指令力闭环实测Fz≈6N", 4.5 < raw_fz < 7.5, f"rawFz={raw_fz:.2f}N")
    check("touch模式: 力闭环稳态无振荡", float(raws[:, 2].std()) < 0.6,
          f"std={raws[:, 2].std():.3f}N")
    check("touch模式: 传感器输出跟踪物理真值", abs(sen_fz - raw_fz) < 0.4,
          f"senFz={sen_fz:.2f}N")

    # B2 grasp: 薄板15mm厚贴传感器侧, ~76%闭合才接触; 100%全闭合由GRASP_FZ_CAP
    # 限在~35N(全闭时连杆近奇异, 腱力15N可放大到125N且持续爬升)
    s.mode = "grasp"
    s.reset_scene()
    s.signal.reset()
    s.grasp_pct = 100.0
    raws, sens = run(9000, 8000)
    raw_fz, sen_fz = raws[:, 2].mean(), sens[:, 2].mean()
    check("grasp模式: 全闭合物理真值Fz≈35N(力上限)", 26.0 < raw_fz < 44.0, f"rawFz={raw_fz:.2f}N")
    check("grasp模式: 传感器输出跟踪物理真值", abs(sen_fz - raw_fz) < 1.5,
          f"senFz={sen_fz:.2f}N")

    # B3 arm: 5N指令 -> 机械臂IK按压物理真值Fz≈4.9N
    s.mode = "arm"
    s.reset_scene()
    s.signal.reset()
    s.arm_force = 5.0
    raws, sens = run(9000, 8000)
    raw_fz, sen_fz = raws[:, 2].mean(), sens[:, 2].mean()
    check("arm模式: 5N指令物理真值Fz≈4.9N", 3.5 < raw_fz < 6.5, f"rawFz={raw_fz:.2f}N")
    check("arm模式: 传感器输出跟踪物理真值", abs(sen_fz - raw_fz) < 0.6,
          f"senFz={sen_fz:.2f}N")

    grid = all(np.max(np.abs(o / lsb - np.round(o / lsb))) < 1e-6 for o in sens)
    check("集成链路: 传感器输出保持ADC量化栅格", grid)


def main():
    cfg = load_effective_cfg()
    print("=" * 62)
    print("  信号特征层验证 (无界面)")
    print("=" * 62)
    unit_tests(cfg)
    mode_tests()
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print("\n" + "=" * 62)
    print(f"  结果: {len(RESULTS) - n_fail}/{len(RESULTS)} 通过"
          + ("" if n_fail == 0 else f", {n_fail} 项失败"))
    print("=" * 62)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
