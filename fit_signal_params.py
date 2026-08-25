"""
从真实传感器录制数据拟合信号特征参数 -> signal_config.json
之后重启 kinova_sensor_sim.py (或 GET /api/signal?k=__reload__) 即生效

用法:
  python fit_signal_params.py <录制文件.csv> [--out signal_config.json]

CSV 格式 (空格或逗号分隔, 首行可以是表头):
  列1:   t     采样时间戳 (s)
  列2-4: fx fy fz    传感器输出 (N)
  列5-7: (可选) refx refy refz  参考真实力 (N, 砝码/标定台), 用于拟合串扰矩阵

自动估计: 采样率 / ADC量化(LSB与位数) / 白噪声σ / 零漂 / 带宽 / 串扰(需参考列)
  蠕变与迟滞需要专门加载谱(恒载平台+加卸载循环), 自动估计不可靠 -> 报告中给出人工建议值
录制建议: 开头录 >=10s 空载段(估噪声/零漂/量化), 中间含若干静置和运动段
"""
import sys
import os
import json
import numpy as np


def load_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
    if not rows:
        raise ValueError("没有解析到数值行")
    return np.array(rows)


def autocorr_bandwidth(x, fs):
    x = x - x.mean()
    n = len(x)
    if n < 64:
        return None
    lags = min(n // 2, max(int(fs * 0.2), 8))
    ac = np.correlate(x, x, "full")[n - 1:n - 1 + lags]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    idx = np.where(ac < 1.0 / np.e)[0]
    if len(idx) == 0 or idx[0] == 0:
        return None
    tau = idx[0] / fs
    return 1.0 / (2 * np.pi * tau)


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    path = args[0]
    out = os.path.join(os.path.dirname(os.path.abspath(path)), "signal_config.json")
    if "--out" in args:
        out = args[args.index("--out") + 1]

    a = load_csv(path)
    if a.shape[1] < 4:
        print("CSV 至少需要 4 列: t, fx, fy, fz")
        return
    t = a[:, 0]
    sig = a[:, 1:4]
    ref = a[:, 4:7] if a.shape[1] >= 7 else None
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    print(f"样本 {len(t)} 个, 采样率 {fs:.1f} Hz, 时长 {t[-1] - t[0]:.1f} s")

    # 静默窗口 (最低方差的 1/4) 用于噪声/零漂估计
    win = max(int(fs * 0.5), 8)
    nwin = len(t) // win
    if nwin < 2:
        print("数据太短, 无法分段")
        return
    wvar = np.array([sig[i * win:(i + 1) * win].var(axis=0).mean() for i in range(nwin)])
    quiet_idx = np.argsort(wvar)[:max(1, nwin // 4)]

    # 白噪声: 静默窗去线性趋势后的残差 std
    acc, cnt = 0.0, 0
    for i in quiet_idx:
        seg = sig[i * win:(i + 1) * win]
        x = np.arange(win)
        for c in range(3):
            p = np.polyfit(x, seg[:, c], 1)
            r = seg[:, c] - np.polyval(p, x)
            acc += float(r.std()) ** 2
            cnt += 1
    noise_sigma = float(np.sqrt(acc / max(cnt, 1)))

    # 零漂: 静默窗均值的最大偏离 / 3
    quiet_means = np.array([sig[i * win:(i + 1) * win].mean(axis=0) for i in quiet_idx])
    drift_sigma = float(np.percentile(np.abs(quiet_means).max(axis=0), 90) / 3.0) if len(quiet_means) else 0.0

    # ADC 量化: 所有输出数值排序后的最小间隔 (1 分位, 抗异常)
    vals = np.unique(np.round(sig, 9))
    lsb = 0.0
    bits_est = None
    if len(vals) > 8:
        gaps = np.diff(vals)
        gaps = gaps[gaps > 0]
        if len(gaps):
            lsb = float(np.percentile(gaps, 1))
    full_scale = float(np.abs(sig).max()) * 1.2
    if lsb > 0 and full_scale > 0:
        est = np.log2(2.0 * full_scale / lsb)
        if 6 <= est <= 24:
            bits_est = int(round(est))

    # 带宽: 静默段噪声自相关指数衰减常数
    bws = []
    for i in quiet_idx:
        seg = sig[i * win:(i + 1) * win]
        for c in range(3):
            b = autocorr_bandwidth(seg[:, c], fs)
            if b is not None:
                bws.append(b)
    bandwidth = float(np.median(bws)) if bws else None

    # 串扰矩阵 (需要参考列): 逐轴最小二乘 F_meas[i] = M[i,:] @ F_ref
    crosstalk = np.eye(3)
    ct_note = "未提供参考列 -> 保留单位阵, 请手动设置"
    if ref is not None:
        ct_note = "由参考列最小二乘拟合"
        for i in range(3):
            coef, *_ = np.linalg.lstsq(ref, sig[:, i], rcond=None)
            crosstalk[i] = coef

    cfg = {
        "enabled": True,
        "sample_rate": round(fs, 1),
        "adc_bits": bits_est if bits_est else 12,
        "full_scale": round(full_scale, 2) if full_scale > 0 else 50.0,
        "noise_sigma": round(noise_sigma, 5),
        "noise_relative": 0.002,
        "drift_sigma": round(drift_sigma, 4),
        "drift_tau": 60.0,
        "bandwidth": round(max(bandwidth, 0.5), 1) if bandwidth else 50.0,
        "crosstalk": [[round(v, 4) for v in row] for row in crosstalk],
        "hysteresis_frac": 0.02,
        "creep_gain": 0.02,
        "creep_tau": 8.0,
        "deadband": round(min(noise_sigma / 3.0, 0.01), 5),
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)

    print()
    print("=" * 56)
    print("  拟合结果 -> " + out)
    print("=" * 56)
    print(f"  采样率      {cfg['sample_rate']} Hz")
    print(f"  ADC 位数    {cfg['adc_bits']} bit"
          + ("" if bits_est else "   [未检出量化台阶, 用默认值]"))
    print(f"  量程        ±{cfg['full_scale']} N"
          + (f"  (LSB={lsb:.5f}N)" if lsb > 0 else ""))
    print(f"  白噪声 σ    {cfg['noise_sigma']} N")
    print(f"  零漂 σ      {cfg['drift_sigma']} N (τ=60s)")
    print(f"  带宽        {cfg['bandwidth']} Hz"
          + ("" if bws else "   [静默段太短, 用默认值]"))
    print(f"  串扰矩阵    {ct_note}")
    for row in cfg["crosstalk"]:
        print("              " + str(row))
    print(f"  死区        {cfg['deadband']} N")
    print()
    print("  蠕变/迟滞未自动估计 (需恒载平台与加卸载循环的录制谱):")
    print("    恒载下读数30s内缓升x% -> creep_gain=x/100")
    print("    加卸载同力点读数差y%  -> hysteresis_frac=y/100")
    print("    手动改 signal_config.json 后 GET /api/signal?k=__reload__ 生效")


if __name__ == "__main__":
    main()
