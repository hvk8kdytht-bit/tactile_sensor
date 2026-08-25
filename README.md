# 触觉传感器三维力实时显示

基于 UART 串口通信的触觉传感器三维力实时可视化工具。支持多种工作模式和输出频率，提供 Web 端三维力实时显示。

## MuJoCo 触觉传感器仿真

`kinova_sensor_sim.py` — Kinova Gen3 机械臂 + Robotiq 2F-85 夹爪 + 左指垫触觉传感器的物理仿真，在物理真值力之上叠加真实传感器信号特性层。

```bash
pip install mujoco numpy
python kinova_sensor_sim.py
```

浏览器打开 `http://127.0.0.1:8771/`（Web 控制面板）并同时显示 MuJoCo 原生 3D 窗口。

### 三种交互模式

| 模式 | 说明 |
|------|------|
| touch | 网页触控板拖拽，模拟手指按压传感器表面（法向力 + 剪切力，力闭环精确跟踪指令力） |
| grasp | 夹爪闭合抓取薄板，右指垫把物体压向传感器（限速逼近防冲击弹出，35N 夹持力上限） |
| arm | 机械臂 IK 主动伸过去按压场景中的红色目标物（力闭环，撤力后退到目标面外 5mm 真间隙） |

### 信号特性层（SignalEmulator）

物理真值力 → 轴间串扰 → 迟滞（回隙）→ 蠕变 → 带宽低通 → 零漂 OU → 白噪声 → 死区 → ADC 量化限幅 → 采样保持。
参数存于 `signal_config.json`，可用 `fit_signal_params.py` 从真实传感器录制数据自动拟合后注入；Web 面板可实时调节信号参数，并以双曲线对比物理真值 vs 传感器输出。

### 验证（无头，无需界面）

```bash
python probe_fix_verify.py    # 11 项: 三模式力闭环 / 水平保持 / 薄板夹取回归
python test_signal_layer.py   # 20 项: 信号层单元特性 + 三模式集成回归
```

### 模型与主要文件

| 文件 | 说明 |
|------|------|
| `kinova_sensor_sim.py` | 主仿真程序（三种模式 + 信号层 + Web 面板） |
| `mujoco_menagerie/kinova_gen3/gen3_2f85_sensor.xml` | 传感器化夹爪模型（不要直接修改） |
| `mujoco_menagerie/kinova_gen3/gen3_2f85_sensor_scene.xml` | 场景（触控 mocap、薄板、按压目标） |
| `signal_config.json` | 信号特性层参数 |
| `fit_signal_params.py` | 从真实数据拟合信号参数 |

## 功能特性

- 🎯 **三维力向量实时显示** — 使用 Three.js 渲染 3D 力向量，直观展示力的空间方向
- 📈 **力分量时序图** — 实时记录 X/Y/Z 三个方向及合力的变化曲线
- ⚡ **分力条形图** — 双向条形图直观显示各方向受力大小
- 🧭 **极坐标方向图** — XY 平面内力的方向可视化
- ⏸️ **暂停/继续** — 一键暂停显示，方便观察瞬间状态
- 📐 **灵敏度调节** — 0.5x ~ 10x 可调，微小力变化也能清晰观察
- ⚖️ **清零校准** — 一键 tare，消除零点漂移
- 🔄 **多种工作模式** — 标准模式、动态模式、抗磁模式
- 📊 **多档输出频率** — 10Hz / 100Hz / 500Hz / 1000Hz

## 文件说明

| 文件 | 说明 |
|------|------|
| `sensor_web.py` | **推荐使用** — Web 版三维力实时显示（Python 后端 + 浏览器前端） |
| `tactile_sensor.py` | 基础传感器驱动库 |
| `tactile_sensor_3d.html` | 纯浏览器版（使用 Web Serial API，无需 Python 后端） |
| `tactile_sensor_3d.py` | Matplotlib 版 3D 可视化（需要 tkinter） |

## 快速开始

### 方式一：Web 版（推荐）

```bash
# 安装依赖
pip install pyserial

# 运行
python sensor_web.py
```

然后在浏览器中打开：`http://127.0.0.1:8765/`

### 方式二：纯浏览器版

直接在 Chrome/Edge 浏览器中打开 `tactile_sensor_3d.html`，点击"连接传感器"按钮即可。

> 注意：Web Serial API 仅支持 Chrome / Edge / Opera 等 Chromium 内核浏览器。

## 传感器参数

- **端口**: COM3（可在代码中修改）
- **波特率**: 921600
- **数据格式**: 自定义 UART 协议（帧头 0xB5 0xA5 0x55）
- **支持型号**: 高灵敏型 (0x13) 等

## 协议说明

### 帧结构

```
| 帧头(3B) | 保留(1B) | 长度(2B) | MID(1B) | 数据(NB) | CRC8(1B) |
| B5 A5 55 |    xx    |  LSB MSB |   mid   |   ...    |   crc    |
```

### 命令字

| 命令 | 值 | 说明 |
|------|-----|------|
| 停止 | 0xF0 | 停止输出 |
| 标准模式 | 0xB1 | 标准测力模式 |
| 动态模式 | 0xB2 | 高速动态模式 |
| 抗磁模式 | 0xB3 | 抗磁干扰模式 |
| 1000Hz | 0xA1 | 输出频率 1000Hz |
| 500Hz | 0xA2 | 输出频率 500Hz |
| 100Hz | 0xA3 | 输出频率 100Hz |
| 10Hz | 0xA4 | 输出频率 10Hz |

## 使用技巧

1. **首次使用** — 先点击"清零校准"消除零点偏移
2. **观察微小力** — 将灵敏度调到 5x~10x，轻微触碰即可看到明显变化
3. **捕捉瞬间** — 看到感兴趣的瞬间按暂停，慢慢分析
4. **动态测量** — 切换到动态模式和更高频率（500Hz/1000Hz）

## 许可证

MIT License
