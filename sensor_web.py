import serial
import struct
import time
import math
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import sys

# ============================================================
# 传感器驱动
# ============================================================
class TactileSensor:
    HEADER = bytes([0xB5, 0xA5, 0x55])
    CMD_STOP = 0xF0
    CMD_STANDARD = 0xB1
    CMD_DYNAMIC = 0xB2
    CMD_ANTI_MAGNETIC = 0xB3
    FREQ_1000HZ = 0xA1
    FREQ_500HZ = 0xA2
    FREQ_100HZ = 0xA3
    FREQ_10HZ = 0xA4
    
    MODEL_NAMES = {
        0x00: "初始化/校准中",
        0x11: "0x11 标准型",
        0x12: "0x12 高精度型",
        0x13: "0x13 高灵敏型",
        0x14: "0x14 超量程型",
    }
    
    def __init__(self, port='COM3', baudrate=921600):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.buffer = b''
        self.lock = threading.Lock()
        self.running = False
        
        # 数据
        self.fx = 0.0
        self.fy = 0.0
        self.fz = 0.0
        self.model_id = 0
        self.board_id = 0
        self.status = 0
        
        # 统计
        self.sample_count = 0
        self.fps = 0
        self.peak_force = 0.0
        self.start_time = 0
        
        # 校准偏移
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_z = 0.0
        
    def connect(self):
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1
            )
            time.sleep(0.3)
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False
    
    def disconnect(self):
        self.running = False
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(bytes([self.CMD_STOP]))
            except:
                pass
            self.ser.close()
    
    def send_cmd(self, cmd):
        if self.ser and self.ser.is_open:
            self.ser.write(bytes([cmd]))
    
    def crc8(self, data):
        crc = 0
        for b in data:
            crc ^= b
        return crc
    
    def start(self):
        # 初始化传感器
        self.send_cmd(self.CMD_STOP)
        time.sleep(0.3)
        self.send_cmd(self.FREQ_100HZ)
        time.sleep(0.1)
        self.send_cmd(self.CMD_STANDARD)
        time.sleep(0.5)
        
        self.start_time = time.time()
        self.running = True
        
        t = threading.Thread(target=self._read_loop, daemon=True)
        t.start()
        return True
    
    def _read_loop(self):
        fps_counter = 0
        last_fps_time = time.time()
        
        while self.running:
            try:
                # 读取数据
                if self.ser and self.ser.in_waiting > 0:
                    self.buffer += self.ser.read(self.ser.in_waiting)
                
                # 解析帧
                while True:
                    idx = self.buffer.find(self.HEADER)
                    if idx < 0:
                        break
                    
                    if idx > 0:
                        self.buffer = self.buffer[idx:]
                    
                    if len(self.buffer) < 5:
                        break
                    
                    pkt_len = self.buffer[3] | (self.buffer[4] << 8)
                    
                    if pkt_len < 20 or pkt_len > 50:
                        self.buffer = self.buffer[1:]
                        continue
                    
                    if len(self.buffer) < pkt_len:
                        break
                    
                    frame = self.buffer[:pkt_len]
                    self.buffer = self.buffer[pkt_len:]
                    
                    # CRC校验
                    if frame[-1] != self.crc8(frame[:-1]):
                        continue
                    
                    mid = frame[5]
                    if mid == 0:
                        continue  # 跳过校准帧
                    
                    # 解析float (小端)
                    def parse_float(offset):
                        import struct
                        return struct.unpack_from('<f', frame, offset)[0]
                    
                    x = parse_float(8)
                    y = parse_float(12)
                    z = parse_float(16)
                    
                    with self.lock:
                        self.model_id = mid
                        self.board_id = frame[6]
                        self.status = frame[7]
                        self.fx = x - self.offset_x
                        self.fy = y - self.offset_y
                        self.fz = z - self.offset_z
                        self.sample_count += 1
                        fps_counter += 1
                        
                        mag = math.sqrt(self.fx**2 + self.fy**2 + self.fz**2)
                        if mag > self.peak_force:
                            self.peak_force = mag
                
                # FPS
                now = time.time()
                if now - last_fps_time >= 1.0:
                    self.fps = fps_counter
                    fps_counter = 0
                    last_fps_time = now
                    
            except Exception as e:
                time.sleep(0.01)
            
            time.sleep(0.001)
    
    def get_data(self):
        with self.lock:
            mag = math.sqrt(self.fx**2 + self.fy**2 + self.fz**2)
            if mag > 1e-6:
                theta = math.degrees(math.acos(max(-1, min(1, self.fz / mag))))
                phi = math.degrees(math.atan2(self.fy, self.fx))
            else:
                theta = 0.0
                phi = 0.0
            
            return {
                'fx': round(self.fx, 6),
                'fy': round(self.fy, 6),
                'fz': round(self.fz, 6),
                'mag': round(mag, 6),
                'theta': round(theta, 2),
                'phi': round(phi, 2),
                'modelId': self.model_id,
                'modelName': self.MODEL_NAMES.get(self.model_id, f"0x{self.model_id:02X}"),
                'boardId': self.board_id,
                'status': self.status,
                'sampleCount': self.sample_count,
                'fps': self.fps,
                'peakForce': round(self.peak_force, 6),
                'elapsed': round(time.time() - self.start_time, 1)
            }
    
    def tare(self):
        with self.lock:
            self.offset_x += self.fx
            self.offset_y += self.fy
            self.offset_z += self.fz
            self.peak_force = 0.0
    
    def set_mode(self, mode):
        self.send_cmd(self.CMD_STOP)
        time.sleep(0.2)
        self.send_cmd(mode)
        with self.lock:
            self.offset_x = 0
            self.offset_y = 0
            self.offset_z = 0
            self.peak_force = 0.0
    
    def set_freq(self, freq):
        self.send_cmd(self.CMD_STOP)
        time.sleep(0.1)
        self.send_cmd(freq)
        time.sleep(0.1)
        self.send_cmd(self.CMD_STANDARD)
        with self.lock:
            self.peak_force = 0.0


# ============================================================
# HTTP 处理
# ============================================================
sensor = None
cur_mode = 'B1'
cur_freq = 'A3'

class Handler(BaseHTTPRequestHandler):
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self._send_html()
        elif self.path == '/api/data':
            self._send_json(sensor.get_data() if sensor else {})
        elif self.path == '/api/tare':
            if sensor: sensor.tare()
            self._send_json({'ok': True})
        elif self.path.startswith('/api/mode/'):
            m = self.path.split('/')[-1].upper()
            modes = {'B1': 0xB1, 'B2': 0xB2, 'B3': 0xB3}
            if m in modes and sensor:
                sensor.set_mode(modes[m])
                global cur_mode
                cur_mode = m
            self._send_json({'ok': True})
        elif self.path.startswith('/api/freq/'):
            f = self.path.split('/')[-1].upper()
            freqs = {'A0': 0xA0, 'A1': 0xA1, 'A2': 0xA2, 'A3': 0xA3, 'A4': 0xA4}
            if f in freqs and sensor:
                sensor.set_freq(freqs[f])
                global cur_freq
                cur_freq = f
            self._send_json({'ok': True})
        else:
            self.send_error(404)
    
    def _send_html(self):
        html = HTML_PAGE.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', len(html))
        self.end_headers()
        self.wfile.write(html)
    
    def _send_json(self, data):
        j = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(j))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(j)
    
    def log_message(self, fmt, *args):
        pass  # 静默


# ============================================================
# HTML 页面
# ============================================================
HTML_PAGE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>触觉传感器 - 三维力实时显示</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Microsoft YaHei','PingFang SC',sans-serif;background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);color:#e2e8f0;min-height:100vh;padding:16px}
.header{text-align:center;margin-bottom:16px}
.header h1{font-size:24px;background:linear-gradient(90deg,#667eea,#f093fb);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.status-bar{display:flex;justify-content:center;gap:24px;margin-bottom:16px;font-size:13px;color:#94a3b8;flex-wrap:wrap}
.status-item{display:flex;align-items:center;gap:6px}
.status-dot{width:8px;height:8px;border-radius:50%;background:#22c55e;box-shadow:0 0 8px rgba(34,197,94,.6);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:1200px;margin:0 auto}
.card{background:rgba(255,255,255,.05);backdrop-filter:blur(10px);border-radius:12px;padding:16px;border:1px solid rgba(255,255,255,.1)}
.card-title{font-size:14px;font-weight:600;margin-bottom:12px;color:#f1f5f9;display:flex;align-items:center;gap:8px}
.card-title::before{content:'';width:3px;height:14px;background:linear-gradient(180deg,#667eea,#764ba2);border-radius:2px}
#canvas3d{width:100%;height:300px;border-radius:8px;background:radial-gradient(ellipse at center,rgba(30,41,59,.6),rgba(15,23,42,.8))}
.force-values{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}
.force-item{text-align:center;padding:8px;background:rgba(255,255,255,.03);border-radius:8px}
.force-label{font-size:11px;color:#64748b;margin-bottom:2px}
.force-value{font-size:18px;font-weight:700;font-family:Consolas,monospace}
.fx{color:#60a5fa}.fy{color:#4ade80}.fz{color:#f472b6}
.chart-container{height:200px}
#timeChart{width:100%;height:100%}
.legend{display:flex;justify-content:center;gap:16px;margin-top:8px;font-size:11px;color:#94a3b8}
.legend-item{display:flex;align-items:center;gap:4px}
.legend-color{width:12px;height:2px;border-radius:1px}
.mag-display{text-align:center;padding:16px 0}
.mag-value{font-size:48px;font-weight:800;font-family:Consolas,monospace;background:linear-gradient(135deg,#f093fb,#f5576c);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.mag-unit{font-size:16px;color:#94a3b8;margin-left:4px}
.mag-label{font-size:12px;color:#64748b;margin-top:4px}
.bars{margin-top:16px}
.bar-row{display:flex;align-items:center;margin-bottom:8px;gap:8px}
.bar-label{width:28px;font-size:12px;font-weight:600;text-align:right}
.bar-track{flex:1;height:18px;background:rgba(255,255,255,.06);border-radius:9px;position:relative;overflow:hidden}
.bar-zero{position:absolute;left:50%;top:0;bottom:0;width:2px;background:rgba(255,255,255,.2);transform:translateX(-1px)}
.bar-fill{height:100%;position:absolute;transition:width .08s ease-out;border-radius:9px}
.bar-fill.pos{left:50%;background:linear-gradient(90deg,rgba(102,126,234,.4),#667eea)}
.bar-fill.neg{right:50%;background:linear-gradient(270deg,rgba(245,87,108,.4),#f5576c)}
.bar-val{width:60px;font-size:11px;font-family:Consolas,monospace;color:#94a3b8;text-align:right}
.angle-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
.angle-item{text-align:center;padding:12px;background:rgba(255,255,255,.03);border-radius:8px}
.angle-value{font-size:28px;font-weight:700;font-family:Consolas,monospace}
.theta{color:#a78bfa}.phi{color:#38bdf8}
.angle-label{font-size:11px;color:#64748b;margin-top:2px}
.polar-wrap{display:flex;justify-content:center}
.polar-svg{width:160px;height:160px}
.controls{margin-top:12px}
.control-row{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
.ctrl-btn{padding:6px 12px;border-radius:6px;font-size:11px;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.15);color:#cbd5e0;cursor:pointer;transition:all .15s}
.ctrl-btn:hover{background:rgba(255,255,255,.12)}
.ctrl-btn.active{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;border-color:transparent}
.ctrl-btn.tare{background:rgba(34,197,94,.2);border-color:rgba(34,197,94,.4);color:#4ade80;width:100%;padding:10px;font-size:13px;font-weight:600}
.ctrl-btn.tare:hover{background:rgba(34,197,94,.3)}
.ctrl-btn.pause{background:rgba(251,191,36,.15);border-color:rgba(251,191,36,.4);color:#fbbf24;width:100%;padding:10px;font-size:13px;font-weight:600;margin-top:8px}
.ctrl-btn.pause:hover{background:rgba(251,191,36,.25)}
.ctrl-btn.pause.paused{background:rgba(239,68,68,.2);border-color:rgba(239,68,68,.4);color:#f87171}
.sensitivity{margin-top:12px}
.sensitivity-label{display:flex;justify-content:space-between;font-size:11px;color:#94a3b8;margin-bottom:6px}
.sensitivity-val{color:#fbbf24;font-weight:600;font-family:Consolas,monospace}
.sensitivity-slider{-webkit-appearance:none;appearance:none;width:100%;height:6px;border-radius:3px;background:rgba(255,255,255,.1);outline:none}
.sensitivity-slider::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:16px;height:16px;border-radius:50%;background:linear-gradient(135deg,#fbbf24,#f59e0b);cursor:pointer;box-shadow:0 0 8px rgba(251,191,36,.5)}
.sensitivity-slider::-moz-range-thumb{width:16px;height:16px;border-radius:50%;background:linear-gradient(135deg,#fbbf24,#f59e0b);cursor:pointer;border:none;box-shadow:0 0 8px rgba(251,191,36,.5)}
.info-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:11px;color:#94a3b8;margin-top:8px}
.info-grid span{color:#e2e8f0;font-family:Consolas,monospace}
@media(max-width:800px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="header"><h1>🧲 触觉传感器 - 三维力实时显示</h1></div>
<div class="status-bar">
<div class="status-item"><div class="status-dot"></div><span id="statusText">连接中...</span></div>
<div class="status-item">📊 <span id="fpsVal">0</span> Hz</div>
<div class="status-item">🔢 <span id="sampleVal">0</span></div>
<div class="status-item">⏱️ <span id="timeVal">0.0</span>s</div>
<div class="status-item">📈 峰值: <span id="peakVal">0.000</span> N</div>
</div>
<div class="grid">
<div class="card">
<div class="card-title">三维力向量</div>
<canvas id="canvas3d"></canvas>
<div class="force-values">
<div class="force-item"><div class="force-label">Fx (X方向)</div><div class="force-value fx" id="fxVal">0.000</div></div>
<div class="force-item"><div class="force-label">Fy (Y方向)</div><div class="force-value fy" id="fyVal">0.000</div></div>
<div class="force-item"><div class="force-label">Fz (Z方向)</div><div class="force-value fz" id="fzVal">0.000</div></div>
</div>
</div>
<div class="card">
<div class="card-title">力分量时序</div>
<div class="chart-container"><canvas id="timeChart"></canvas></div>
<div class="legend">
<div class="legend-item"><div class="legend-color" style="background:#60a5fa"></div>Fx</div>
<div class="legend-item"><div class="legend-color" style="background:#4ade80"></div>Fy</div>
<div class="legend-item"><div class="legend-color" style="background:#f472b6"></div>Fz</div>
<div class="legend-item"><div class="legend-color" style="background:#fbbf24"></div>|F|</div>
</div>
</div>
<div class="card">
<div class="card-title">合力大小</div>
<div class="mag-display">
<span class="mag-value" id="magVal">0.000</span><span class="mag-unit">N</span>
<div class="mag-label">合力 |F| = √(Fx² + Fy² + Fz²)</div>
</div>
<div class="bars">
<div class="bar-row"><div class="bar-label" style="color:#60a5fa">Fx</div><div class="bar-track"><div class="bar-zero"></div><div class="bar-fill pos" id="barFxP" style="width:0%"></div><div class="bar-fill neg" id="barFxN" style="width:0%"></div></div><div class="bar-val" id="barFxV">0.00</div></div>
<div class="bar-row"><div class="bar-label" style="color:#4ade80">Fy</div><div class="bar-track"><div class="bar-zero"></div><div class="bar-fill pos" id="barFyP" style="width:0%"></div><div class="bar-fill neg" id="barFyN" style="width:0%"></div></div><div class="bar-val" id="barFyV">0.00</div></div>
<div class="bar-row"><div class="bar-label" style="color:#f472b6">Fz</div><div class="bar-track"><div class="bar-zero"></div><div class="bar-fill pos" id="barFzP" style="width:0%"></div><div class="bar-fill neg" id="barFzN" style="width:0%"></div></div><div class="bar-val" id="barFzV">0.00</div></div>
</div>
</div>
<div class="card">
<div class="card-title">力方向与控制</div>
<div class="angle-grid">
<div class="angle-item"><div class="angle-value theta"><span id="thetaVal">0.0</span>°</div><div class="angle-label">俯仰角 θ (与Z轴夹角)</div></div>
<div class="angle-item"><div class="angle-value phi"><span id="phiVal">0.0</span>°</div><div class="angle-label">偏航角 φ (XY平面内)</div></div>
</div>
<div class="polar-wrap">
<svg class="polar-svg" viewBox="-100 -100 200 200">
<circle cx="0" cy="0" r="90" fill="none" stroke="rgba(255,255,255,.08)" stroke-width="1"/>
<circle cx="0" cy="0" r="60" fill="none" stroke="rgba(255,255,255,.08)" stroke-width="1"/>
<circle cx="0" cy="0" r="30" fill="none" stroke="rgba(255,255,255,.08)" stroke-width="1"/>
<line x1="-90" y1="0" x2="90" y2="0" stroke="rgba(255,255,255,.12)" stroke-width="1"/>
<line x1="0" y1="-90" x2="0" y2="90" stroke="rgba(255,255,255,.12)" stroke-width="1"/>
<text x="82" y="4" fill="#60a5fa" font-size="10" text-anchor="end">X+</text>
<text x="-82" y="4" fill="#60a5fa" font-size="10" text-anchor="start">X-</text>
<text x="4" y="-82" fill="#4ade80" font-size="10" text-anchor="middle">Y+</text>
<text x="4" y="88" fill="#4ade80" font-size="10" text-anchor="middle">Y-</text>
<line id="polarArrow" x1="0" y1="0" x2="0" y2="0" stroke="#fbbf24" stroke-width="3" stroke-linecap="round"/>
<circle id="polarDot" cx="0" cy="0" r="6" fill="#fbbf24" style="filter:drop-shadow(0 0 6px rgba(251,191,36,.6))"/>
</svg>
</div>
<div class="controls">
<button class="ctrl-btn tare" onclick="doTare()">⚖️ 清零校准 (Tare)</button>
<button class="ctrl-btn pause" id="pauseBtn" onclick="togglePause()">⏸️ 暂停显示</button>
<div class="sensitivity">
<div class="sensitivity-label"><span>📐 显示灵敏度</span><span class="sensitivity-val" id="sensVal">2.0x</span></div>
<input type="range" class="sensitivity-slider" id="sensSlider" min="0.5" max="10" step="0.1" value="2" oninput="setSensitivity(this.value)">
</div>
<div style="font-size:11px;color:#64748b;margin-top:12px">工作模式</div>
<div class="control-row" id="modeRow">
<button class="ctrl-btn active" data-mode="B1" onclick="setMode('B1')">标准</button>
<button class="ctrl-btn" data-mode="B2" onclick="setMode('B2')">动态</button>
<button class="ctrl-btn" data-mode="B3" onclick="setMode('B3')">抗磁</button>
</div>
<div style="font-size:11px;color:#64748b;margin-top:10px">输出频率</div>
<div class="control-row" id="freqRow">
<button class="ctrl-btn" data-freq="A1" onclick="setFreq('A1')">1000Hz</button>
<button class="ctrl-btn active" data-freq="A3" onclick="setFreq('A3')">100Hz</button>
<button class="ctrl-btn" data-freq="A4" onclick="setFreq('A4')">10Hz</button>
</div>
</div>
<div class="info-grid">
<div>传感器: <span id="infoModel">-</span></div>
<div>节点号: <span id="infoBoard">-</span></div>
<div>状态位: <span id="infoStatus">0x00</span></div>
<div>端口: <span>COM3</span></div>
</div>
</div>
</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script>
let scene,camera,renderer,forceArrow,xArrow,yArrow,zArrow,forceSphere;
let displayRange=5.0;
let sensitivity=2.0;
let paused=false;
let timeChart;
const HLEN=150;
let tData=[],fxData=[],fyData=[],fzData=[],magData=[];
let startT=Date.now();

function init3D(){
  const c=document.getElementById('canvas3d');
  const w=c.clientWidth,h=c.clientHeight;
  scene=new THREE.Scene();
  camera=new THREE.PerspectiveCamera(45,w/h,0.1,1000);
  camera.position.set(3,2.5,3);
  camera.lookAt(0,0,0);
  renderer=new THREE.WebGLRenderer({canvas:c,antialias:true,alpha:true});
  renderer.setSize(w,h);
  renderer.setPixelRatio(window.devicePixelRatio);
  scene.add(new THREE.AxesHelper(2));
  const g=new THREE.GridHelper(4,20,0x333333,0x222222);
  g.position.y=-0.01;scene.add(g);
  scene.add(new THREE.AmbientLight(0xffffff,0.6));
  const dl=new THREE.DirectionalLight(0xffffff,0.8);
  dl.position.set(5,10,7);scene.add(dl);
  xArrow=new THREE.ArrowHelper(new THREE.Vector3(1,0,0),new THREE.Vector3(0,0,0),0.1,0x60a5fa,0.1,0.06);
  yArrow=new THREE.ArrowHelper(new THREE.Vector3(0,1,0),new THREE.Vector3(0,0,0),0.1,0x4ade80,0.1,0.06);
  zArrow=new THREE.ArrowHelper(new THREE.Vector3(0,0,1),new THREE.Vector3(0,0,0),0.1,0xf472b6,0.1,0.06);
  scene.add(xArrow);scene.add(yArrow);scene.add(zArrow);
  forceArrow=new THREE.ArrowHelper(new THREE.Vector3(1,1,1).normalize(),new THREE.Vector3(0,0,0),0.1,0xfbbf24,0.15,0.08);
  scene.add(forceArrow);
  forceSphere=new THREE.Mesh(new THREE.SphereGeometry(0.06,16,16),new THREE.MeshPhongMaterial({color:0xfbbf24,emissive:0xf59e0b,emissiveIntensity:0.3}));
  scene.add(forceSphere);
  window.addEventListener('resize',()=>{const w2=c.clientWidth,h2=c.clientHeight;camera.aspect=w2/h2;camera.updateProjectionMatrix();renderer.setSize(w2,h2)});
  anim3D();
}
function anim3D(){
  requestAnimationFrame(anim3D);
  const t=Date.now()*0.0003;
  camera.position.x=Math.cos(t)*3.5;
  camera.position.z=Math.sin(t)*3.5;
  camera.position.y=2.5;
  camera.lookAt(0,0,0);
  renderer.render(scene,camera);
}
function upd3D(fx,fy,fz){
  const s=1.5/Math.max(displayRange,0.1)*sensitivity;
  if(Math.abs(fx)>0.001){xArrow.setDirection(new THREE.Vector3(Math.sign(fx),0,0));xArrow.setLength(Math.abs(fx)*s,0.1,0.06);xArrow.visible=true}else xArrow.visible=false;
  if(Math.abs(fy)>0.001){yArrow.setDirection(new THREE.Vector3(0,Math.sign(fy),0));yArrow.setLength(Math.abs(fy)*s,0.1,0.06);yArrow.visible=true}else yArrow.visible=false;
  if(Math.abs(fz)>0.001){zArrow.setDirection(new THREE.Vector3(0,0,Math.sign(fz)));zArrow.setLength(Math.abs(fz)*s,0.1,0.06);zArrow.visible=true}else zArrow.visible=false;
  const m=Math.sqrt(fx*fx+fy*fy+fz*fz);
  if(m>0.001){
    const d=new THREE.Vector3(fx,fy,fz).normalize();
    forceArrow.setDirection(d);forceArrow.setLength(m*s,0.15,0.08);forceArrow.visible=true;
    forceSphere.position.copy(d.clone().multiplyScalar(m*s));forceSphere.visible=true;
  }else{forceArrow.visible=false;forceSphere.visible=false}
}
function initChart(){
  const ctx=document.getElementById('timeChart').getContext('2d');
  timeChart=new Chart(ctx,{
    type:'line',
    data:{labels:[],datasets:[
      {label:'Fx',data:[],borderColor:'#60a5fa',borderWidth:1.5,pointRadius:0,tension:0.2},
      {label:'Fy',data:[],borderColor:'#4ade80',borderWidth:1.5,pointRadius:0,tension:0.2},
      {label:'Fz',data:[],borderColor:'#f472b6',borderWidth:1.5,pointRadius:0,tension:0.2},
      {label:'|F|',data:[],borderColor:'#fbbf24',borderWidth:1.5,borderDash:[5,5],pointRadius:0,tension:0.2}
    ]},
    options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{display:false}},
      scales:{x:{display:true,grid:{color:'rgba(255,255,255,.05)'},ticks:{color:'#64748b',maxTicksLimit:6,callback:v=>v.toFixed(1)+'s'}},
              y:{display:true,grid:{color:'rgba(255,255,255,.05)'},ticks:{color:'#64748b',callback:v=>v.toFixed(1)+'N'}}}}
  });
}
function updBars(fx,fy,fz){
  const set=(ax,v)=>{
    const r=Math.min(Math.abs(v)/Math.max(displayRange,0.1)*sensitivity,1)*50;
    const p=document.getElementById('bar'+ax+'P');
    const n=document.getElementById('bar'+ax+'N');
    const val=document.getElementById('bar'+ax+'V');
    if(v>=0){p.style.width=r+'%';n.style.width='0%'}else{p.style.width='0%';n.style.width=r+'%'}
    val.textContent=v.toFixed(2)+'N';
  };
  set('Fx',fx);set('Fy',fy);set('Fz',fz);
}
function updPolar(fx,fy){
  const fxy=Math.sqrt(fx*fx+fy*fy);
  const r=Math.min(fxy/Math.max(displayRange,0.1)*sensitivity,1)*85;
  const pr=Math.atan2(fy,fx);
  const x=r*Math.cos(pr),y=-r*Math.sin(pr);
  document.getElementById('polarArrow').setAttribute('x2',x.toFixed(1));
  document.getElementById('polarArrow').setAttribute('y2',y.toFixed(1));
  document.getElementById('polarDot').setAttribute('cx',x.toFixed(1));
  document.getElementById('polarDot').setAttribute('cy',y.toFixed(1));
}
async function getData(){try{const r=await fetch('/api/data');return await r.json()}catch(e){return null}}
async function doTare(){try{await fetch('/api/tare')}catch(e){}}
async function setMode(m){
  try{await fetch('/api/mode/'+m);
  document.querySelectorAll('#modeRow .ctrl-btn').forEach(b=>b.classList.toggle('active',b.dataset.mode===m))}catch(e){}
}
async function setFreq(f){
  try{await fetch('/api/freq/'+f);
  document.querySelectorAll('#freqRow .ctrl-btn').forEach(b=>b.classList.toggle('active',b.dataset.freq===f))}catch(e){}
}
function togglePause(){
  paused=!paused;
  const btn=document.getElementById('pauseBtn');
  if(paused){btn.textContent='▶️ 继续显示';btn.classList.add('paused')}
  else{btn.textContent='⏸️ 暂停显示';btn.classList.remove('paused')}
}
function setSensitivity(v){
  sensitivity=parseFloat(v);
  document.getElementById('sensVal').textContent=sensitivity.toFixed(1)+'x';
}
async function loop(){
  const d=await getData();
  if(d&&d.fps>0){
    document.getElementById('statusText').textContent=paused?'已连接 (已暂停)':'已连接';
    document.getElementById('fpsVal').textContent=d.fps;
    document.getElementById('sampleVal').textContent=d.sampleCount;
    document.getElementById('timeVal').textContent=d.elapsed.toFixed(1);
    document.getElementById('peakVal').textContent=d.peakForce.toFixed(3);
    if(!paused){
      document.getElementById('fxVal').textContent=d.fx.toFixed(3);
      document.getElementById('fyVal').textContent=d.fy.toFixed(3);
      document.getElementById('fzVal').textContent=d.fz.toFixed(3);
      document.getElementById('magVal').textContent=d.mag.toFixed(3);
      document.getElementById('thetaVal').textContent=d.theta.toFixed(1);
      document.getElementById('phiVal').textContent=d.phi.toFixed(1);
      document.getElementById('infoModel').textContent=d.modelName||'-';
      document.getElementById('infoBoard').textContent=d.boardId;
      document.getElementById('infoStatus').textContent='0x'+d.status.toString(16).toUpperCase().padStart(2,'0');
      upd3D(d.fx,d.fy,d.fz);
      updBars(d.fx,d.fy,d.fz);
      updPolar(d.fx,d.fy);
      const mx=Math.max(Math.abs(d.fx),Math.abs(d.fy),Math.abs(d.fz),d.mag);
      if(mx>displayRange*0.8)displayRange=mx*1.5;
      const t=(Date.now()-startT)/1000;
      tData.push(t);fxData.push(d.fx);fyData.push(d.fy);fzData.push(d.fz);magData.push(d.mag);
      while(tData.length>HLEN){tData.shift();fxData.shift();fyData.shift();fzData.shift();magData.shift()}
      if(timeChart&&tData.length>2){
        timeChart.data.labels=tData;
        timeChart.data.datasets[0].data=fxData;
        timeChart.data.datasets[1].data=fyData;
        timeChart.data.datasets[2].data=fzData;
        timeChart.data.datasets[3].data=magData;
        timeChart.update('none');
      }
    }
  }else{document.getElementById('statusText').textContent='连接中...'}
  setTimeout(loop,50);
}
window.addEventListener('DOMContentLoaded',()=>{init3D();initChart();loop()});
</script>
</body>
</html>'''


# ============================================================
# 主函数
# ============================================================
def main():
    global sensor
    
    print("=" * 60)
    print("  触觉传感器 - 三维力实时显示 (Web版)")
    print("=" * 60)
    print()
    
    sensor = TactileSensor('COM3', 921600)
    
    if not sensor.connect():
        print("✗ 无法连接传感器 (COM3)")
        print("请检查传感器是否正确连接")
        return
    
    try:
        print("  ✓ 传感器连接成功")
        sensor.start()
        print("  ✓ 数据读取已启动")
        print()
        
        # 启动HTTP服务器
        server = HTTPServer(('127.0.0.1', 8765), Handler)
        print("  🌐 Web服务器已启动")
        print(f"  📱 请在浏览器打开: http://127.0.0.1:8765/")
        print()
        print("  按 Ctrl+C 停止程序")
        print("  " + "=" * 56)
        print()
        
        # 尝试打开浏览器
        try:
            import webbrowser
            webbrowser.open('http://127.0.0.1:8765/')
        except:
            pass
        
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\n\n  正在停止...")
    except Exception as e:
        print(f"\n  错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sensor.disconnect()
        print("  传感器已断开连接")


if __name__ == '__main__':
    main()
