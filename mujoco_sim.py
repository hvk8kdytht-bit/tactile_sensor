"""
MuJoCo 触觉传感器仿真 - 三维力提取与可视化
传感器尺寸: 17mm x 17mm x 5.2mm
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
import json
import http.server
import socketserver
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "mujoco_menagerie", "kinova_gen3", "gen3_sensor.xml")

class SimSensor:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)

        self.fx = 0.0
        self.fy = 0.0
        self.fz = 0.0
        self.peak_force = 0.0
        self.running = False
        self.start_time = 0.0
        self.sample_count = 0
        self.fps = 0

        self.tare_offset = np.array([0.0, 0.0, 0.0])
        self.noise_sigma = 0.003
        self.zero_drift_rate = 0.0001
        self.tare_offset_drift = np.array([0.0, 0.0, 0.0])

        self.press_force = 0.0
        self.press_angle_x = 0.0
        self.press_angle_y = 0.0
        self.auto_mode = False
        self.auto_t = 0.0

        self.sensor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sensor_pad")
        self.wall_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "contact_wall")
        self.act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint_{i}")
                        for i in range(1, 8)]
        key = self.model.key("home")
        self.home_qpos = key.qpos.copy()
        self.data.qpos[:] = self.home_qpos
        self.data.ctrl[:] = key.ctrl
        mujoco.mj_forward(self.model, self.data)
        bias = self.data.qfrc_bias.copy()
        self.home_ctrl = key.ctrl.copy()
        for aid in self.act_ids:
            jid = self.model.actuator_trnid[aid][0]
            dof = self.model.jnt_dofadr[jid]
            kp = self.model.actuator_gainprm[aid][0]
            self.home_ctrl[aid] += bias[dof] / kp
        for _ in range(2):
            self.data.qpos[:] = self.home_qpos
            self.data.qvel[:] = 0
            self.data.ctrl[:] = self.home_ctrl
            for _ in range(1500):
                mujoco.mj_step(self.model, self.data)
            err = self.data.qpos[:7] - self.home_qpos[:7]
            for k, aid in enumerate(self.act_ids):
                self.home_ctrl[aid] += err[k]
        self.data.qpos[:] = self.home_qpos
        self.data.qvel[:] = 0
        self.data.ctrl[:] = self.home_ctrl
        mujoco.mj_forward(self.model, self.data)

    def get_contact_forces(self):
        f_world = np.zeros(3)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            if {g1, g2} != {self.sensor_geom_id, self.wall_geom_id}:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            frame = con.frame.reshape(3, 3)
            f_world += frame.T @ force[:3]

        R = self.data.geom_xmat[self.sensor_geom_id].reshape(3, 3)
        f_local = R.T @ f_world
        return f_local[0], f_local[1], -f_local[2]

    def apply_noise(self, fx, fy, fz):
        elapsed = time.time() - self.start_time if self.start_time > 0 else 0
        self.tare_offset_drift += np.random.randn(3) * self.zero_drift_rate * 0.01

        noise = np.random.randn(3) * self.noise_sigma
        fx += noise[0] + self.tare_offset_drift[0]
        fy += noise[1] + self.tare_offset_drift[1]
        fz += noise[2] + self.tare_offset_drift[2]

        return fx, fy, fz

    def tare(self):
        self.tare_offset = np.array([self.fx, self.fy, self.fz])
        self.tare_offset_drift = np.array([0.0, 0.0, 0.0])

    def update_press(self):
        if self.auto_mode:
            self.auto_t += 0.005
            force = (np.sin(self.auto_t * 2) * 0.5 + 0.5) * 6.0
            angle_x = np.sin(self.auto_t * 0.7) * 30
            angle_y = np.cos(self.auto_t * 0.5) * 30
        else:
            force = self.press_force
            angle_x = self.press_angle_x
            angle_y = self.press_angle_y

        depth = force / 5000.0
        shear_z = np.sin(np.radians(angle_x)) * 0.0015
        shear_y = np.sin(np.radians(angle_y)) * 0.0015

        b = -depth / 0.354
        ctrl = self.home_ctrl.copy()
        ctrl[self.act_ids[1]] += 0.777 * b
        ctrl[self.act_ids[3]] += b
        ctrl[self.act_ids[0]] += -shear_y / 0.473
        ctrl[self.act_ids[5]] += shear_z / 0.183
        self.data.ctrl[:] = ctrl

    def get_data(self):
        fx = self.fx - self.tare_offset[0]
        fy = self.fy - self.tare_offset[1]
        fz = self.fz - self.tare_offset[2]
        mag = np.sqrt(fx*fx + fy*fy + fz*fz)
        if mag > self.peak_force:
            self.peak_force = mag
        if mag < 0.001:
            theta = 0.0
            phi = 0.0
        else:
            theta = np.degrees(np.arccos(np.clip(fz / mag, -1, 1)))
            phi = np.degrees(np.arctan2(fy, fx))
        return {
            "fx": round(fx, 4), "fy": round(fy, 4), "fz": round(fz, 4),
            "mag": round(mag, 4), "theta": round(theta, 2), "phi": round(phi, 2),
            "fps": self.fps, "sampleCount": self.sample_count,
            "peakForce": round(self.peak_force, 4),
            "elapsed": time.time() - self.start_time if self.start_time > 0 else 0,
            "modelName": "MuJoCo Sim", "boardId": "SIM-013",
            "status": 0x01, "connected": True
        }


HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MuJoCo 触觉传感器仿真 - 三维力实时显示</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0e1a;color:#e2e8f0;font-family:'Segoe UI',system-ui,sans-serif;overflow:hidden;height:100vh}
#app{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;gap:8px;padding:8px;height:100vh}
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
.ctrl{display:flex;flex-direction:column;gap:8px;padding:8px}
.ctrl-row{display:flex;gap:8px;align-items:center}
.ctrl-btn{background:rgba(56,189,248,.15);border:1px solid rgba(56,189,248,.3);color:#7dd3fc;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:600;transition:all .2s}
.ctrl-btn:hover{background:rgba(56,189,248,.25)}
.ctrl-btn.tare{background:rgba(34,197,94,.15);border-color:rgba(34,197,94,.3);color:#4ade80}
.ctrl-btn.pause{background:rgba(251,191,36,.15);border-color:rgba(251,191,36,.3);color:#fbbf24}
.ctrl-btn.pause.paused{background:rgba(239,68,68,.15);border-color:rgba(239,68,68,.3);color:#f87171}
.info-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px}
.info-item{background:rgba(255,255,255,.03);padding:6px 10px;border-radius:6px;font-size:11px}
.info-item span{color:#64748b;font-size:10px;display:block}
.info-item b{color:#e2e8f0;font-family:Consolas,monospace;font-size:14px}
.sens-slider{width:100%;-webkit-appearance:none;height:6px;border-radius:3px;background:rgba(255,255,255,.1);outline:none}
.sens-slider::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:#fbbf24;cursor:pointer}
.sens-val{color:#fbbf24;font-weight:700;font-family:Consolas,monospace}
.status-bar{position:fixed;top:8px;right:12px;z-index:100;display:flex;gap:12px;align-items:center;font-size:11px}
.status-dot{width:8px;height:8px;border-radius:50%;background:#22c55e;box-shadow:0 0 8px #22c55e}
.sim-badge{background:rgba(168,85,247,.2);border:1px solid rgba(168,85,247,.4);color:#c084fc;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700}
#touchpad{position:relative;flex:1;border-radius:8px;border:1px solid rgba(56,189,248,.15);background:radial-gradient(circle at 50% 50%,rgba(56,189,248,.08),rgba(10,14,26,.95));overflow:hidden;cursor:crosshair;touch-action:none;user-select:none;-webkit-user-select:none}
#touchpad:active{border-color:rgba(251,191,36,.45)}
.tp-grid{position:absolute;inset:0;background-image:linear-gradient(rgba(56,189,248,.08) 1px,transparent 1px),linear-gradient(90deg,rgba(56,189,248,.08) 1px,transparent 1px);background-size:12.5% 12.5%;pointer-events:none}
.tp-sensor{position:absolute;left:50%;top:50%;width:26%;aspect-ratio:1;transform:translate(-50%,-50%);background:rgba(34,197,94,.16);border:1px solid rgba(34,197,94,.55);border-radius:3px;box-shadow:0 0 14px rgba(34,197,94,.2);pointer-events:none}
.tp-sensor::after{content:'传感器';position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:10px;color:#4ade80}
.tp-cross-h{position:absolute;left:0;right:0;top:50%;height:1px;background:rgba(255,255,255,.12);pointer-events:none}
.tp-cross-v{position:absolute;top:0;bottom:0;left:50%;width:1px;background:rgba(255,255,255,.12);pointer-events:none}
.tp-ring{position:absolute;left:50%;top:50%;width:92%;aspect-ratio:1;transform:translate(-50%,-50%);border:1px dashed rgba(251,191,36,.25);border-radius:50%;pointer-events:none}
.tp-finger{position:absolute;left:50%;top:50%;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;background:radial-gradient(circle,rgba(251,191,36,.85),rgba(251,191,36,.15));border:2px solid #fbbf24;box-shadow:0 0 10px rgba(251,191,36,.45);pointer-events:none;transition:left .4s cubic-bezier(.2,1.7,.4,1),top .4s cubic-bezier(.2,1.7,.4,1),width .1s,height .1s,margin .1s,border-color .1s,box-shadow .1s}
.tp-finger.pressed{transition:width .08s,height .08s,margin .08s,border-color .08s,box-shadow .08s}
.tp-readout{position:absolute;top:6px;left:10px;font:700 13px Consolas,monospace;color:#fbbf24;pointer-events:none;text-shadow:0 0 8px rgba(0,0,0,.8)}
.tp-hint{position:absolute;bottom:6px;left:0;right:0;text-align:center;font-size:10px;color:#64748b;pointer-events:none}
</style>
</head>
<body>
<div class="status-bar">
  <span class="sim-badge">MuJoCo 仿真</span>
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
    <div class="panel-title">力分量时序图</div>
    <canvas id="timeChart"></canvas>
  </div>
  <div class="panel">
    <div class="panel-title">分力条形图</div>
    <div class="bars" id="barsContainer">
      <div class="bar-row"><span class="bar-label" style="color:#22d3ee">Fx</span><div class="bar-track"><div class="bar-fill bar-pos" id="barFxP"></div><div class="bar-fill bar-neg" id="barFxN"></div><div class="bar-center"></div></div><span class="bar-val" id="barFxV">0.00N</span></div>
      <div class="bar-row"><span class="bar-label" style="color:#f87171">Fy</span><div class="bar-track"><div class="bar-fill bar-pos" id="barFyP"></div><div class="bar-fill bar-neg" id="barFyN"></div><div class="bar-center"></div></div><span class="bar-val" id="barFyV">0.00N</span></div>
      <div class="bar-row"><span class="bar-label" style="color:#a78bfa">Fz</span><div class="bar-track"><div class="bar-fill bar-pos" id="barFzP"></div><div class="bar-fill bar-neg" id="barFzN"></div><div class="bar-center"></div></div><span class="bar-val" id="barFzV">0.00N</span></div>
      <div class="bar-row"><span class="bar-label" style="color:#fbbf24">|F|</span><div class="bar-track"><div class="bar-fill bar-pos" id="barMagP" style="background:linear-gradient(90deg,transparent,#fbbf24)"></div><div class="bar-center"></div></div><span class="bar-val" id="barMagV" style="color:#fbbf24">0.00N</span></div>
    </div>
  </div>
  <div class="panel">
    <div class="panel-title">控制面板</div>
    <div class="ctrl">
      <div class="ctrl-row">
        <button class="ctrl-btn tare" onclick="doTare()">清零校准</button>
        <button class="ctrl-btn pause" id="pauseBtn" onclick="togglePause()">暂停</button>
      </div>
      <div style="margin-top:4px">
        <div style="display:flex;justify-content:space-between;font-size:11px;color:#94a3b8;margin-bottom:4px">
          <span>显示灵敏度</span><span class="sens-val" id="sensVal">3.0x</span>
        </div>
        <input type="range" class="sens-slider" id="sensSlider" min="0.5" max="10" step="0.1" value="3" oninput="setSens(this.value)">
      </div>
      <div style="margin-top:4px">
        <div style="display:flex;justify-content:space-between;font-size:11px;color:#94a3b8;margin-bottom:4px">
          <span>噪声强度</span><span class="sens-val" id="noiseVal" style="color:#a78bfa">0.003N</span>
        </div>
        <input type="range" class="sens-slider" id="noiseSlider" min="0" max="0.02" step="0.001" value="0.003" oninput="setNoise(this.value)" style="--c:#a78bfa">
      </div>
      <div style="margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,.08)">
        <div style="font-size:11px;color:#c084fc;font-weight:600;margin-bottom:6px">机械臂按压控制 (Kinova Gen3)</div>
        <div style="margin-bottom:6px">
          <div style="display:flex;justify-content:space-between;font-size:11px;color:#94a3b8;margin-bottom:3px">
            <span>按压深度</span><span style="color:#c084fc;font-family:Consolas,monospace" id="pressVal">0.0N</span>
          </div>
          <input type="range" class="sens-slider" id="pressSlider" min="0" max="10" step="0.1" value="0" oninput="setPress(this.value)" style="background:rgba(168,85,247,.15)">
        </div>
        <div style="margin-bottom:6px">
          <div style="display:flex;justify-content:space-between;font-size:11px;color:#94a3b8;margin-bottom:3px">
            <span>切向偏移 X</span><span style="color:#c084fc;font-family:Consolas,monospace" id="angleXVal">0°</span>
          </div>
          <input type="range" class="sens-slider" id="angleXSlider" min="-45" max="45" step="1" value="0" oninput="setAngleX(this.value)">
        </div>
        <div style="margin-bottom:6px">
          <div style="display:flex;justify-content:space-between;font-size:11px;color:#94a3b8;margin-bottom:3px">
            <span>切向偏移 Y</span><span style="color:#c084fc;font-family:Consolas,monospace" id="angleYVal">0°</span>
          </div>
          <input type="range" class="sens-slider" id="angleYSlider" min="-45" max="45" step="1" value="0" oninput="setAngleY(this.value)">
        </div>
        <button class="ctrl-btn" id="autoBtn" onclick="toggleAuto()" style="background:rgba(168,85,247,.15);border-color:rgba(168,85,247,.3);color:#c084fc;width:100%;margin-top:2px">自动按压模式</button>
      </div>
      <div class="info-grid">
        <div class="info-item"><span>合力 |F|</span><b id="magVal" style="color:#fbbf24">0.000</b></div>
        <div class="info-item"><span>俯仰角 θ</span><b id="thetaVal">0.0°</b></div>
        <div class="info-item"><span>偏航角 φ</span><b id="phiVal">0.0°</b></div>
        <div class="info-item"><span>峰值力</span><b id="peakVal">0.000</b></div>
        <div class="info-item"><span>采样数</span><b id="sampleVal">0</b></div>
        <div class="info-item"><span>运行时间</span><b id="timeVal">0.0s</b></div>
      </div>
      <div style="font-size:10px;color:#64748b;margin-top:8px;line-height:1.5">
        提示: 拖动紫色滑块模拟按压传感器<br>
        自动模式: 周期性变化力度和角度
      </div>
    </div>
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
let sensitivity=3.0, paused=false, displayRange=2.0;
let scene,camera,renderer,forceArrow,xArrow,yArrow,zArrow,forceSphere;
let timeChart;
const HLEN=200;
let tData=[],fxData=[],fyData=[],fzData=[],magData=[];
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

  const sGeom=new THREE.BoxGeometry(1.7,1.7,0.52);
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
    {label:'|F|',data:[],borderColor:'#fbbf24',backgroundColor:'rgba(251,191,36,.1)',borderWidth:2,pointRadius:0,tension:0.3}
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
async function getData(){
  try{const r=await fetch('/api/data');return await r.json()}catch(e){return null}
}
async function doTare(){try{await fetch('/api/tare')}catch(e){}}
function togglePause(){
  paused=!paused;
  const b=document.getElementById('pauseBtn');
  if(paused){b.textContent='继续';b.classList.add('paused')}
  else{b.textContent='暂停';b.classList.remove('paused')}
}
function setSens(v){sensitivity=parseFloat(v);document.getElementById('sensVal').textContent=sensitivity.toFixed(1)+'x'}
async function setNoise(v){
  document.getElementById('noiseVal').textContent=parseFloat(v).toFixed(3)+'N';
  try{await fetch('/api/noise?v='+v)}catch(e){}
}
let pressForce=0,pressAX=0,pressAY=0;
async function setPress(v){
  pressForce=parseFloat(v);
  document.getElementById('pressVal').textContent=pressForce.toFixed(1)+'N';
  try{await fetch('/api/press?force='+pressForce+'&ax='+pressAX+'&ay='+pressAY)}catch(e){}
}
async function setAngleX(v){
  pressAX=parseFloat(v);
  document.getElementById('angleXVal').textContent=pressAX+'°';
  try{await fetch('/api/press?force='+pressForce+'&ax='+pressAX+'&ay='+pressAY)}catch(e){}
}
async function setAngleY(v){
  pressAY=parseFloat(v);
  document.getElementById('angleYVal').textContent=pressAY+'°';
  try{await fetch('/api/press?force='+pressForce+'&ax='+pressAX+'&ay='+pressAY)}catch(e){}
}
async function toggleAuto(){
  try{
    const r=await fetch('/api/auto?v=toggle');
    const d=await r.json();
    const btn=document.getElementById('autoBtn');
    if(d.auto){btn.textContent='停止自动';btn.style.background='rgba(239,68,68,.15)';btn.style.borderColor='rgba(239,68,68,.3)';btn.style.color='#f87171'}
    else{btn.textContent='自动按压模式';btn.style.background='rgba(168,85,247,.15)';btn.style.borderColor='rgba(168,85,247,.3)';btn.style.color='#c084fc'}
  }catch(e){}
}
async function loop(){
  const d=await getData();
  if(d){
    document.getElementById('fpsVal').textContent=d.fps;
    document.getElementById('sampleVal').textContent=d.sampleCount;
    document.getElementById('timeVal').textContent=d.elapsed.toFixed(1)+'s';
    document.getElementById('peakVal').textContent=d.peakForce.toFixed(3);
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

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        elif self.path == '/api/data':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(self.sensor.get_data()).encode('utf-8'))
        elif self.path.startswith('/api/tare'):
            self.sensor.tare()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        elif self.path.startswith('/api/noise'):
            try:
                v = float(self.path.split('v=')[1])
                self.sensor.noise_sigma = v
            except:
                pass
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        elif self.path.startswith('/api/press'):
            try:
                params = self.path.split('?')[1] if '?' in self.path else ''
                kv = dict(p.split('=') for p in params.split('&') if '=' in p)
                self.sensor.press_force = float(kv.get('force', '0'))
                self.sensor.press_angle_x = float(kv.get('ax', '0'))
                self.sensor.press_angle_y = float(kv.get('ay', '0'))
            except:
                pass
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        elif self.path.startswith('/api/auto'):
            try:
                v = self.path.split('v=')[1]
                if v == 'toggle':
                    self.sensor.auto_mode = not self.sensor.auto_mode
                else:
                    self.sensor.auto_mode = (v == '1')
            except:
                pass
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "auto": self.sensor.auto_mode}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()


def run_web_server():
    with socketserver.TCPServer(("127.0.0.1", 8770), WebHandler) as httpd:
        print(f"[Web] http://127.0.0.1:8770/")
        httpd.serve_forever()


def main():
    sensor = SimSensor()
    WebHandler.sensor = sensor

    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()

    sensor.running = True
    sensor.start_time = time.time()

    print("=" * 50)
    print("  MuJoCo 触觉传感器仿真")
    print("  传感器尺寸: 17mm x 17mm x 5.2mm")
    print("=" * 50)
    print()
    print("  操作说明:")
    print("    1. 在网页上拖动'按压力度'滑块模拟按压传感器")
    print("    2. 调节'按压角度'模拟不同方向的力")
    print("    3. 可开启自动模式，模拟周期性按压")
    print()
    print("  网页可视化: http://127.0.0.1:8770/")
    print("=" * 50)
    print()

    fps_count = 0
    last_fps_time = time.time()

    while sensor.running:
        step_start = time.time()

        sensor.update_press()
        mujoco.mj_step(sensor.model, sensor.data)

        fx, fy, fz = sensor.get_contact_forces()
        fx, fy, fz = sensor.apply_noise(fx, fy, fz)
        sensor.fx = fx
        sensor.fy = fy
        sensor.fz = fz
        sensor.sample_count += 1

        fps_count += 1
        now = time.time()
        if now - last_fps_time >= 1.0:
            sensor.fps = fps_count
            fps_count = 0
            last_fps_time = now

        elapsed = time.time() - step_start
        sleep_time = sensor.model.opt.timestep - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    sensor.running = False
    print("\n仿真已停止")


if __name__ == "__main__":
    main()
