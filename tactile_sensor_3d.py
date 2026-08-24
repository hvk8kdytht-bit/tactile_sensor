import sys
import os
import math
import time
import serial
import struct

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False


class TactileSensor3D:
    """带3D可视化的触觉传感器驱动"""
    
    HEADER = bytes([0xB5, 0xA5, 0x55])
    
    CMD_STOP = 0xF0
    CMD_STANDARD = 0xB1
    CMD_DYNAMIC = 0xB2
    CMD_ANTI_MAGNETIC = 0xB3
    
    FREQ_1000HZ = 0xA1
    FREQ_500HZ = 0xA2
    FREQ_100HZ = 0xA3
    
    def __init__(self, port='COM3', baudrate=921600):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.buffer = b''
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.status = 0
        self.model_id = 0
        self.board_id = 0
        
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
        if self.ser and self.ser.is_open:
            try:
                self.send_command(self.CMD_STOP)
            except:
                pass
            self.ser.close()
    
    def send_command(self, cmd):
        if self.ser and self.ser.is_open:
            self.ser.write(bytes([cmd]))
    
    def calculate_crc(self, data):
        crc = 0
        for b in data:
            crc ^= b
        return crc
    
    def update(self):
        """读取最新数据，返回是否有新数据"""
        if not self.ser or not self.ser.is_open:
            return False
        
        # 读取所有可用数据
        if self.ser.in_waiting > 0:
            self.buffer += self.ser.read(self.ser.in_waiting)
        
        has_new = False
        
        # 解析所有完整帧，保留最后一帧
        while True:
            header_pos = self.buffer.find(self.HEADER)
            if header_pos == -1:
                break
            
            if header_pos > 0:
                self.buffer = self.buffer[header_pos:]
            
            if len(self.buffer) < 5:
                break
            
            packet_len = struct.unpack_from('<H', self.buffer, 3)[0]
            
            if packet_len < 20 or packet_len > 50:
                self.buffer = self.buffer[1:]
                continue
            
            if len(self.buffer) < packet_len:
                break
            
            frame = self.buffer[:packet_len]
            self.buffer = self.buffer[packet_len:]
            
            recv_crc = frame[-1]
            calc_crc = self.calculate_crc(frame[:-1])
            if recv_crc != calc_crc:
                continue
            
            self.model_id = frame[5]
            self.board_id = frame[6]
            self.status = frame[7]
            
            if len(frame) >= 21:
                self.x = struct.unpack_from('<f', frame, 8)[0]
                self.y = struct.unpack_from('<f', frame, 12)[0]
                self.z = struct.unpack_from('<f', frame, 16)[0]
                has_new = True
        
        return has_new
    
    def get_force_magnitude(self):
        """计算合力大小"""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def get_force_angles(self):
        """计算力的方向角 (俯仰角theta, 偏航角phi)"""
        mag = self.get_force_magnitude()
        if mag < 1e-6:
            return 0.0, 0.0
        
        theta = math.acos(max(-1, min(1, self.z / mag)))  # 与Z轴夹角
        phi = math.atan2(self.y, self.x)  # XY平面内与X轴夹角
        
        return math.degrees(theta), math.degrees(phi)
    
    def stop_stream(self):
        self.send_command(self.CMD_STOP)
        time.sleep(0.2)
        if self.ser:
            self.ser.reset_input_buffer()
        self.buffer = b''
    
    def start_standard(self):
        self.send_command(self.CMD_STANDARD)


class ForceVisualizer3D:
    """三维力可视化器"""
    
    def __init__(self, sensor):
        self.sensor = sensor
        self.running = False
        
        # 历史数据
        self.history_len = 200
        self.x_history = []
        self.y_history = []
        self.z_history = []
        self.mag_history = []
        self.time_history = []
        
        # 最大值（用于归一化显示）
        self.max_force = 5.0  # 默认5N
        
    def setup_plots(self):
        """设置图表"""
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.canvas.manager.set_window_title('触觉传感器 - 三维力实时可视化')
        
        # 1. 3D力向量图 (左上)
        self.ax3d = self.fig.add_subplot(2, 2, 1, projection='3d')
        self.ax3d.set_title('三维力向量', fontsize=12, fontweight='bold')
        
        # 2. XYZ分量时序图 (右上)
        self.ax_time = self.fig.add_subplot(2, 2, 2)
        self.ax_time.set_title('力分量时序', fontsize=12, fontweight='bold')
        self.ax_time.set_xlabel('时间 (s)')
        self.ax_time.set_ylabel('力 (N)')
        self.ax_time.grid(True, alpha=0.3)
        
        # 3. 合力大小柱状图 (左下)
        self.ax_mag = self.fig.add_subplot(2, 2, 3)
        self.ax_mag.set_title('合力大小', fontsize=12, fontweight='bold')
        
        # 4. 方向极坐标图 (右下)
        self.ax_polar = self.fig.add_subplot(2, 2, 4, projection='polar')
        self.ax_polar.set_title('力方向 (XY平面)', fontsize=12, fontweight='bold')
        
        plt.tight_layout(pad=3.0)
        
    def update_3d_plot(self):
        """更新3D力向量图"""
        self.ax3d.clear()
        self.ax3d.set_title('三维力向量', fontsize=12, fontweight='bold')
        
        # 坐标范围
        r = max(self.max_force * 0.1, abs(self.sensor.x), abs(self.sensor.y), abs(self.sensor.z))
        r = max(r, 1.0)
        
        self.ax3d.set_xlim(-r, r)
        self.ax3d.set_ylim(-r, r)
        self.ax3d.set_zlim(-r, r)
        
        # 绘制坐标轴
        self.ax3d.plot([0, r*0.9], [0, 0], [0, 0], 'b--', alpha=0.3, linewidth=1)
        self.ax3d.plot([0, 0], [0, r*0.9], [0, 0], 'g--', alpha=0.3, linewidth=1)
        self.ax3d.plot([0, 0], [0, 0], [0, r*0.9], 'r--', alpha=0.3, linewidth=1)
        
        self.ax3d.text(r*0.95, 0, 0, 'X', color='blue', fontsize=10)
        self.ax3d.text(0, r*0.95, 0, 'Y', color='green', fontsize=10)
        self.ax3d.text(0, 0, r*0.95, 'Z', color='red', fontsize=10)
        
        # 绘制原点
        self.ax3d.scatter([0], [0], [0], color='black', s=30, marker='o')
        
        # 绘制力向量
        fx, fy, fz = self.sensor.x, self.sensor.y, self.sensor.z
        
        # X分量
        self.ax3d.plot([0, fx], [0, 0], [0, 0], 'b-', linewidth=2, alpha=0.7, label='Fx')
        # Y分量
        self.ax3d.plot([fx, fx], [0, fy], [0, 0], 'g-', linewidth=2, alpha=0.7, label='Fy')
        # Z分量
        self.ax3d.plot([fx, fx], [fy, fy], [0, fz], 'r-', linewidth=2, alpha=0.7, label='Fz')
        
        # 合力
        mag = math.sqrt(fx**2 + fy**2 + fz**2)
        if mag > 1e-6:
            self.ax3d.plot([0, fx], [0, fy], [0, fz], 'k-', linewidth=3, label='合力 F')
            # 箭头（用小球表示端点）
            self.ax3d.scatter([fx], [fy], [fz], color='red', s=80, marker='o', depthshade=True)
        
        # 投影辅助线
        self.ax3d.plot([fx, fx], [fy, fy], [0, 0], 'k:', alpha=0.3)
        self.ax3d.plot([0, fx], [0, fy], [0, 0], 'k:', alpha=0.3)
        
        # 显示数值
        mag = self.sensor.get_force_magnitude()
        theta, phi = self.sensor.get_force_angles()
        
        info_text = f'Fx={fx:.3f}N\nFy={fy:.3f}N\nFz={fz:.3f}N\n|F|={mag:.3f}N\nθ={theta:.1f}°\nφ={phi:.1f}°'
        self.ax3d.text2D(0.02, 0.98, info_text, transform=self.ax3d.transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        self.ax3d.legend(loc='upper right', fontsize=8)
        self.ax3d.view_init(elev=20, azim=-60)
        
    def update_time_plot(self):
        """更新时序图"""
        self.ax_time.clear()
        self.ax_time.set_title('力分量时序', fontsize=12, fontweight='bold')
        self.ax_time.set_xlabel('时间 (s)')
        self.ax_time.set_ylabel('力 (N)')
        self.ax_time.grid(True, alpha=0.3)
        
        if len(self.time_history) > 1:
            t = np.array(self.time_history) - self.time_history[0]
            
            self.ax_time.plot(t, self.x_history, 'b-', label='Fx', linewidth=1.5)
            self.ax_time.plot(t, self.y_history, 'g-', label='Fy', linewidth=1.5)
            self.ax_time.plot(t, self.z_history, 'r-', label='Fz', linewidth=1.5)
            self.ax_time.plot(t, self.mag_history, 'k-', label='|F|', linewidth=1.5, alpha=0.7)
            
            self.ax_time.legend(loc='upper right', fontsize=9)
            
            # 自适应Y轴
            all_vals = self.x_history + self.y_history + self.z_history + self.mag_history
            if all_vals:
                ymin = min(all_vals)
                ymax = max(all_vals)
                if ymin == ymax:
                    ymin -= 0.5
                    ymax += 0.5
                margin = (ymax - ymin) * 0.1
                self.ax_time.set_ylim(ymin - margin, ymax + margin)
    
    def update_mag_plot(self):
        """更新合力大小柱状图"""
        self.ax_mag.clear()
        self.ax_mag.set_title('合力大小与方向角', fontsize=12, fontweight='bold')
        
        mag = self.sensor.get_force_magnitude()
        theta, phi = self.sensor.get_force_angles()
        
        # 三个指标：合力大小、俯仰角、偏航角
        metrics = ['合力 |F| (N)', '俯仰角 θ (°)', '偏航角 φ (°)']
        values = [mag, theta, phi]
        colors = ['#e53e3e', '#805ad5', '#3182ce']
        
        bars = self.ax_mag.bar(metrics, values, color=colors, alpha=0.7, width=0.5)
        
        # 在柱顶上显示数值
        for bar, val in zip(bars, values):
            height = bar.get_height()
            self.ax_mag.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.3f}' if abs(val) < 10 else f'{val:.1f}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        self.ax_mag.set_ylim(0, max(max(values) * 1.2, 1.0))
        self.ax_mag.grid(True, alpha=0.3, axis='y')
    
    def update_polar_plot(self):
        """更新极坐标方向图"""
        self.ax_polar.clear()
        self.ax_polar.set_title('力方向 (XY平面)', fontsize=12, fontweight='bold', pad=15)
        
        fx, fy = self.sensor.x, self.sensor.y
        mag_xy = math.sqrt(fx**2 + fy**2)
        phi = math.atan2(fy, fx)
        
        # 归一化半径显示
        display_r = min(mag_xy / max(self.max_force, 0.1), 1.0)
        
        if mag_xy > 1e-6:
            # 绘制方向箭头
            self.ax_polar.arrow(0, 0, phi, display_r, 
                              alpha=0.8, width=0.05, 
                              head_width=0.2, head_length=0.1,
                              fc='#e53e3e', ec='#e53e3e')
            
            # 端点标记
            self.ax_polar.scatter([phi], [display_r], c='red', s=80, zorder=5)
        
        # 显示数值
        self.ax_polar.text(0, 1.3, f'Fxy={mag_xy:.3f}N\nφ={math.degrees(phi):.1f}°',
                          ha='center', va='center', fontsize=10,
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        self.ax_polar.set_ylim(0, 1.2)
        self.ax_polar.set_yticklabels([])  # 隐藏径向刻度
    
    def animate(self, frame):
        """动画更新函数"""
        # 读取新数据
        new_data = False
        for _ in range(10):  # 每次读取多帧
            if self.sensor.update():
                new_data = True
        
        if not new_data:
            return
        
        # 记录历史
        current_time = time.time()
        self.x_history.append(self.sensor.x)
        self.y_history.append(self.sensor.y)
        self.z_history.append(self.sensor.z)
        self.mag_history.append(self.sensor.get_force_magnitude())
        self.time_history.append(current_time)
        
        # 限制历史长度
        while len(self.time_history) > self.history_len:
            self.x_history.pop(0)
            self.y_history.pop(0)
            self.z_history.pop(0)
            self.mag_history.pop(0)
            self.time_history.pop(0)
        
        # 更新最大值
        current_max = max(abs(self.sensor.x), abs(self.sensor.y), abs(self.sensor.z),
                         self.sensor.get_force_magnitude())
        if current_max > self.max_force * 0.8:
            self.max_force = current_max * 1.5
        
        # 更新所有图表
        self.update_3d_plot()
        self.update_time_plot()
        self.update_mag_plot()
        self.update_polar_plot()
    
    def run(self):
        """运行可视化"""
        self.setup_plots()
        
        # 使用定时器更新
        from matplotlib.animation import FuncAnimation
        
        self.anim = FuncAnimation(
            self.fig, self.animate,
            interval=50,  # 20 FPS
            blit=False,
            cache_frame_data=False
        )
        
        print("三维力可视化已启动！")
        print("请触摸传感器观察力向量变化...")
        print("关闭窗口即可退出。")
        
        plt.show()
        self.running = False


def main():
    print("=" * 60)
    print("  触觉传感器 - 三维力实时可视化")
    print("=" * 60)
    
    sensor = TactileSensor3D('COM3', 921600)
    
    if not sensor.connect():
        print("\n✗ 无法连接传感器")
        return
    
    try:
        # 初始化传感器
        sensor.stop_stream()
        time.sleep(0.3)
        sensor.send_command(TactileSensor3D.FREQ_100HZ)
        time.sleep(0.1)
        sensor.start_standard()
        time.sleep(0.5)
        
        # 读取几帧确认数据正常
        for i in range(20):
            sensor.update()
            if sensor.model_id != 0x00 and sensor.model_id != 0:
                break
        
        model_names = {
            0x11: "0x11 标准型",
            0x12: "0x12 高精度型",
            0x13: "0x13 高灵敏型",
            0x14: "0x14 超量程型",
        }
        model_name = model_names.get(sensor.model_id, f"未知(0x{sensor.model_id:02X})")
        
        print(f"\n  传感器型号: {model_name}")
        print(f"  节点编号: {sensor.board_id}")
        print(f"  通信端口: COM3 @ 921600 bps")
        print(f"\n正在启动3D可视化...")
        
        # 启动可视化
        viz = ForceVisualizer3D(sensor)
        viz.run()
        
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sensor.disconnect()
        print("\n传感器已断开")


if __name__ == '__main__':
    main()
