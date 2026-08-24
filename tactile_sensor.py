import serial
import struct
import time
import sys
import os

class TactileSensor:
    """消费级指尖指腹传感器驱动类
    支持型号: 0x11 / 0x12 / 0x13 / 0x14
    """
    
    HEADER = bytes([0xB5, 0xA5, 0x55])
    
    # 指令
    CMD_STOP = 0xF0
    CMD_STANDARD = 0xB1
    CMD_DYNAMIC = 0xB2
    CMD_ANTI_MAGNETIC = 0xB3
    CMD_SINGLE = 0x1C
    
    # 频率
    FREQ_2800HZ = 0xA0
    FREQ_1000HZ = 0xA1
    FREQ_500HZ = 0xA2
    FREQ_100HZ = 0xA3
    FREQ_10HZ = 0xA4
    
    # 型号名称
    MODEL_NAMES = {
        0x00: "初始化/校准中",
        0x11: "0x11 标准型",
        0x12: "0x12 高精度型",
        0x13: "0x13 高灵敏型",
        0x14: "0x14 超量程型",
    }
    
    STATUS_NORMAL = 0x00
    STATUS_OVERRANGE = 0x01
    STATUS_NOT_READY = 0x02
    
    def __init__(self, port='COM3', baudrate=921600):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.buffer = b''
        self.model_id = None
        self.board_id = None
        
    def connect(self):
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.5
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
    
    def send_command(self, cmd_byte):
        if self.ser and self.ser.is_open:
            self.ser.write(bytes([cmd_byte]))
            return True
        return False
    
    def calculate_crc(self, data):
        crc = 0
        for b in data:
            crc ^= b
        return crc
    
    def read_frame(self, timeout=1.0):
        """读取并解析一帧数据"""
        if not self.ser or not self.ser.is_open:
            return None
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.ser.in_waiting > 0:
                self.buffer += self.ser.read(self.ser.in_waiting)
            
            # 查找帧头
            header_pos = self.buffer.find(self.HEADER)
            if header_pos == -1:
                time.sleep(0.001)
                continue
            
            # 移除帧头前的垃圾数据
            if header_pos > 0:
                self.buffer = self.buffer[header_pos:]
            
            # 检查是否够读取长度
            if len(self.buffer) < 5:
                time.sleep(0.001)
                continue
            
            # 读取包长度
            packet_len = struct.unpack_from('<H', self.buffer, 3)[0]
            
            # 检查长度合理性
            if packet_len < 20 or packet_len > 50:
                self.buffer = self.buffer[1:]
                continue
            
            # 检查是否收齐完整帧
            if len(self.buffer) < packet_len:
                time.sleep(0.001)
                continue
            
            # 提取帧
            frame = self.buffer[:packet_len]
            self.buffer = self.buffer[packet_len:]
            
            # CRC校验
            recv_crc = frame[-1]
            calc_crc = self.calculate_crc(frame[:-1])
            if recv_crc != calc_crc:
                continue
            
            # 解析
            model_id = frame[5]
            board_id = frame[6]
            status = frame[7]
            
            if len(frame) >= 21:
                x = struct.unpack_from('<f', frame, 8)[0]
                y = struct.unpack_from('<f', frame, 12)[0]
                z = struct.unpack_from('<f', frame, 16)[0]
            else:
                x = y = z = 0.0
            
            return {
                'model_id': model_id,
                'board_id': board_id,
                'status': status,
                'x': x,
                'y': y,
                'z': z,
                'timestamp': time.time()
            }
        
        return None
    
    def stop_stream(self):
        self.send_command(self.CMD_STOP)
        time.sleep(0.2)
        if self.ser:
            self.ser.reset_input_buffer()
        self.buffer = b''
    
    def start_mode(self, mode_cmd):
        self.send_command(mode_cmd)
    
    def set_frequency(self, freq_cmd):
        self.send_command(freq_cmd)
        time.sleep(0.1)


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def bar_graph(value, max_val, width=20):
    """生成条形图字符串"""
    if max_val == 0:
        return '[' + ' ' * width + ']'
    ratio = min(abs(value) / max_val, 1.0)
    filled = int(ratio * width)
    bar = '█' * filled + '░' * (width - filled)
    return f'[{bar}]'


def realtime_test():
    """实时数据测试"""
    clear_screen()
    print("=" * 70)
    print("  消费级指尖指腹传感器 - 实时测试")
    print("=" * 70)
    
    sensor = TactileSensor('COM3', 921600)
    
    if not sensor.connect():
        print("\n✗ 无法连接到传感器，请检查:")
        print("  1. 传感器是否正确连接到电脑")
        print("  2. COM端口是否正确 (当前: COM3)")
        print("  3. 传感器是否已上电")
        return
    
    try:
        # 停止自动发送
        sensor.stop_stream()
        time.sleep(0.3)
        
        # 设置频率为 100Hz 便于观察
        sensor.set_frequency(TactileSensor.FREQ_100HZ)
        time.sleep(0.1)
        
        # 启动标准模式
        sensor.start_mode(TactileSensor.CMD_STANDARD)
        time.sleep(0.5)
        
        # 读取第一帧获取传感器信息
        first_data = sensor.read_frame(timeout=2.0)
        if not first_data:
            print("\n✗ 无法读取传感器数据")
            return
        
        model_name = TactileSensor.MODEL_NAMES.get(first_data['model_id'], f"未知型号(0x{first_data['model_id']:02X})")
        
        clear_screen()
        print("=" * 70)
        print("  消费级指尖指腹传感器 - 实时测试")
        print("=" * 70)
        print(f"  传感器型号: {model_name}")
        print(f"  节点编号: {first_data['board_id']}")
        print(f"  端口: COM3 @ 921600 bps")
        print(f"  模式: 标准模式 (0xB1)")
        print(f"  输出频率: 100Hz")
        print("-" * 70)
        print()
        print("  请用手指触摸/按压传感器，观察数据变化")
        print("  按 Ctrl+C 退出测试")
        print()
        
        # 记录最大值
        max_abs = 0.1
        x_max = y_max = z_max = 0.0
        x_min = y_min = z_min = 0.0
        
        frame_count = 0
        start_time = time.time()
        last_print = 0
        
        while True:
            data = sensor.read_frame(timeout=1.0)
            if not data:
                continue
            
            frame_count += 1
            
            # 更新最大最小值
            x_max = max(x_max, data['x'])
            y_max = max(y_max, data['y'])
            z_max = max(z_max, data['z'])
            x_min = min(x_min, data['x'])
            y_min = min(y_min, data['y'])
            z_min = min(z_min, data['z'])
            
            current_max = max(abs(x_max), abs(x_min), abs(y_max), abs(y_min), abs(z_max), abs(z_min))
            max_abs = max(max_abs, current_max)
            
            # 每 50ms 更新一次显示
            elapsed = time.time() - start_time
            if elapsed - last_print >= 0.05:
                last_print = elapsed
                
                status_text = {
                    0x00: "正常",
                    0x01: "超量程",
                    0x02: "未就绪"
                }.get(data['status'], f"0x{data['status']:02X}")
                
                # 使用 \r 回到行首更新
                output = f"""  当前数据:
    X: {data['x']:+9.4f} {bar_graph(data['x'], max_abs)}
    Y: {data['y']:+9.4f} {bar_graph(data['y'], max_abs)}
    Z: {data['z']:+9.4f} {bar_graph(data['z'], max_abs)}
  
  本次测量范围:
    X: [{x_min:+.4f}, {x_max:+.4f}]
    Y: [{y_min:+.4f}, {y_max:+.4f}]
    Z: [{z_min:+.4f}, {z_max:+.4f}]
  
  状态: {status_text}  |  帧率: {frame_count/elapsed:.1f} Hz  |  运行: {elapsed:.1f}s
  
  按 Ctrl+C 退出"""
                
                # 移动光标到顶部并更新
                print(f"\033[12A", end='')
                print(output, end='')
                sys.stdout.flush()
    
    except KeyboardInterrupt:
        pass
    finally:
        sensor.disconnect()
        print("\n\n" + "=" * 70)
        print("  测试结束")
        print("=" * 70)


def quick_test():
    """快速连接测试"""
    print("=" * 70)
    print("  消费级指尖指腹传感器 - 快速连接测试")
    print("=" * 70)
    
    sensor = TactileSensor('COM3', 921600)
    
    if not sensor.connect():
        print("\n✗ 连接失败")
        return False
    
    try:
        # 停止并重新启动
        sensor.stop_stream()
        time.sleep(0.3)
        sensor.set_frequency(TactileSensor.FREQ_10HZ)
        sensor.start_mode(TactileSensor.CMD_STANDARD)
        time.sleep(0.5)
        
        # 先读取几帧跳过校准帧，找到有效型号
        model_id = None
        board_id = None
        for _ in range(10):
            data = sensor.read_frame(timeout=2.0)
            if data and data['model_id'] != 0x00:
                model_id = data['model_id']
                board_id = data['board_id']
                break
        
        if model_id is not None:
            model_name = TactileSensor.MODEL_NAMES.get(model_id, f"0x{model_id:02X}")
            print(f"\n  传感器型号: {model_name}")
            print(f"  节点编号: {board_id}")
            print(f"  通信端口: COM3 @ 921600 bps")
        
        # 读取5帧有效数据
        print(f"\n  {'序号':>4s}  {'X':>10s}  {'Y':>10s}  {'Z':>10s}  {'状态':>6s}")
        print("  " + "-" * 55)
        
        valid_count = 0
        total_read = 0
        while valid_count < 5 and total_read < 20:
            data = sensor.read_frame(timeout=2.0)
            total_read += 1
            if data and data['model_id'] != 0x00:
                valid_count += 1
                status_str = {0x00: "正常", 0x01: "超量程", 0x02: "未就绪"}.get(data['status'], "?")
                print(f"  {valid_count:4d}  {data['x']:10.4f}  {data['y']:10.4f}  {data['z']:10.4f}  {status_str:>6s}")
        
        print("\n  ✓ 传感器连接正常！")
        print("  提示: 当前数值为0是因为未施加压力，属于正常状态。")
        return True
        
    finally:
        sensor.disconnect()


if __name__ == '__main__':
    print("\n请选择测试模式:")
    print("  1. 快速连接测试 (读取5帧数据)")
    print("  2. 实时数据测试 (持续显示，可触摸传感器)")
    print()
    
    choice = input("请输入选项 (1/2，默认2): ").strip() or '2'
    
    if choice == '1':
        quick_test()
    else:
        realtime_test()
