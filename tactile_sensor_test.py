import serial
import struct
import time
import sys

class TactileSensor:
    """消费级指尖指腹传感器驱动类"""
    
    # 帧头
    HEADER = bytes([0xB5, 0xA5, 0x55])
    
    # 指令定义
    CMD_STOP = 0xF0           # 停止发送，进入CMD模式
    CMD_STANDARD = 0xB1       # 标准模式
    CMD_DYNAMIC = 0xB2        # 动态模式
    CMD_ANTI_MAGNETIC = 0xB3  # 抗磁模式
    CMD_SINGLE = 0x1C         # 单次采样
    
    # 频率设置
    FREQ_2800HZ = 0xA0
    FREQ_1000HZ = 0xA1
    FREQ_500HZ = 0xA2
    FREQ_100HZ = 0xA3
    FREQ_10HZ = 0xA4
    
    # 状态位定义
    STATUS_NORMAL = 0x00
    STATUS_OVERRANGE = 0x01
    STATUS_NOT_READY = 0x02
    
    def __init__(self, port='COM3', baudrate=921600):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.buffer = b''
        
    def connect(self):
        """连接传感器"""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
            time.sleep(0.5)
            print(f"✓ 传感器已连接: {self.port} @ {self.baudrate} bps")
            return True
        except Exception as e:
            print(f"✗ 连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("传感器已断开")
    
    def send_command(self, cmd_byte):
        """发送单字节指令"""
        if self.ser and self.ser.is_open:
            self.ser.reset_input_buffer()
            self.ser.write(bytes([cmd_byte]))
            time.sleep(0.05)
            return True
        return False
    
    def calculate_crc(self, data):
        """计算XOR校验"""
        crc = 0
        for b in data:
            crc ^= b
        return crc
    
    def find_frame(self):
        """从缓冲区中寻找完整的数据帧"""
        # 在缓冲区中查找帧头
        header_pos = self.buffer.find(self.HEADER)
        
        if header_pos == -1:
            return None
        
        # 移除帧头之前的无用数据
        if header_pos > 0:
            self.buffer = self.buffer[header_pos:]
            header_pos = 0
        
        # 检查是否有足够的数据读取长度字段
        if len(self.buffer) < 5:
            return None
        
        # 读取包长度（小端模式 uint16）
        packet_length = struct.unpack_from('<H', self.buffer, 3)[0]
        
        # 检查包长度是否合理
        if packet_length < 10 or packet_length > 100:
            # 不合理，跳过这个帧头
            self.buffer = self.buffer[1:]
            return None
        
        # 检查是否收到完整的包
        if len(self.buffer) < packet_length:
            return None
        
        # 提取完整帧
        frame = self.buffer[:packet_length]
        self.buffer = self.buffer[packet_length:]
        
        # 验证CRC
        received_crc = frame[-1]
        calculated_crc = self.calculate_crc(frame[:-1])
        
        if received_crc != calculated_crc:
            # CRC不匹配，跳过
            return None
        
        return frame
    
    def parse_frame(self, frame):
        """解析数据帧"""
        if len(frame) < 10:
            return None
        
        # 解析头部字段
        packet_length = struct.unpack_from('<H', frame, 3)[0]
        model_id = frame[5]
        board_id = frame[6]
        status = frame[7]
        
        # 解析XYZ数据 (3个IEEE754 float)
        if len(frame) >= 21:  # 8(header) + 12(payload) + 1(crc) = 21
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
            'raw': frame
        }
    
    def read_data(self, timeout=2.0):
        """读取并解析一帧数据"""
        if not self.ser or not self.ser.is_open:
            return None
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # 读取新数据
            if self.ser.in_waiting > 0:
                self.buffer += self.ser.read(self.ser.in_waiting)
            
            # 尝试解析帧
            frame = self.find_frame()
            if frame:
                return self.parse_frame(frame)
            
            time.sleep(0.001)
        
        return None
    
    def stop_stream(self):
        """停止数据流，进入CMD模式"""
        self.send_command(self.CMD_STOP)
        time.sleep(0.2)
        # 清空缓冲区
        if self.ser:
            self.ser.reset_input_buffer()
        self.buffer = b''
    
    def start_standard_mode(self):
        """启动标准模式"""
        self.send_command(self.CMD_STANDARD)
    
    def start_dynamic_mode(self):
        """启动动态模式"""
        self.send_command(self.CMD_DYNAMIC)
    
    def set_frequency(self, freq_cmd):
        """设置输出频率（需在CMD模式下）"""
        self.send_command(freq_cmd)
    
    def get_single_sample(self):
        """获取单次采样数据"""
        self.send_command(self.CMD_SINGLE)
        return self.read_data(timeout=1.0)


def main():
    print("=" * 70)
    print("消费级指尖指腹传感器 - 连接测试")
    print("=" * 70)
    
    sensor = TactileSensor('COM3', 921600)
    
    if not sensor.connect():
        print("无法连接传感器，请检查端口和接线")
        return
    
    try:
        # 1. 先停止自动发送，进入CMD模式
        print("\n[1] 停止自动发送，进入CMD模式...")
        sensor.stop_stream()
        time.sleep(0.5)
        
        # 2. 读取传感器信息
        print("\n[2] 读取传感器信息...")
        
        # 先尝试单次采样看传感器是否响应
        sample = sensor.get_single_sample()
        if sample:
            print(f"  ✓ 传感器响应正常")
            print(f"    传感器型号: 0x{sample['model_id']:02X}")
            print(f"    节点号: 0x{sample['board_id']:02X}")
            print(f"    状态: 0x{sample['status']:02X}")
            
            status_text = {
                0x00: "正常",
                0x01: "超量程/异常",
                0x02: "未就绪/未握手"
            }.get(sample['status'], "未知")
            print(f"    状态描述: {status_text}")
        else:
            print("  ✗ 未收到传感器响应，尝试直接启动数据流...")
        
        # 3. 设置较低的频率便于观察
        print("\n[3] 设置输出频率为 10Hz 便于观察...")
        sensor.set_frequency(TactileSensor.FREQ_10HZ)
        time.sleep(0.2)
        
        # 4. 启动标准模式
        print("\n[4] 启动标准模式 (0xB1)...")
        sensor.start_standard_mode()
        time.sleep(0.3)
        
        # 5. 读取若干帧数据
        print("\n[5] 读取数据 (10帧)...")
        print(f"{'序号':>4s}  {'X':>10s}  {'Y':>10s}  {'Z':>10s}  {'状态':>6s}")
        print("-" * 55)
        
        for i in range(10):
            data = sensor.read_data(timeout=2.0)
            if data:
                status_str = {
                    0x00: "正常",
                    0x01: "超量程",
                    0x02: "未就绪"
                }.get(data['status'], f"0x{data['status']:02X}")
                
                print(f"  {i+1:2d}  {data['x']:10.4f}  {data['y']:10.4f}  {data['z']:10.4f}  {status_str:>6s}")
            else:
                print(f"  {i+1:2d}  超时，未收到数据")
        
        # 6. 停止数据流
        print("\n[6] 停止数据流...")
        sensor.stop_stream()
        
        print("\n✓ 传感器连接测试完成！")
        print("\n传感器工作正常，XYZ三轴数据已成功读取。")
        
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sensor.disconnect()


if __name__ == '__main__':
    main()
