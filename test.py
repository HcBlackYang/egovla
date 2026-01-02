import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()

print(f"检测到 {len(devices)} 个 RealSense 设备:")
for dev in devices:
    sn = dev.get_info(rs.camera_info.serial_number)
    name = dev.get_info(rs.camera_info.name)
    print(f"  - 设备名: {name}, 序列号: {sn}")

if len(devices) == 0:
    print("警告: 未检测到任何 RealSense 设备！请检查 USB 连接。")