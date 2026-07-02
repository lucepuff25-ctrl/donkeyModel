# -*- coding: utf-8 -*-
"""
双路视觉：Cam0-YOLO检测、Cam1-车道线检测+串口发送角度
启动阶段：固定直行，直到识别到左转/右转标志并执行完成后进入正常车道线模式
串口协议：AA 80 <angle> FF   angle∈[0,255] 对应 -90°~+90°
"""
import cv2
import time
import threading
import serial
import queue
import numpy as np
import math
import signal
import sys
from pathlib import Path
from ultralytics import YOLO

# -------------------- 基础配置 --------------------
ROOT = Path(__file__).resolve().parent
SERIAL_PORT = "/dev/ttyCH341USB0"
SERIAL_BAUD = 115200
FRAME_SIZE = 640
LANE_CROP = 640
CONF_THRES = 0.25
set_hight = 0.75
FPS = 15  # 平均帧率
FRAME_DELAY = 1.0 / FPS  # 每帧时间

# -------------------- 全局变量 --------------------
flag = 1  # 暂停标志
flag_lock = threading.Lock()
wait_for_first_turn = True  # True=尚未识别到左转/右转
turn_in_progress = False  # 转向进行中标志，防止转向被冲掉
flag_zebra_line = 0
red_light_stop = 0

# 全局角度变量和锁
angle_byte = 0x80  # 默认直行
angle_lock = threading.Lock()

# -------------------- 帧计数器（替代sleep） --------------------
turn_frame_count = 0
turn_total_frames = 0
turn_type = None
passing_bay_frame_count = 0
passing_bay_total_frames = 0
warning_frame_count = 0
warning_total_frames = 0
zebra_crossing_frame_count = 0
zebra_crossing_total_frames = 0
roundabout_turn_frame_count = 0
roundabout_turn_total_frames = 0

# -------------------- 识别任务字典 --------------------
# key: 识别名称, value: 1=可执行, 0=已执行过不再执行
test = {
    "warning": 1,  # 警告标志
    "passing_bay": 1,  # 避让标志
    "green": 1,  # 绿灯
    "red": 1,  # 红灯
    "PPC": 1  # 人行横道
}
test_lock = threading.Lock()

# -------------------- 环岛模式 --------------------
roundabout_mode = False
roundabout_counter = 0
left_line_lost_count = 0
LEFT_LINE_LOST_THRESHOLD = 5

# -------------------- 状态机 --------------------
BOTH_LINES_SEEN = 0
ONLY_LEFT_SEEN = 1
ONLY_RIGHT_SEEN = 2
NO_LINES_SEEN = 3
current_state = NO_LINES_SEEN
last_known_track_width = 500
TARGET_OFFSET = 100
MIN_VALID_TRACK_WIDTH = 90
MIN_SUPPORT_WINDOWS = 3
MIN_BIMODAL_GAP_PCT = 0.08
VALLEY_RATIO_MAX = 0.55
EMA_ALPHA = 0.2
TWO_LANE_STREAK_MIN = 3
FALLBACK_TRACK_WIDTH = 220
SAFE_INWARD = 20

# -------------------- 曲率趋势 --------------------
CURVATURE_HISTORY = 8
CURVATURE_THRESHOLD = 0.001
curvature_history = []
centerline_trend = 0
FORCE_SINGLE_SIDE = False
track_width_ema = None
two_lane_streak = 0

# -------------------- 颜色阈值 --------------------
YELLOW_LOWER = np.array([20, 90, 90])
YELLOW_UPPER = np.array([40, 255, 255])
WHITE_LOWER = np.array([20, 90, 90])
WHITE_UPPER = np.array([40, 255, 255])
RED_LOWER = np.array([0, 50, 50])
RED_UPPER = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 50, 50])
RED_UPPER2 = np.array([180, 255, 255])

# -------------------- ROI --------------------
ROI_TOP_HEIGHT = 0.60
ROI_BOTTOM_MARGIN = 0.0
ROI_TOP_WIDTH = 0.9
ROI_BOTTOM_WIDTH = 1.0

# -------------------- 滑动窗口 --------------------
NWINDOWS = 9
MARGIN = 120
MINPIX = 30

# -------------------- 串口 --------------------
ser_lock = threading.Lock()
ser_data = bytearray([0xAA, 0x40, 0x80, 0xFF])
shutdown_flag = False


def signal_handler(signum, frame):
    global shutdown_flag, flag
    print(f"\n[INFO] 收到信号 {signum}，正在安全退出...")
    shutdown_flag = True
    flag = 0

    # 发送五次停止指令
    for _ in range(5):
        send_serial_data(0x00, 0x80)
        time.sleep(0.05)  # 保持少量sleep用于退出

    if ser and ser.is_open:
        ser.close()
    cv2.destroyAllWindows()
    sys.exit(0)


def open_serial(port, baud):
    try:
        return serial.Serial(port, baud, timeout=0.05)
    except Exception as e:
        print(f"[WARN] 串口打开失败 {port}: {e}")
        return None


ser = open_serial(SERIAL_PORT, SERIAL_BAUD)


def send_serial_data(speed_byte, angle_byte):
    with ser_lock:
        ser_data[1] = speed_byte
        ser_data[2] = angle_byte
        if ser and ser.is_open:
            try:
                ser.write(ser_data)
                print(f"[SERIAL] 发送: {list(ser_data)}")
            except Exception as e:
                print("[WARN] 串口写入失败:", e)


# -------------------- 识别任务管理 --------------------
def can_execute_task(task_name):
    """检查任务是否可以执行"""
    with test_lock:
        return test.get(task_name, 0) == 1


def mark_task_executed(task_name):
    """标记任务已执行"""
    with test_lock:
        test[task_name] = 0
        print(f"[TASK] 任务 '{task_name}' 已标记为已执行")


def get_task_status():
    """获取任务状态"""
    with test_lock:
        return test.copy()


# -------------------- 暂停/恢复 --------------------
def reset_flag_after_frames(frame_count=15):
    """基于帧数恢复车道线检测"""
    global flag

    def reset():
        nonlocal frame_count
        while frame_count > 0 and not shutdown_flag:
            frame_count -= 1
            time.sleep(FRAME_DELAY)
        with flag_lock:
            flag = 1
        print("[FLAG] 车道线检测已恢复")

    threading.Thread(target=reset, daemon=True).start()


# -------------------- 执行转向动作 --------------------
def start_turn_action(turn_type_name):
    """开始转向动作（非阻塞）"""
    global turn_in_progress, turn_frame_count, turn_total_frames, turn_type, roundabout_mode, roundabout_counter

    print(f"[TURN] 开始执行{turn_type_name}动作...")
    turn_in_progress = True
    turn_type = turn_type_name
    turn_frame_count = 0
    turn_total_frames = int(1.0 / FRAME_DELAY)  # 1秒对应的帧数

    # 暂停车道线检测
    with flag_lock:
        flag = 0
    print("[FLAG] 转向进行中，暂停车道线检测")

    if turn_type == "trun_left":
        send_serial_data(0x80, 0xF0)  # 左转
        # 进入环岛模式
        roundabout_mode = True
        roundabout_counter = 0
        print("[ROUNDABOUT] 检测到左转，进入环岛模式")
    elif turn_type == "trun_right":
        send_serial_data(0x70, 0x20)  # 右转


def update_turn_action():
    """更新转向动作状态"""
    global turn_in_progress, turn_frame_count, turn_total_frames, turn_type, wait_for_first_turn

    if not turn_in_progress:
        return False

    turn_frame_count += 1

    if turn_frame_count >= turn_total_frames:
        # 转向完成，恢复直行
        send_serial_data(0x80, 0x80)
        print("[TURN] 转向完成，恢复直行")

        # 清除转向进行中标志
        turn_in_progress = False
        turn_type = None

        # 恢复车道线检测
        with flag_lock:
            flag = 1
        print("[FLAG] 转向完成，车道线检测已恢复")

        # 标记转向完成，启动正常车道线检测模式
        wait_for_first_turn = False
        print("[START] 转向完成，启动车道线检测模式")
        return True

    return False


# -------------------- 其他动作的帧计数管理 --------------------
def start_passing_bay_action():
    """开始避让动作"""
    global passing_bay_frame_count, passing_bay_total_frames
    passing_bay_frame_count = 0
    passing_bay_total_frames = int(2.0 / FRAME_DELAY)  # 2秒对应的帧数
    send_serial_data(0x80, 0x20)
    print("[ACTION] 开始避让动作")


def update_passing_bay_action():
    """更新避让动作状态"""
    global passing_bay_frame_count, passing_bay_total_frames

    if passing_bay_frame_count < passing_bay_total_frames:
        passing_bay_frame_count += 1
        if passing_bay_frame_count >= passing_bay_total_frames:
            send_serial_data(0x80, 0x80)
            print("[ACTION] 避让动作完成")
            return True
    return False


def start_warning_action():
    """开始警告动作"""
    global warning_frame_count, warning_total_frames
    warning_frame_count = 0
    warning_total_frames = int(1.0 / FRAME_DELAY)  # 1秒对应的帧数
    send_serial_data(0X70, 0xF0)
    print("[ACTION] 开始警告动作")


def update_warning_action():
    """更新警告动作状态"""
    global warning_frame_count, warning_total_frames

    if warning_frame_count < warning_total_frames:
        warning_frame_count += 1
        if warning_frame_count >= warning_total_frames:
            send_serial_data(0x80, 0x80)
            print("[ACTION] 警告动作完成")
            return True
    return False


def start_zebra_crossing_action():
    """开始人行横道动作"""
    global zebra_crossing_frame_count, zebra_crossing_total_frames, flag_zebra_line, roundabout_mode, roundabout_counter, left_line_lost_count
    zebra_crossing_frame_count = 0
    zebra_crossing_total_frames = int(3.0 / FRAME_DELAY)  # 3秒对应的帧数

    # 如果在环岛模式中检测到人行横道，退出环岛模式
    if roundabout_mode:
        roundabout_mode = False
        roundabout_counter = 0
        left_line_lost_count = 0
        print("[ROUNDABOUT] 检测到人行横道，退出环岛模式")

    with flag_lock:
        flag = 0
    send_serial_data(0x00, 0x80)
    print("[ACTION] 开始人行横道停车")


def update_zebra_crossing_action():
    """更新人行横道动作状态"""
    global zebra_crossing_frame_count, zebra_crossing_total_frames, flag_zebra_line

    if zebra_crossing_frame_count < zebra_crossing_total_frames:
        zebra_crossing_frame_count += 1
        if zebra_crossing_frame_count >= zebra_crossing_total_frames:
            current_speed = 0x80  # 正常速度
            send_serial_data(current_speed, 0x80)
            with flag_lock:
                flag = 1
            print("[ACTION] 人行横道通过，恢复行驶")
            return True
    return False


def start_roundabout_turn_action():
    """开始环岛转向动作"""
    global roundabout_turn_frame_count, roundabout_turn_total_frames
    roundabout_turn_frame_count = 0
    roundabout_turn_total_frames = int(0.5 / FRAME_DELAY)  # 0.5秒对应的帧数
    send_serial_data(0x80, 0xF0)
    print("[ROUNDABOUT] 开始环岛转向")


def update_roundabout_turn_action():
    """更新环岛转向动作状态"""
    global roundabout_turn_frame_count, roundabout_turn_total_frames, roundabout_counter

    if roundabout_turn_frame_count < roundabout_turn_total_frames:
        roundabout_turn_frame_count += 1
        if roundabout_turn_frame_count >= roundabout_turn_total_frames:
            send_serial_data(0x80, 0x80)
            roundabout_counter += 1
            print(f"[ROUNDABOUT] 完成第{roundabout_counter}次出环岛尝试")
            return True
    return False


# -------------------- 更新全局角度 --------------------
def update_global_angle(new_angle_byte):
    """更新全局角度变量"""
    global angle_byte
    with angle_lock:
        angle_byte = new_angle_byte


def get_global_angle():
    """获取全局角度变量"""
    global angle_byte
    with angle_lock:
        return angle_byte


# -------------------- 车道线处理 --------------------
# -------------------- 车道线处理（从detect49.py移植） --------------------
def process_lane(frame):
    global current_state, last_known_track_width, TARGET_OFFSET, track_width_ema, two_lane_streak
    global roundabout_mode, roundabout_counter, left_line_lost_count

    # 1. 启动阶段：固定直行
    global wait_for_first_turn, turn_in_progress, red_light_stop
    if wait_for_first_turn and not turn_in_progress:
        update_global_angle(0x80)  # 更新全局角度为直行
        return frame

    # 2. 转向进行中时，跳过车道线检测
    if turn_in_progress:
        return frame

    # 3. 暂停标志
    with flag_lock:
        if flag == 0:
            return frame

    h, w = frame.shape[:2]
    if w > LANE_CROP:
        scale = LANE_CROP / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        h, w = frame.shape[:2]

    # 颜色掩码 - 红色、黄色、白色车道线同等效益
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    white_mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)

    # 红色检测：使用两个范围（因为红色在HSV中跨越0度边界）
    red_mask1 = cv2.inRange(hsv, RED_LOWER, RED_UPPER)
    red_mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # 合并三种颜色的掩码，给予同等权重
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    mask = cv2.bitwise_or(mask, red_mask)

    # 调试信息：显示各颜色车道线检测情况
    yellow_pixels = np.sum(yellow_mask)
    white_pixels = np.sum(white_mask)
    red_pixels1 = np.sum(red_mask1)
    red_pixels2 = np.sum(red_mask2)
    red_pixels = np.sum(red_mask)
    total_pixels = np.sum(mask)
    # print(f"[COLOR_DEBUG] 黄色:{yellow_pixels}, 白色:{white_pixels}, 红色1:{red_pixels1}, 红色2:{red_pixels2}, 红色总计:{red_pixels}, 总计:{total_pixels}")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # ROI
    pts = np.array([[
        (int(w * ROI_BOTTOM_MARGIN), int(h)),
        (int(w * (1 - ROI_TOP_WIDTH) / 2), int(h * ROI_TOP_HEIGHT)),
        (int(w * (1 + ROI_TOP_WIDTH) / 2), int(h * ROI_TOP_HEIGHT)),
        (int(w * ROI_BOTTOM_WIDTH), int(h))
    ]], dtype=np.int32)
    roi = np.zeros_like(mask)
    cv2.fillPoly(roi, pts, 255)
    mask = cv2.bitwise_and(mask, roi)

    if mask.sum() < 1000:
        return frame

    # 滑动窗口 - 严格限制在ROI区域内
    roi_start_y = int(h * ROI_TOP_HEIGHT)  # ROI顶部高度
    roi_end_y = h  # ROI底部高度

    # 调试信息：显示ROI搜索范围
    # print(f"[ROI] 搜索范围: Y={roi_start_y} 到 {roi_end_y} (总高度={h})")

    # 只在ROI区域内计算直方图
    hist = np.sum(mask[roi_start_y:roi_end_y, :], axis=0)
    mid = hist.shape[0] // 2
    leftx_base = np.argmax(hist[:mid])
    rightx_base = np.argmax(hist[mid:]) + mid

    # 窗口高度基于ROI区域
    win_h = (roi_end_y - roi_start_y) // NWINDOWS
    nonzero = mask.nonzero()
    nony, nonx = np.array(nonzero[0]), np.array(nonzero[1])
    leftx, rightx = leftx_base, rightx_base
    left_idx, right_idx = [], []
    left_support, right_support = 0, 0

    # 贴边优先参数（分位点与限幅）
    EDGE_PCT = 75
    MAX_STEP = int(MARGIN * 0.6)

    for win in range(NWINDOWS):
        # 严格限制在ROI区域内
        y_low = roi_end_y - (win + 1) * win_h
        y_high = roi_end_y - win * win_h

        # 确保不超出ROI边界
        y_low = max(roi_start_y, y_low)
        y_high = min(roi_end_y, y_high)

        xl, xh = max(0, leftx - MARGIN), min(w, leftx + MARGIN)
        xr, xh_r = max(0, rightx - MARGIN), min(w, rightx + MARGIN)

        good_left = ((nony >= y_low) & (nony < y_high) & (nonx >= xl) & (nonx < xh)).nonzero()[0]
        good_right = ((nony >= y_low) & (nony < y_high) & (nonx >= xr) & (nonx < xh_r)).nonzero()[0]
        left_idx.append(good_left)
        right_idx.append(good_right)

        if len(good_left) > MINPIX:
            x_left_win = nonx[good_left]
            # 贴左边：取靠左分位点并对相邻窗口漂移做限幅
            x_new_left = int(np.percentile(x_left_win, EDGE_PCT))
            x_new_left = int(np.clip(x_new_left, leftx - MAX_STEP, leftx + MAX_STEP))
            leftx = x_new_left
            left_support += 1
        if len(good_right) > MINPIX:
            x_right_win = nonx[good_right]
            # 贴右边：取靠右分位点并限幅
            x_new_right = int(np.percentile(x_right_win, 100 - EDGE_PCT))
            x_new_right = int(np.clip(x_new_right, rightx - MAX_STEP, rightx + MAX_STEP))
            rightx = x_new_right
            right_support += 1

    left_idx = np.concatenate(left_idx) if left_idx else np.array([])
    right_idx = np.concatenate(right_idx) if right_idx else np.array([])

    leftx, lefty = nonx[left_idx], nony[left_idx]
    rightx, righty = nonx[right_idx], nony[right_idx]

    # 检测左右车道线
    left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 10 else None
    right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 10 else None

    # -------------------- 环岛模式：左侧线消失检测 --------------------
    if roundabout_mode:
        if left_fit is None:
            left_line_lost_count += 1
            print(f"[ROUNDABOUT] 左侧线丢失计数: {left_line_lost_count}")

            if left_line_lost_count >= LEFT_LINE_LOST_THRESHOLD:
                start_roundabout_turn_action()
                left_line_lost_count = 0
        else:
            # 左侧线存在，重置计数器
            left_line_lost_count = 0

    # 计算中心线拟合结果（用于曲率计算）
    center_fit = None
    ploty = np.linspace(0, h - 1, h)  # 统一计算一次ploty
    target_y_idx = int(h * set_hight)  # 统一计算一次target_y_idx

    if left_fit is not None and right_fit is not None:
        # 两条线都有，计算中心线拟合
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        center_fitx = (left_fitx + right_fitx) / 2
        center_fit = np.polyfit(ploty, center_fitx, 2)
    elif left_fit is not None:
        # 只有左线，推算中心线拟合
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        center_fitx = left_fitx + 260
        center_fit = np.polyfit(ploty, center_fitx, 2)
    elif right_fit is not None:
        # 只有右线，推算中心线拟合
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        center_fitx = right_fitx - 260
        center_fit = np.polyfit(ploty, center_fitx, 2)

    # 更新中心线曲率趋势（在状态机之前）
    centerline_trend, current_curvature = update_centerline_trend(center_fit, target_y_idx)

    # 状态机：根据检测到的车道线更新状态

    # 先进行双线置信度判定（在状态机之前）
    is_true_two_lane = False
    if left_fit is not None and right_fit is not None:
        # 计算车道线位置（使用已计算的ploty）
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # 更新赛道宽度参考值（在特定高度处测量）
        left_pos = left_fitx[target_y_idx]
        right_pos = right_fitx[target_y_idx]
        track_width = right_pos - left_pos

        # —— 双线置信度判定：需同时满足多条件 ——
        gap_px_ok = (rightx_base - leftx_base) >= int(MIN_BIMODAL_GAP_PCT * w)
        # 峰-谷分离度（防把一条线当两条）
        left_peak = np.max(hist[:mid]) if mid > 0 else 1
        right_peak = np.max(hist[mid:]) if mid < len(hist) else 1
        valley = np.min(
            hist[min(leftx_base, rightx_base): max(leftx_base, rightx_base)]) if rightx_base > leftx_base else np.min(
            hist[max(rightx_base, leftx_base):min(rightx_base, leftx_base)])
        valley_ratio_ok = valley <= VALLEY_RATIO_MAX * max(left_peak, right_peak)
        support_ok = (left_support >= MIN_SUPPORT_WINDOWS) and (right_support >= MIN_SUPPORT_WINDOWS)
        width_ok = track_width >= MIN_VALID_TRACK_WIDTH

        is_true_two_lane = gap_px_ok and valley_ratio_ok and support_ok and width_ok
        # print(f"[CHECK] two_lane? gap={gap_px_ok}, valley={valley_ratio_ok}, support=({left_support},{right_support}), width_ok={width_ok}, width={track_width:.1f}")

        # 更新赛道宽度（只有在确认为真双线时）
        if is_true_two_lane and 50 < track_width < 800:  # 合理的赛道宽度范围
            two_lane_streak += 1
            if two_lane_streak >= TWO_LANE_STREAK_MIN:
                if track_width_ema is None:
                    track_width_ema = track_width
                else:
                    track_width_ema = EMA_ALPHA * track_width + (1 - EMA_ALPHA) * track_width_ema
                last_known_track_width = track_width_ema
                TARGET_OFFSET = last_known_track_width / 2
                # print(f"[STATE] 更新赛道宽度(EMA): {last_known_track_width:.1f}px, 原始={track_width:.1f}")

    # 状态机：根据检测结果和置信度更新状态
    if left_fit is not None and right_fit is not None and is_true_two_lane:
        # 状态A: 两条线都可见且置信度高
        current_state = BOTH_LINES_SEEN
        # print(f"[STATE] 两条线都可见（真双线）")

        # 计算车道线位置（使用已计算的ploty）
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # 优先检查单边检测（无论是否真双线）
        if last_known_track_width < 480:  # 赛道宽度小于480时强制左单边检测
            current_state = ONLY_LEFT_SEEN
            # 使用实际赛道宽度的一半，限制在合理范围内
            actual_half_width = (260) if last_known_track_width > 0 else 110
            center_x = left_fitx + max(actual_half_width, SAFE_INWARD)
            right_fitx = left_fitx + last_known_track_width if last_known_track_width > 0 else left_fitx + 220
            # print(f"[STATE] 赛道宽度<480强制左单边检测，实际半宽={actual_half_width:.1f}")
        elif FORCE_SINGLE_SIDE and centerline_trend == 1:  # 右转趋势，只看左边
            current_state = ONLY_LEFT_SEEN
            actual_half_width = (260) if last_known_track_width > 0 else 110
            center_x = left_fitx + max(actual_half_width, SAFE_INWARD)
            right_fitx = left_fitx + last_known_track_width if last_known_track_width > 0 else left_fitx + 220
            # print("[STATE] 右转趋势单边检测 - 只看左边")
        elif FORCE_SINGLE_SIDE and centerline_trend == -1:  # 左转趋势，只看右边
            current_state = ONLY_RIGHT_SEEN
            actual_half_width = (260) if last_known_track_width > 0 else 110
            center_x = right_fitx - max(actual_half_width, SAFE_INWARD)
            left_fitx = right_fitx - last_known_track_width if last_known_track_width > 0 else right_fitx - 220
            # print("[STATE] 左转趋势单边检测 - 只看右边")
        else:
            # 真实双线（无单边检测趋势）
            center_x = (left_fitx + right_fitx) / 2
    elif left_fit is not None and right_fit is not None and not is_true_two_lane:
        # 状态A': 检测到两条线但置信度低，退化为单线
        two_lane_streak = 0
        if right_support >= left_support:
            current_state = ONLY_RIGHT_SEEN
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
            actual_half_width = (260) if last_known_track_width > 0 else 110
            center_x = right_fitx - max(actual_half_width, SAFE_INWARD)
            left_fitx = right_fitx - last_known_track_width if last_known_track_width > 0 else right_fitx - 220
            # print("[STATE] 判定为右单线（假双线剔除）")
        else:
            current_state = ONLY_LEFT_SEEN
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            actual_half_width = (260) if last_known_track_width > 0 else 110
            center_x = left_fitx + max(actual_half_width, SAFE_INWARD)
            right_fitx = left_fitx + last_known_track_width if last_known_track_width > 0 else left_fitx + 220
            # print("[STATE] 判定为左单线（假双线剔除）")

    elif right_fit is not None:
        # 状态B: 只看到右边线
        current_state = ONLY_RIGHT_SEEN
        # print(f"[STATE] 只看到右边线")

        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # 计算虚拟中心线：右边线位置 - 半个赛道宽度（EMA/保底 + 安全内偏）
        half_width = (260) if (track_width_ema is not None) else (FALLBACK_TRACK_WIDTH / 2)
        center_x = right_fitx - max(half_width, SAFE_INWARD)
        left_fitx = right_fitx - 520  # 推算左边线位置用于显示

    elif left_fit is not None:
        # 状态C: 只看到左边线
        current_state = ONLY_LEFT_SEEN
        # print(f"[STATE] 只看到左边线")

        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]

        # 计算虚拟中心线：左边线位置 + 半个赛道宽度（EMA/保底 + 安全内偏）
        half_width = (260) if (track_width_ema is not None) else (FALLBACK_TRACK_WIDTH / 2)
        center_x = left_fitx + max(half_width, SAFE_INWARD)
        right_fitx = left_fitx + 520  # 推算右边线位置用于显示

    else:
        # 状态D: 所有线都丢失
        current_state = NO_LINES_SEEN
        # print(f"[STATE] 所有线都丢失")

        # 保持上一次的中心线或使用默认值
        ploty = np.linspace(0, h - 1, h)
        center_x = np.full_like(ploty, w // 2)  # 默认直行
        left_fitx = center_x - TARGET_OFFSET
        right_fitx = center_x + TARGET_OFFSET

    # 画车道+中心线 - 严格限制在ROI区域内
    color_warp = np.zeros_like(frame)

    # 创建ROI掩码，只允许在ROI区域内绘制
    roi_mask = np.zeros_like(frame[:, :, 0])
    cv2.fillPoly(roi_mask, [pts], 255)

    # 限制车道线绘制范围到ROI区域
    roi_start_y = int(h * ROI_TOP_HEIGHT)
    roi_end_y = h

    # 只绘制ROI区域内的车道线
    ploty_roi = np.linspace(roi_start_y, roi_end_y - 1, roi_end_y - roi_start_y)
    left_fitx_roi = left_fitx[roi_start_y:roi_end_y]
    right_fitx_roi = right_fitx[roi_start_y:roi_end_y]
    center_x_roi = center_x[roi_start_y:roi_end_y]

    # 绘制车道填充区域（绿色）——仅在真实双线状态下绘制
    if current_state == BOTH_LINES_SEEN:
        pts_left_roi = np.array([np.transpose(np.vstack([left_fitx_roi, ploty_roi]))])
        pts_right_roi = np.array([np.flipud(np.transpose(np.vstack([right_fitx_roi, ploty_roi])))])
        cv2.fillPoly(color_warp, np.int_([np.hstack((pts_left_roi, pts_right_roi))]), (0, 255, 0))

    # 绘制虚拟中心线（黄色）- 只在ROI区域内
    center_pts_roi = np.column_stack([center_x_roi, ploty_roi]).astype(np.int32)
    for i in range(len(center_pts_roi) - 1):
        cv2.line(color_warp, tuple(center_pts_roi[i]), tuple(center_pts_roi[i + 1]), (0, 255, 255), 3)

    # 计算角度（图像底部中点 → 中心线在y=0.8h处的点）
    car_x = w // 2
    car_y = h
    target_y = int(h * set_hight)  # 使用y=0.8h位置的点
    target_x = int(center_x[target_y])
    dx = target_x - car_x
    dy = car_y - target_y

    # 修正角度计算：左转为负角度，右转为正角度
    angle_rad = math.atan2(dx, dy)  # 注意：dx在前，dy在后
    angle_deg = math.degrees(angle_rad) * 1.4

    # 角度范围限制：-45°到+45°（避免过度转向）
    angle_deg = max(-45, min(45, angle_deg))

    # 修正映射：左转(-45°)→255, 直行(0°)→128, 右转(+45°)→0
    # 下位机协议：>0x80=左拐, <0x80=右拐
    calculated_angle_byte = int((45 - angle_deg) * 255 / 90) & 0xFF

    # 更新全局角度变量（不发送串口）
    update_global_angle(calculated_angle_byte)

    # 在图像上标记计算角度的点 - 只在ROI区域内绘制
    if roi_start_y <= car_y <= roi_end_y:
        cv2.circle(color_warp, (car_x, car_y), 8, (255, 0, 0), -1)  # 车辆位置（蓝色）

    if roi_start_y <= target_y <= roi_end_y:
        cv2.circle(color_warp, (target_x, target_y), 8, (0, 0, 255), -1)  # 目标点位置（红色）
        cv2.line(color_warp, (car_x, car_y), (target_x, target_y), (255, 255, 0), 2)  # 连线（青色）

    # 绘制ROI区域边界（红色线条）
    cv2.polylines(color_warp, [pts], True, (0, 0, 255), 3)

    # -------------------- 新增：在y=0.5高度处绘制水平线，连接深色区域的左右边界 --------------------
    y_half = int(h * 0.5)  # y=0.5高度

    # 计算在y=0.5高度处的左右边界位置（直接连接到图像边框）
    if current_state == BOTH_LINES_SEEN:
        # 双线状态：使用实际检测到的左右车道线位置
        left_x_half = int(left_fitx[y_half])
        right_x_half = int(right_fitx[y_half])
    elif current_state == ONLY_LEFT_SEEN:
        # 只有左线：左边界使用检测到的左线，右边界直接到图像右边框
        left_x_half = int(left_fitx[y_half])
        right_x_half = w - 1  # 图像最右边
    elif current_state == ONLY_RIGHT_SEEN:
        # 只有右线：右边界使用检测到的右线，左边界直接到图像左边框
        right_x_half = int(right_fitx[y_half])
        left_x_half = 0  # 图像最左边
    else:
        # 无线状态：直接连接左右边框
        left_x_half = 0  # 图像最左边
        right_x_half = w - 1  # 图像最右边

    # 绘制深色区域的水平连接线（深灰色，线宽3）
    cv2.line(color_warp, (left_x_half, y_half), (right_x_half, y_half), (64, 64, 64), 3)

    # 在水平线两端绘制标记点
    cv2.circle(color_warp, (left_x_half, y_half), 6, (64, 64, 64), -1)  # 左端点
    cv2.circle(color_warp, (right_x_half, y_half), 6, (64, 64, 64), -1)  # 右端点

    # 在图像上显示角度信息
    direction = "左转" if angle_deg < -5 else "右转" if angle_deg > 5 else "直行"
    cv2.putText(color_warp, f"Angle:{angle_deg:.1f}*", (10, h - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(color_warp, f"Dir:{direction}", (10, h - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 0, 0) if angle_deg < -5 else (0, 255, 0) if angle_deg > 5 else (255, 255, 0), 2)
    cv2.putText(color_warp, f"Byte:{calculated_angle_byte:3d}(0x{calculated_angle_byte:02X})", (10, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 显示状态和参数信息
    state_names = ["double", "left", "right", "none"]
    trend_names = ["直行", "右转", "左转"]
    cv2.putText(color_warp, f"State:{state_names[current_state]}", (400, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(color_warp, f"Trend:{trend_names[centerline_trend + 1]}", (400, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(color_warp, f"Curvature:{current_curvature:.6f}", (400, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 显示环岛模式信息
    if roundabout_mode:
        cv2.putText(color_warp, f"Roundabout:ON Count:{roundabout_counter}", (400, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(color_warp, f"LeftLost:{left_line_lost_count}", (400, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return cv2.addWeighted(frame, 1, color_warp, 0.3, 0)


# -------------------- 曲率趋势 --------------------
def calculate_centerline_curvature(center_fit, y_eval):
    try:
        if center_fit is None or len(center_fit) != 3:
            return 0.0
        a, b, c = center_fit[0], center_fit[1], center_fit[2]
        denominator = 1 + (2 * a * y_eval + b) ** 2
        if denominator == 0:
            return 0.0
        curvature = abs(2 * a) / (denominator ** (3 / 2))
        return curvature if a > 0 else -curvature
    except Exception as e:
        print(f"[ERROR] 曲率计算失败: {e}")
        return 0.0


def update_centerline_trend(center_fit, y_eval):
    global curvature_history, centerline_trend, FORCE_SINGLE_SIDE
    current_curvature = calculate_centerline_curvature(center_fit, y_eval)
    curvature_history.append(current_curvature)
    if len(curvature_history) > CURVATURE_HISTORY:
        curvature_history.pop(0)
    if len(curvature_history) < CURVATURE_HISTORY:
        return centerline_trend, current_curvature
    avg_curvature = sum(curvature_history) / len(curvature_history)
    if avg_curvature > CURVATURE_THRESHOLD:
        new_trend = 1
    elif avg_curvature < -CURVATURE_THRESHOLD:
        new_trend = -1
    else:
        new_trend = 0
    if new_trend != centerline_trend:
        centerline_trend = new_trend
        FORCE_SINGLE_SIDE = (centerline_trend != 0)
    return centerline_trend, current_curvature


# -------------------- 采集线程 --------------------
def capture_thread(cam_id, q):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f'[FATAL] Cam {cam_id} open failed')
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    for _ in range(10):
        cap.read()

    print(
        f'[INFO] Cam {cam_id} started: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} @ {cap.get(cv2.CAP_PROP_FPS)}fps')

    while not shutdown_flag:
        ret, frame = cap.read()
        if not ret:
            continue
        while q.qsize() > 1:
            try:
                q.get_nowait()
            except queue.Empty:
                break
        q.put(frame)
    cap.release()
    print(f'[INFO] Cam {cam_id} 已关闭')


# -------------------- 推理线程 --------------------
def infer_thread(model, q_in, q_out, lane_mode):
    global flag, wait_for_first_turn, turn_in_progress, flag_zebra_line, red_light_stop
    global roundabout_mode, roundabout_counter, left_line_lost_count

    speed_limit_active = False
    normal_speed = 0x80
    limited_speed = 0x20

    while not shutdown_flag:
        try:
            frame = q_in.get(timeout=0.1)
        except queue.Empty:
            continue

        # 更新所有进行中的动作
        update_turn_action()
        update_passing_bay_action()
        update_warning_action()
        update_zebra_crossing_action()
        update_roundabout_turn_action()

        if lane_mode:
            out = process_lane(frame)
        else:
            results = model(frame, conf=CONF_THRES, imgsz=320, verbose=False)[0]
            if results.boxes and results.boxes.cls.numel():
                boxes = results.boxes.xyxy
                clses = results.boxes.cls.long()
                confs = results.boxes.conf
                names = results.names

                # 初始化检测标志
                limitation_detected = warning_detected = passing_bay_detected = False
                green_light_detected = red_light_detected = turn_right_detected = False
                turn_left_detected = pedestrain_passagemay_detected = False

                for i, (box, cls_id, conf) in enumerate(zip(boxes, clses, confs)):
                    name = names[cls_id.item()]
                    x1, y1, x2, y2 = box.int().tolist()
                    box_area = (x2 - x1) * (y2 - y1)
                    total_pixels = frame.shape[1] * frame.shape[0]
                    area_ratio = box_area / total_pixels

                    if name == "limitation" and area_ratio >= 0.15:
                        limitation_detected = True
                        print(f"[AREA] {name}: {area_ratio:.4f}")
                    elif name == "warning" and area_ratio >= 0.008:
                        warning_detected = True
                        print(f"[AREA] {name}: {area_ratio:.4f}")
                    elif name == "passing_bay" and area_ratio >= 0.003:
                        passing_bay_detected = True
                        print(f"[AREA] {name}: {area_ratio:.4f}")
                    elif name == "green" and area_ratio >= 0.12:
                        green_light_detected = True
                        print(f"[AREA] {name}: {area_ratio:.4f}")
                    elif name == "red" and area_ratio >= 0.12:
                        red_light_detected = True
                        print(f"[AREA] {name}: {area_ratio:.4f}")
                    elif name == "trun_right" and area_ratio >= 0.007:
                        turn_right_detected = True
                        print(f"[AREA] {name}: {area_ratio:.4f}")
                    elif name == "trun_left" and area_ratio >= 0.003:
                        turn_left_detected = True
                        print(f"[AREA] {name}: {area_ratio:.4f}")
                    elif name == "PPC" and area_ratio >= 0.012:
                        pedestrain_passagemay_detected = True
                        print(f"[AREA] {name}: {area_ratio:.4f}")

                    # 启动阶段：首次识别到左转/右转标志，执行转向动作
                    if wait_for_first_turn and name in ("trun_left", "trun_right") and (
                            turn_left_detected or turn_right_detected) and not turn_in_progress:
                        print(f"[START] 首次识别到 {name}，开始执行转向动作")
                        start_turn_action(name)
                        break

                # 只有在车道线模式才处理其他标志
                if not wait_for_first_turn and not turn_in_progress:
                    # 暂停车道线检测（除了限速标志之外的所有标志）
                    other_signs_detected = any([warning_detected, passing_bay_detected,
                                                green_light_detected, red_light_detected])

                    if other_signs_detected:
                        with flag_lock:
                            if flag == 1:
                                flag = 0
                                print("[FLAG] 检测到非限速标志，暂停车道线检测")
                                reset_flag_after_frames(15)  # 1秒对应的帧数

                    # 人行横道单独处理
                    if (pedestrain_passagemay_detected and flag_zebra_line == 0 and can_execute_task("PPC")
                            and zebra_crossing_frame_count == 0):
                        print("============== 检测到人行横道 ==============")
                        flag_zebra_line = 1
                        mark_task_executed("PPC")
                        start_zebra_crossing_action()

                    # 交通标志动作（其他人行横道之外的标志）
                    if warning_detected and can_execute_task("warning") and warning_frame_count == 0:
                        print("============== 检测到警告标志 ==============")
                        mark_task_executed("warning")
                        start_warning_action()

                    elif passing_bay_detected and can_execute_task("passing_bay") and passing_bay_frame_count == 0:
                        print("============== 检测到避让标志 ==============")
                        mark_task_executed("passing_bay")
                        start_passing_bay_action()

                    elif red_light_detected and not red_light_stop and can_execute_task("red"):
                        print("============== 检测到红灯 ==============")
                        red_light_stop = True
                        send_serial_data(0x00, 0x80)
                        mark_task_executed("red")

                    elif green_light_detected and red_light_stop and can_execute_task("green"):
                        print("============== 检测到绿灯 ==============")
                        red_light_stop = False
                        current_speed = limited_speed if speed_limit_active else normal_speed
                        send_serial_data(current_speed, 0x80)
                        mark_task_executed("green")

                    # 限速
                    if limitation_detected and not speed_limit_active:
                        speed_limit_active = True
                        if not red_light_stop:
                            send_serial_data(limited_speed, 0x80)
                    elif not limitation_detected and speed_limit_active:
                        speed_limit_active = False
                        if not red_light_stop:
                            send_serial_data(normal_speed, 0x80)

                # 画框
                out = frame.copy()
                for (x1, y1, x2, y2), cls_id, cf in zip(boxes.int().tolist(), clses.tolist(), confs.tolist()):
                    name = names[cls_id]
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # 在画框时也显示面积比例
                    box_area = (x2 - x1) * (y2 - y1)
                    total_pixels = frame.shape[1] * frame.shape[0]
                    area_ratio = box_area / total_pixels
                    task_status = "可执行" if can_execute_task(name) else "已执行"
                    cv2.putText(out, f'{name} {cf:.2f} area:{area_ratio:.3f} [{task_status}]',
                                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 只有在没有检测到需要特殊处理的交通标志时才发送车道线角度
                if not any([limitation_detected, warning_detected, passing_bay_detected,
                            green_light_detected, red_light_detected, turn_right_detected,
                            turn_left_detected, pedestrain_passagemay_detected]):
                    current_angle = get_global_angle()
                    current_speed = limited_speed if speed_limit_active else normal_speed
                    send_serial_data(current_speed, current_angle)

            else:
                # 没有检测到任何目标，发送车道线角度
                out = frame
                current_angle = get_global_angle()
                current_speed = limited_speed if speed_limit_active else normal_speed
                send_serial_data(current_speed, current_angle)

            # 显示状态信息
            task_status = get_task_status()
            status_text = f"Speed: {'LIMITED' if speed_limit_active else 'NORMAL'}"
            status_text2 = f"RedLight: {'STOP' if red_light_stop else 'GO'}"
            status_text3 = f"Mode: {'LANE' if not wait_for_first_turn else 'START (等待转向)'}"
            status_text4 = f"Turning: {'YES' if turn_in_progress else 'NO'}"

            # 显示任务状态
            active_tasks = [k for k, v in task_status.items() if v == 1]
            inactive_tasks = [k for k, v in task_status.items() if v == 0]
            status_text5 = f"Active: {len(active_tasks)}"
            status_text6 = f"Done: {len(inactive_tasks)}"

            cv2.putText(out, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if speed_limit_active else (0, 255, 0), 2)
            cv2.putText(out, status_text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if red_light_stop else (0, 255, 0), 2)
            cv2.putText(out, status_text3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if not wait_for_first_turn else (255, 255, 0), 2)
            cv2.putText(out, status_text4, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 0, 0) if turn_in_progress else (0, 255, 0), 2)
            cv2.putText(out, status_text5, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)
            cv2.putText(out, status_text6, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 0, 0), 2)

        if q_out.full():
            q_out.get()
        q_out.put(out)

    print(f'[INFO] {"车道线" if lane_mode else "YOLO"} 推理线程已退出')


# -------------------- 主函数 --------------------
def main():
    global shutdown_flag
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    model = YOLO(ROOT / 'best.pt')
    q_cap0 = queue.Queue(maxsize=3)
    q_cap1 = queue.Queue(maxsize=3)
    q_out0 = queue.Queue(maxsize=3)
    q_out1 = queue.Queue(maxsize=3)

    threading.Thread(target=capture_thread, args=(2, q_cap0), daemon=True).start()
    threading.Thread(target=capture_thread, args=(0, q_cap1), daemon=True).start()
    threading.Thread(target=infer_thread, args=(model, q_cap0, q_out0, False), daemon=True).start()
    threading.Thread(target=infer_thread, args=(None, q_cap1, q_out1, True), daemon=True).start()

    print('[INFO] 按 Q 退出，或按 Ctrl+C 安全退出')
    print('[INFO] 启动阶段：固定直行，等待识别到左转/右转标志并执行转向...')

    try:
        while not shutdown_flag:
            if not q_out0.empty() and not q_out1.empty():
                cv2.imshow('YOLO Traffic (Cam0)', q_out0.get())
                cv2.imshow('Lane Detect (Cam1)', q_out1.get())
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_flag = True
        time.sleep(0.2)

        for _ in range(5):
            send_serial_data(0x00, 0x80)
            time.sleep(0.05)

        if ser and ser.is_open:
            ser.close()
        cv2.destroyAllWindows()
        print("[INFO] 程序已安全退出")


if __name__ == '__main__':
    main()