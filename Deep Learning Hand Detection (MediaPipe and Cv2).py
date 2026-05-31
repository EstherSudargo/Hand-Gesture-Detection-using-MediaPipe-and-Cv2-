import cv2
import mediapipe as mp
import pyautogui
import numpy as np  # 引入 numpy 處理座標映射 --> Imported numpy to handle coordinate mapping

# ==========================================
# 1. 系統與鏡頭參數設定/System & Camera Parameter Settings
# ==========================================
pyautogui.FAILSAFE = False  
pyautogui.PAUSE = 0         
SCREEN_W, SCREEN_H = pyautogui.size()

# 鏡頭解析度設定/Camera resolution settings
CAM_W, CAM_H = 640, 480

# ⭐ 新增：虛擬滑鼠墊邊界縮減 (Frame Reduction)
# 數值越大，手需要移動的實際範圍越小，滑鼠感覺越靈敏
FRAME_R = 150 

print("正在啟動升級版原力滑鼠引擎...")

# ==========================================
# 2. 初始化 MediaPipe 模型/Initialize Mdeiapipe
# ==========================================
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6
)
recognizer = GestureRecognizer.create_from_options(options)

# ==========================================
# 3. 滑鼠控制狀態與平滑化變數設定/Mouse Control State & Smoothing Variables Settings
# ==========================================
is_dragging = False
SMOOTHING = 5.0  
smooth_x, smooth_y = SCREEN_W / 2, SCREEN_H / 2

# ==========================================
# 4. 啟動攝影機/Start Camera
# ==========================================
CAMERA_INDEX = 0  
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    
    # ⭐ 畫出虛擬滑鼠墊 (Active Zone) 的粉紅色提示框
    # 只要手在這個框內移動，就能觸及螢幕的任何角落
    cv2.rectangle(frame, (FRAME_R, FRAME_R), (CAM_W - FRAME_R, CAM_H - FRAME_R), (255, 0, 255), 2)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = recognizer.recognize(mp_image)

    # ==========================================
    # 5. 核心邏輯：游標控制與動態座標映射
    # ==========================================
    if result.hand_landmarks and result.gestures:
        landmarks = result.hand_landmarks[0]
        top_gesture = result.gestures[0][0].category_name
        
        # 使用 Landmark 9 (中指根部)
        tracking_point = landmarks[9]
        
        # 取得追蹤點在鏡頭畫面上的「實際像素座標」
        cam_x_pixel = tracking_point.x * CAM_W
        cam_y_pixel = tracking_point.y * CAM_H

        # ⭐ 新增：使用 numpy.interp 進行動態座標映射！
        # 將「鏡頭內部矩形的座標」線性映射到「螢幕的實際座標」
        target_x = np.interp(cam_x_pixel, (FRAME_R, CAM_W - FRAME_R), (0, SCREEN_W))
        target_y = np.interp(cam_y_pixel, (FRAME_R, CAM_H - FRAME_R), (0, SCREEN_H))
        
        # 平滑化運算
        smooth_x += (target_x - smooth_x) / SMOOTHING
        smooth_y += (target_y - smooth_y) / SMOOTHING

        # 執行游標移動
        pyautogui.moveTo(int(smooth_x), int(smooth_y))

        if top_gesture == "Closed_Fist":
            if not is_dragging:
                pyautogui.mouseDown(button='left')
                is_dragging = True
            
            cv2.circle(frame, (int(cam_x_pixel), int(cam_y_pixel)), 15, (0, 0, 255), -1)
            cv2.putText(frame, "DRAGGING", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        elif top_gesture in ["Open_Palm", "Pointing_Up", "Victory", "Thumb_Up", "Thumb_Down"]:
            if is_dragging:
                pyautogui.mouseUp(button='left')
                is_dragging = False
            
            cv2.circle(frame, (int(cam_x_pixel), int(cam_y_pixel)), 15, (0, 255, 0), -1)
            cv2.putText(frame, "HOVERING", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow('Jedi Mouse - Gesture Control', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if is_dragging:
    pyautogui.mouseUp(button='left')

cap.release()
cv2.destroyAllWindows()