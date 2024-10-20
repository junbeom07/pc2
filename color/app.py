# app.py
from flask import Flask, render_template, Response
import cv2
import dlib
import numpy as np
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

# dlib 얼굴 감지기 및 랜드마크 감지기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 랜드마크 모델 파일 경로

def determine_skin_tone(frame, landmarks):
    # 랜드마크 좌표 추출
    points = []
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        points.append((x, y))

    # 피부 영역 마스크 생성
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.array(points[0:17]), 255)  # 얼굴 윤곽만 포함

    # 눈과 입 내부를 제외하기 위해 선 그리기
    cv2.fillConvexPoly(mask, np.array(points[36:48]), 0)  # 눈 내부 제외
    cv2.fillConvexPoly(mask, np.array(points[48:68]), 0)  # 입 내부 제외

    # 눈동자 제외 (왼쪽 눈동자)
    cv2.fillConvexPoly(mask, np.array([points[36], points[37], points[38], points[39], points[40], points[41]]), 0)  # 왼쪽 눈동자 제외
    # 눈동자 제외 (오른쪽 눈동자)
    cv2.fillConvexPoly(mask, np.array([points[42], points[43], points[44], points[45], points[46], points[47]]), 0)  # 오른쪽 눈동자 제외

    # 마스크 적용
    skin = cv2.bitwise_and(frame, frame, mask=mask)

    # 피부 영역의 Lab 색 공간으로 변환
    lab_skin = cv2.cvtColor(skin, cv2.COLOR_BGR2Lab)

    # 마스크를 사용하여 b값 추출
    b_channel = lab_skin[:, :, 2]  # Lab 색 공간의 b 채널
    b_values = b_channel[mask > 0]  # 피부 영역의 b값만 추출

    # 평균 b값 계산
    if b_values.size > 0:
        avg_b = np.mean(b_values)
        # 쿨톤과 웜톤 판단
        if avg_b < 128:
            return "cool"
        else:
            return "warm"
    
    return "Unknown"  # b값이 없을 경우


def determine_color_text(frame, landmarks):
    # 랜드마크 좌표 추출
    points = []
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        points.append((x, y))

    # 피부 영역 마스크 생성
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.array(points[0:17]), 255)  # 얼굴 윤곽만 포함

    # 눈과 입 내부를 제외하기 위해 선 그리기
    cv2.fillConvexPoly(mask, np.array(points[36:48]), 0)  # 눈 내부 제외
    cv2.fillConvexPoly(mask, np.array(points[48:68]), 0)  # 입 내부 제외

    # 마스크 적용
    skin = cv2.bitwise_and(frame, frame, mask=mask)

    # HSV 색 공간으로 변환
    hsv_skin = cv2.cvtColor(skin, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_skin)

    # 평균 S와 V 값 계산
    if cv2.countNonZero(mask) > 0:
        avg_s = np.mean(s[mask > 0])
        avg_v = np.mean(v[mask > 0])

        # 색상 판단 로직
        if avg_s >= 226:
            color_text = "Vivid"
        elif 142 <= avg_s < 226:
            if abs(avg_v - 180) < abs(avg_v - 240) and abs(avg_v - 180) < abs(avg_v - 250):
                color_text = "Deep"
            elif abs(avg_v - 240) < abs(avg_v - 180) and abs(avg_v - 240) < abs(avg_v - 250):
                color_text = "Strong"
            else:
                color_text = "Bright"
        elif 57 <= avg_s < 142:
            if abs(avg_v - 31) < abs(avg_v - 102) and abs(avg_v - 31) < abs(avg_v - 182) and abs(avg_v - 31) < abs(avg_v - 225):
                color_text = "Dark"
            elif abs(avg_v - 102) < abs(avg_v - 31) and abs(avg_v - 102) < abs(avg_v - 182) and abs(avg_v - 102) < abs(avg_v - 225):
                color_text = "Dull"
            elif abs(avg_v - 182) < abs(avg_v - 31) and abs(avg_v - 182) < abs(avg_v - 102) and abs(avg_v - 182) < abs(avg_v - 225):
                color_text = "Soft"
            else:
                color_text = "Light"
        else:
            if abs(avg_v - 31) < abs(avg_v - 102) and abs(avg_v - 31) < abs(avg_v - 182) and abs(avg_v - 31) < abs(avg_v - 225):
                color_text = "Dark Grayish"
            elif abs(avg_v - 102) < abs(avg_v - 31) and abs(avg_v - 102) < abs(avg_v - 182) and abs(avg_v - 102) < abs(avg_v - 225):
                color_text = "Grayish"
            elif abs(avg_v - 182) < abs(avg_v - 31) and abs(avg_v - 182) < abs(avg_v - 102) and abs(avg_v - 182) < abs(avg_v - 225):
                color_text = "Light Grayish"
            else:
                color_text = "Pale"
        
        return color_text  # 색상 텍스트 반환

def gen_frames():
    camera = cv2.VideoCapture(0)  # Open webcam
    while True:
        success, frame = camera.read()  # Read frame
        if not success:
            break
        else:
            # 얼굴 감지
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 프레임을 그레이스케일로 변환
            faces = detector(gray)  # 그레이스케일 이미지에서 얼굴 감지
            color = "Unknown"  # 기본값
            color_text = "Unknown"  # 색상 텍스트 기본값
            hsv_values = None  # HSV 값을 저장할 변수

            for face in faces:
                # 랜드마크 감지
                landmarks = predictor(gray, face)

                # 개인 색상 결정
                color = determine_skin_tone(frame, landmarks)
                color_text = determine_color_text(frame, landmarks)  # 색상 텍스트 결정

                # 마스크 적용
                points = []
                for n in range(68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    points.append((x, y))

                # 피부 영역 마스크 생성
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillConvexPoly(mask, np.array(points[0:17]), 255)  # 얼굴 윤곽만 포함

                # 눈과 입 내부를 제외하기 위해 선 그리기
                cv2.fillConvexPoly(mask, np.array(points[36:48]), 0)  # 눈 내부 제외
                cv2.fillConvexPoly(mask, np.array(points[48:68]), 0)  # 입 내부 제외

                # 눈동자 제외 (왼쪽 눈동자)
                cv2.fillConvexPoly(mask, np.array([points[36], points[37], points[38], points[39], points[40], points[41]]), 0)  # 왼쪽 눈동자 제외
                # 눈동자 제외 (오른쪽 눈동자)
                cv2.fillConvexPoly(mask, np.array([points[42], points[43], points[44], points[45], points[46], points[47]]), 0)  # 오른쪽 눈동자 제외

                # 마스크가 적용된 부분의 HSV 값 측정
                masked_skin = cv2.bitwise_and(frame, frame, mask=mask)
                hsv_masked_skin = cv2.cvtColor(masked_skin, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv_masked_skin)

                # 평균 HSV 값 계산
                if cv2.countNonZero(mask) > 0:
                    avg_h = np.mean(h[mask > 0])
                    avg_s = np.mean(s[mask > 0])
                    avg_v = np.mean(v[mask > 0])
                    hsv_values = (avg_h, avg_s, avg_v)

                # 마스크가 적용되지 않은 부분을 어둡게 변경
                frame[mask == 0] = frame[mask == 0] * 0.5  # 어둡게 설정 (50% 밝기)

                # 마스크가 적용된 얼굴 영역 그리기
                cv2.addWeighted(masked_skin, 0.5, frame, 0.5, 0, frame)

            # 결과를 프레임에 추가
            cv2.putText(frame, f'Personal Color: {color}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            cv2.putText(frame, f'Color: {color_text}', (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # HSV 값 표시
            if hsv_values:
                cv2.putText(frame, f'Avg HSV: H={hsv_values[0]:.2f}, S={hsv_values[1]:.2f}, V={hsv_values[2]:.2f}', 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)







