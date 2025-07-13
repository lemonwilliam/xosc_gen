import sys
import subprocess
import time
import vlc
import win32gui
import win32con
import win32process
import win32api
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QFrame
)
from PyQt5.QtCore import QTimer, Qt
from moviepy.editor import VideoFileClip


class VLCPlayer(QWidget):
    def __init__(self, vedio_path, esmini_path, osc_path):
        super().__init__()
        self.setWindowTitle("VLC Player + esmini Controller")
        self.width = 1900
        self.height = 1080
        self.resize(self.width//2, self.height)  

        # 影片播放區 (右半邊)
        self.video_frame = QFrame(self)
        self.video_frame.setStyleSheet("background-color: black;")
        self.video_frame.setFixedSize(self.width//2, self.height)

        # 控制區容器 (疊按鈕+狀態，右半邊底部中央)
        self.controls_container = QWidget(self)
        self.controls_container.setFixedSize(400, 50)
        ctrl_x = (950 - 400) // 2
        ctrl_y = 1000 - 50
        self.controls_container.move(ctrl_x, ctrl_y)

        # 建立控制元件
        self.play_btn = QPushButton("▶ Play")
        self.pause_btn = QPushButton("⏸ Pause")
        self.stop_btn = QPushButton("⏹ Stop")
        self.status_label = QLabel("Status: Stopped")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("background-color: rgba(255,255,255,0.1);")

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.play_btn)
        btn_layout.addWidget(self.pause_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.setSpacing(20)

        controls_layout = QVBoxLayout(self.controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addLayout(btn_layout)
        controls_layout.addWidget(self.status_label)

   
        # VLC 初始化
        self.instance = vlc.Instance()
        self.mediaplayer = self.instance.media_player_new()

        # 路徑設定
        self.video_path = vedio_path
        self.esmini_path = esmini_path
        self.osc_path = osc_path

        self.esmini_proc = None

        self.timer = QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.update_status)

        if sys.platform.startswith('linux'):
            self.mediaplayer.set_xwindow(self.video_frame.winId())
        elif sys.platform == "win32":
            self.mediaplayer.set_hwnd(self.video_frame.winId())
        elif sys.platform == "darwin":
            self.mediaplayer.set_nsobject(int(self.video_frame.winId()))

        # 綁定按鈕事件
        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        self.stop_btn.clicked.connect(self.stop)

    def start_esmini(self):
        if not self.esmini_proc or self.esmini_proc.poll() is not None:
            self.esmini_proc = subprocess.Popen([
            self.esmini_path,
            "--window", "950", "267", "950", "534",
            "--osc", self.osc_path, "--camera_mode", "top", "--trail_mode", "0"
        ])


    def stop_esmini(self):
        if self.esmini_proc:
            self.esmini_proc.terminate()
            self.esmini_proc = None
            
    def esmini_space(self):
        if self.esmini_proc:
            # 找出 esmini 視窗
            def enum_handler(hwnd, param):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if "esmini" in title.lower():  # 根據視窗標題判斷
                        param.append(hwnd)
    
            hwnds = []
            win32gui.EnumWindows(enum_handler, hwnds)
    
            if hwnds:
                hwnd = hwnds[0]
                win32gui.SetForegroundWindow(hwnd)  # 切到 esmini 視窗
                time.sleep(0.1)  # 稍微等待聚焦成功
                
                win32api.PostMessage(hwnd, win32con.WM_KEYDOWN, win32con.VK_SPACE, 0)
                win32api.PostMessage(hwnd, win32con.WM_KEYUP, win32con.VK_SPACE, 0)

    def play(self):
        state = self.mediaplayer.get_state()
        if state == vlc.State.Paused:
            self.mediaplayer.play()
            #self.esmini_space() --> simute press space to continue 
            self.timer.start()
            self.status_label.setText("Status: Playing")
        elif not self.mediaplayer.is_playing():
            media = self.instance.media_new(self.video_path)
            self.mediaplayer.set_media(media)
            self.mediaplayer.play()
            self.start_esmini()
            self.timer.start()
            self.status_label.setText("Status: Playing")
    
    def pause(self):
        if self.mediaplayer.is_playing():
            self.mediaplayer.pause()
            #self.esmini_space() --> simute press space to pause 
            self.status_label.setText("Status: Paused")
            self.timer.stop()
    
    def stop(self):
        self.mediaplayer.stop()
        self.stop_esmini()
        self.status_label.setText("Status: Stopped")
        self.timer.stop()
        self.close()


    def update_status(self):
        if not self.mediaplayer.is_playing():
            self.stop_esmini()
            self.status_label.setText("Status: Finished")
            self.timer.stop()

    def closeEvent(self, event):
        self.stop()
        event.accept()

def gif_to_mp4(gif_path: str, output_path: str = None, fps: int = 60):
    
    if output_path is None:
        output_path = gif_path.rsplit(".", 1)[0] + ".mp4"

    clip = VideoFileClip(gif_path)
    clip.write_videofile(output_path, codec="libx264", fps=fps)
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gif_to_mp4("../test.gif", "../results/vedio/test.mp4")
    player = VLCPlayer("../results/vedio/test.mp4", "../esmini/bin/esmini.exe", "../results/inD/xosc/25_1400_1700_gen.xosc" )
    player.show()
    sys.exit(app.exec_())
