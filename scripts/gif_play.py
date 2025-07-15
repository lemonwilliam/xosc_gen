# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 12:51:54 2025

@author: dianel
"""

import sys
import time
import platform
import subprocess

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QMovie, QPainter

# Windows 模組
if platform.system() == "Windows":
    import win32gui
    import win32con
    import win32api
    import pygetwindow as gw

class AspectRatioGifLabel(QLabel):
    def __init__(self, gif_path, parent=None):
        super().__init__(parent)
        self.movie = QMovie(gif_path)
        self.setMovie(self.movie)
        self.movie.frameChanged.connect(self.update) 

    def paintEvent(self, event):
        painter = QPainter(self)
        frame = self.movie.currentPixmap()
        if not frame.isNull():
            scaled_pixmap = frame.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (self.width() - scaled_pixmap.width()) // 2
            y = (self.height() - scaled_pixmap.height()) // 2
            painter.drawPixmap(x, y, scaled_pixmap)
        else:
            super().paintEvent(event)

class GifPlayer(QWidget):
    def __init__(self, gif_path, esmini_path, osc_path):
        super().__init__()
        self.setWindowTitle("GIF Controller")
        self.resize(950, 1080)

        # 讀取 GIF 原始大小，計算等比例縮放後的高度（寬度固定950）
        temp_movie = QMovie(gif_path)
        temp_movie.jumpToFrame(0)
        original_size = temp_movie.currentPixmap().size()
        fixed_width = 950
        fixed_height = int(original_size.height() * fixed_width / original_size.width())

        # GIF 顯示區，使用自訂 QLabel
        self.gif_label = AspectRatioGifLabel(gif_path, self)
        self.gif_label.setFixedSize(fixed_width, fixed_height)

        # 控制元件容器
        self.controls_container = QWidget(self)
        self.controls_container.setFixedSize(400, 50)
        ctrl_x = (fixed_width - 400) // 2
        ctrl_y = fixed_height - 50
        self.controls_container.move(ctrl_x, ctrl_y)

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
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 20, 0, 20) 
        main_layout.setSpacing(20)
        main_layout.addStretch()
        main_layout.addWidget(self.gif_label, alignment=Qt.AlignCenter)
        main_layout.addWidget(self.controls_container, alignment=Qt.AlignCenter)
        main_layout.addStretch() 
        
        self.setLayout(main_layout)

        self.esmini_path = esmini_path
        self.osc_path = osc_path
        self.esmini_proc = None

        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        self.stop_btn.clicked.connect(self.stop)

        # 自動停止檢查
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.check_finished)
            
    def wait_for_esmini_window(self, process_name="esmini", timeout=10):
   
        os_name = platform.system().lower()
        start_time = time.time()
    
        if os_name == "windows":
            while time.time() - start_time < timeout:
                windows = gw.getWindowsWithTitle("")
                for win in windows:
                    print(time.time() - start_time, win)
                    if "esmini --osc" in win.title.lower():
                        return True
    
        elif os_name == "linux":
            while time.time() - start_time < timeout:
                try:
                    result = subprocess.check_output(["wmctrl", "-l"]).decode("utf-8")
                    if "esmini --osc" in result.lower():
                        return True
                except Exception as e:
                    print(f"[!] wmctrl error: {e}")
    
        print("[X] esmini windows not found ")
        return False
    
    def start_esmini(self):
        if not self.esmini_proc or self.esmini_proc.poll() is not None:
            self.esmini_proc = subprocess.Popen([
                self.esmini_path,
                "--window", "950", "267", "950", "534",
                "--osc", self.osc_path,
                "--camera_mode", "top",
                "--trail_mode", "0", "-q"
            ])
        self.wait_for_esmini_window()
    
    def stop_esmini(self):
            if self.esmini_proc:
                self.esmini_proc.terminate()
                self.esmini_proc = None
    '''
    def esmini_space(self):
        if not self.esmini_proc:
            return

        if platform.system() == "Windows":
            def enum_handler(hwnd, param):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if "esmini" in title.lower():
                        param.append(hwnd)

            hwnds = []
            win32gui.EnumWindows(enum_handler, hwnds)

            if hwnds:
                hwnd = hwnds[0]
                win32gui.SetForegroundWindow(hwnd)
                time.sleep(0.1)
                win32api.PostMessage(hwnd, win32con.WM_KEYDOWN, win32con.VK_SPACE, 0)
                win32api.PostMessage(hwnd, win32con.WM_KEYUP, win32con.VK_SPACE, 0)

        elif platform.system() == "Linux":
            try:
                win_id = subprocess.check_output(["xdotool", "search", "--name", "esmini"]).decode().strip()
                subprocess.run(["xdotool", "windowactivate", win_id])
                time.sleep(0.1)
                subprocess.run(["xdotool", "key", "--window", win_id, "space"])
            except subprocess.CalledProcessError:
                print("[Linux] 找不到 esmini 視窗（需要安裝 xdotool）")
    '''
    def play(self):
        gif = self.gif_label.movie
        if gif.state() == QMovie.Paused:
            gif.setPaused(False)
#            self.esmini_space()  
            self.status_label.setText("Status: Playing")
        elif gif.state() != QMovie.Running:
            self.start_esmini()
            gif.start()
            self.status_label.setText("Status: Playing")
            self.timer.start()

    def pause(self):
        gif = self.gif_label.movie
        if gif.state() == QMovie.Running:
            gif.setPaused(True)
#            self.esmini_space()  
            self.status_label.setText("Status: Paused")

    def stop(self):
        self.gif_label.movie.stop()
        self.stop_esmini()
        self.status_label.setText("Status: Stopped")
        self.timer.stop()
        self.close()

    def check_finished(self):
        gif = self.gif_label.movie
        if gif.currentFrameNumber() == gif.frameCount() - 1:
            gif.stop()
            self.stop_esmini()
            self.status_label.setText("Status: Finished")
            self.timer.stop()

    def closeEvent(self, event):
        self.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = GifPlayer(
        "./test1.gif",
        "./esmini/bin/esmini",
        "./esmini/resources/xosc/ind/07_3100_3500_gen.xosc"
    )
    player.show()
    sys.exit(app.exec_())
