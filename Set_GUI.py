import sys
import numpy as np

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class SetTab(QWidget):

    def __init__(self):
        super().__init__()
        self.fuzzy_label = QLabel('Set Fuzzy Set Number :')  # 模糊集个数标签
        self.fuzzy_edit = QLineEdit()  # 模糊集输入框
        self.layer_label = QLabel('Set Layer Number :')  #
        self.layer_edit = QLineEdit()  #
        self.window_size_label = QLabel('Set Window Size :')  # 滑窗大小标签
        self.window_size_edit = QLineEdit()  # 滑窗大小输入框
        self.slide_step_label = QLabel('Set Sliding Step :')  # 滑窗格大小标签
        self.slide_step_edit = QLineEdit()  # 滑窗大小输入框
        self.s_window = QMessageBox()
        self.f_window = QMessageBox()
        self.set_btn = QPushButton('Set Parameter')
        self.set_data_gui()

    def set_data_gui(self):
        # 表单布局
        layout = QGridLayout()
        label_align = Qt.AlignLeft
        # 设置tab的面板
        self.setStyleSheet('background-color:#F0F8FF;')  # 设置主题

        # 设置fuzzy_set_number
        self.fuzzy_label.setFont(QFont('Microsoft YaHei', 15))
        self.fuzzy_label.sizeHint()
        self.fuzzy_label.setFixedSize(500, 30)
        self.fuzzy_label.setStyleSheet("background:transparent")
        self.fuzzy_label.setAlignment(label_align)
        layout.addWidget(self.fuzzy_label)

        self.fuzzy_edit.sizeHint()
        self.fuzzy_edit.setFixedSize(500, 40)
        self.fuzzy_edit.setPlaceholderText('Input The Fuzzy_set_num:')
        self.fuzzy_edit.setFont(QFont('Microsoft YaHei', 12))
        layout.addWidget(self.fuzzy_edit)

        # 设置layer_number
        self.layer_label.setFont(QFont('Microsoft YaHei', 15))
        self.layer_label.sizeHint()
        self.layer_label.setFixedSize(500, 30)
        self.layer_label.setStyleSheet("background:transparent")
        self.layer_label.setAlignment(label_align)
        layout.addWidget(self.layer_label)

        self.layer_edit.sizeHint()
        self.layer_edit.setFixedSize(500, 40)
        self.layer_edit.setPlaceholderText('Input The Layer_num:')
        self.layer_edit.setFont(QFont('Microsoft YaHei', 12))
        layout.addWidget(self.layer_edit)

        # 设置window_size
        self.window_size_label.setFont(QFont('Microsoft YaHei', 15))
        self.window_size_label.sizeHint()
        self.window_size_label.setFixedSize(500, 30)
        self.window_size_label.setStyleSheet("background:transparent")
        self.window_size_label.setAlignment(label_align)
        layout.addWidget(self.window_size_label)

        self.window_size_edit.sizeHint()
        self.window_size_edit.setFixedSize(500, 40)
        self.window_size_edit.setPlaceholderText('Input The Window_size:')
        self.window_size_edit.setFont(QFont('Microsoft YaHei', 12))
        layout.addWidget(self.window_size_edit)

        # 设置sliding_step
        self.slide_step_label.setFont(QFont('Microsoft YaHei', 15))
        self.slide_step_label.sizeHint()
        self.slide_step_label.setFixedSize(500, 30)
        self.slide_step_label.setStyleSheet("background:transparent")
        self.slide_step_label.setAlignment(label_align)
        layout.addWidget(self.slide_step_label)

        self.slide_step_edit.sizeHint()
        self.slide_step_edit.setFixedSize(500, 40)
        self.slide_step_edit.setPlaceholderText('Input The Sliding_step:')
        self.slide_step_edit.setFont(QFont('Microsoft YaHei', 12))
        layout.addWidget(self.slide_step_edit)

        # 设置按钮
        self.set_btn.setFont(QFont('Microsoft YaHei', 12))
        self.set_btn.sizeHint()
        self.set_btn.setFixedSize(500, 30)
        self.set_btn.setStyleSheet(
            "QPushButton{background-color:#228B22}"  # 按键背景色
            "QPushButton:hover{color:white}"  # 光标移动到上面后的前景色
            "QPushButton{border-radius:6px}"  # 圆角半径
            "QPushButton:pressed{background-color:rgb(180,180,180);border: None;}"  # 按下时的样式
        )
        layout.addWidget(self.set_btn)
        # 按钮绑定
        self.set_btn.clicked.connect(self.button_click)
        self.set_btn.setAutoRepeat(True)  # 可反复执行
        self.setLayout(layout)  # 外层设置

    def failure(self):
        self.f_window.about(self, 'Message', 'Set Failed')  # pass

    def success(self):
        self.s_window.about(self, 'Message', 'Set Success')  # pass

    def button_click(self):
        fuzzy_set_number = int(self.fuzzy_edit.text())
        layer_number = int(self.layer_edit.text())
        window_size = int(self.window_size_edit.text())
        sliding_step = int(self.slide_step_edit.text())
        print(fuzzy_set_number)
        print(layer_number)
        print(window_size)
        print(sliding_step)
        if fuzzy_set_number == '' or layer_number == '' or \
                layer_number == '' or sliding_step == '':
            self.failure()
        else:
            self.success()
            parameter = [fuzzy_set_number, layer_number, window_size, sliding_step]
            print(parameter)
            np.save('./parameter/parameter.npy', parameter)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SetTab()
    ex.show()
    sys.exit(app.exec_())
