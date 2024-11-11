import sys
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from Main_func import generate_time_series
from Main_func import generate_data_table

Series = generate_time_series()

# print(len(Series))


class DealTab(QWidget):

    def __init__(self):
        super().__init__()
        self.episode_label = QLabel('Period:')
        self.episode = QLineEdit()
        self.file_btn = QPushButton('Generate the Dataset')
        self.save_btn = QPushButton('Save the Dataset')
        self.display_bar = QLineEdit()
        self.dataset_frame = QTableWidget()  # 设置20行8列
        self.s_window = QMessageBox()
        self.deal_data_gui()

    def deal_data_gui(self):
        layout = QGridLayout()
        # layout = QGridLayout()  # 布局
        # 设置tab的面板
        self.setLayout(layout)  # 设置面板内铅直布局
        self.setStyleSheet('background-color:#F0F8FF;')  # 背景
        # 设置周期标签
        self.episode_label.setFont(QFont('Microsoft YaHei', 15))
        self.episode_label.sizeHint()
        self.episode_label.setFixedSize(500, 30)
        self.episode_label.setStyleSheet("background:transparent")
        layout.addWidget(self.episode_label)
        # 设置周期输入框
        self.episode.sizeHint()
        self.episode.setFixedSize(500, 30)
        self.episode.setPlaceholderText('Input The Period:')
        self.episode.setFont(QFont('Microsoft YaHei', 12))
        layout.addWidget(self.episode)
        # 设置表格
        self.dataset_frame.resizeColumnsToContents()  # 内容列匹配
        self.dataset_frame.resizeRowsToContents()  # 内容行匹配
        self.dataset_frame.setFixedHeight(450)
        self.dataset_frame.setFixedWidth(500)
        self.dataset_frame.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 设置表格禁止编辑
        layout.addWidget(self.dataset_frame)  # 水平布局顶对齐
        # 设置按钮
        self.file_btn.setFont(QFont('Microsoft YaHei', 15))
        self.file_btn.sizeHint()
        self.file_btn.setFixedSize(500, 30)
        self.file_btn.setStyleSheet(
            "QPushButton{background-color:#228B22}"  # 按键背景色
            "QPushButton:hover{color:white}"  # 光标移动到上面后的前景色
            "QPushButton{border-radius:6px}"  # 圆角半径
            "QPushButton:pressed{background-color:rgb(180,180,180);border: None;}"  # 按下时的样式
        )
        # 按钮绑定
        self.file_btn.clicked.connect(self.button_click)
        self.file_btn.setAutoRepeat(True)  # 可反复执行
        layout.addWidget(self.file_btn)  # 按钮布局

        # 设置保存数据集路径的窗口
        self.display_bar.sizeHint()
        self.display_bar.setFixedSize(500, 40)
        self.display_bar.setPlaceholderText('Save as dataset.npy')
        self.display_bar.setFont(QFont('Microsoft YaHei', 12))
        self.display_bar.setFocusPolicy(QtCore.Qt.NoFocus)
        # display_bar.setText(button_click)
        # form_layout.addRow(path_label, display_bar)
        layout.addWidget(self.display_bar)

        # 设置保存按钮
        self.save_btn.setFont(QFont('Microsoft YaHei', 15))
        self.save_btn.sizeHint()
        self.save_btn.setFixedSize(500, 30)
        self.save_btn.setStyleSheet(
            "QPushButton{background-color:#228B22}"  # 按键背景色
            "QPushButton:hover{color:white}"  # 光标移动到上面后的前景色
            "QPushButton{border-radius:6px}"  # 圆角半径
            "QPushButton:pressed{background-color:rgb(180,180,180);border: None;}"  # 按下时的样式
        )
        # 保存按钮绑定
        self.save_btn.clicked.connect(self.save_click)
        self.save_btn.setAutoRepeat(True)  # 可反复执行
        layout.addWidget(self.save_btn)  # 按钮布局

    def button_click(self):
        t = self.episode.text()  # 获取周期输入值
        np.save('./parameter/period.npy', int(t))
        table_row = 3000  # 样本数量
        table_col = int(t)  # 周期
        self.dataset_frame.setRowCount(3000)  # 设置表行数
        self.dataset_frame.setColumnCount(int(t))  # 设置表列数
        self.dataset_frame.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 水平方向自适应
        self.dataset_frame.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 垂直方向自适应
        array_data = generate_data_table(int(t), Series)  # 导入数据
        # 赋值
        for row in range(0, table_row):
            for col in range(0, table_col):
                self.dataset_frame.setItem(row, col, QTableWidgetItem(str(array_data[row][col])))

    def success(self):
        self.s_window.about(self, 'Message', 'Set Success')  # pass

    def save_click(self):
        # 设置选择的文件夹起始位置
        directory = QFileDialog.getSaveFileName(None, 'Save Dataset', './dataset/', '*.npy')
        while directory[0] == '':
            break
        else:
            print(directory)
            self.display_bar.setFont(QFont('Microsoft YaHei', 10))
            self.display_bar.setText(directory[0])
            t = self.episode.text()  # 获取周期输入值
            array_data = generate_data_table(int(t), Series)  # 导入数据
            np.save(directory[0], array_data)
            self.success()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DealTab()
    ex.show()
    sys.exit(app.exec_())
