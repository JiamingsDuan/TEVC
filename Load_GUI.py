import sys
import matplotlib.pyplot as plt
import matplotlib
from PyQt5 import QtCore

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from Main_func import generate_time_series


matplotlib.use('Agg')


class LoadTab(QWidget):
    def __init__(self):
        super().__init__()
        self.palette = QPalette()
        self.select_label = QLabel('Select Dataset :')
        self.file_btn = QPushButton('Select the Dataset')
        self.path_label = QLabel('Dataset Path :')
        self.display_bar = QLineEdit()
        self.edit_label = QLabel('Dataset Content :')
        self.display_txt = QTextEdit()
        self.plot_label = QLabel('Plot the Dataset ：')
        self.display_plot = QLabel()
        self.load_data_gui()

    def load_data_gui(self):
        # 布局管理
        grid_layout = QGridLayout()  # 网格布局

        # 设置tab的面板
        self.setLayout(grid_layout)  # 设置面板内网格布局
        self.setStyleSheet('background-color:#F0F8FF;')  # 设置主题

        # 设置标签
        self.select_label.setFont(QFont('Microsoft YaHei', 15))
        self.select_label.sizeHint()
        self.select_label.setFixedSize(200, 20)
        self.select_label.setStyleSheet("background:transparent")
        grid_layout.addWidget(self.select_label)

        # 设置选择数据集的按钮
        self.file_btn.setFont(QFont('Microsoft YaHei', 15))
        self.file_btn.sizeHint()
        self.file_btn.setFixedSize(500, 30)
        self.file_btn.setStyleSheet(
            "QPushButton{background-color:#228B22}"  # 按键背景色
            "QPushButton:hover{color:white}"  # 光标移动到上面后的前景色
            "QPushButton{border-radius:6px}"  # 圆角半径
            "QPushButton:pressed{background-color:rgb(180,180,180);border: None;}"  # 按下时的样式
        )
        # form_layout.addRow(select_label, file_btn)
        self.file_btn.setAutoRepeat(True)  # 设置按钮可重复执行
        grid_layout.addWidget(self.file_btn)  # 按钮布局

        # 设置标签
        self.path_label.setFont(QFont('Microsoft YaHei', 15))
        self.path_label.sizeHint()
        self.path_label.setFixedSize(200, 20)
        self.path_label.setFocusPolicy(QtCore.Qt.NoFocus)
        self.path_label.setStyleSheet("background:transparent")
        grid_layout.addWidget(self.path_label)

        # 设置显示数据集路径的窗口
        self.display_bar.sizeHint()
        self.display_bar.setFixedSize(500, 30)
        self.display_bar.setPlaceholderText('Path:')
        self.display_bar.setFont(QFont('Microsoft YaHei', 12))
        # display_bar.setText(button_click)
        # form_layout.addRow(path_label, display_bar)
        grid_layout.addWidget(self.display_bar)

        # 设置标签
        self.edit_label.setFont(QFont('Microsoft YaHei', 15))
        self.edit_label.sizeHint()
        self.edit_label.setFixedSize(200, 20)
        self.edit_label.setStyleSheet("background:transparent")
        grid_layout.addWidget(self.edit_label)

        # 设置显示数据集内容的窗口
        self.display_txt.sizeHint()
        self.display_txt.setFixedSize(500, 50)
        self.display_txt.setPlaceholderText('Dataset:')
        self.display_txt.setFont(QFont('Microsoft YaHei', 12))
        self.display_txt.setLineWrapMode(QTextEdit.NoWrap)
        self.display_txt.setFocusPolicy(QtCore.Qt.NoFocus)
        grid_layout.addWidget(self.display_txt)

        # 为按钮绑定事件
        self.file_btn.clicked.connect(self.button_click)
        # self.file_btn.clicked.connect(QCoreApplication.quit)

        # 设置图像展示标签
        self.plot_label.setFont(QFont('Microsoft YaHei', 15))
        self.plot_label.sizeHint()
        self.plot_label.setFixedSize(200, 20)
        self.plot_label.setStyleSheet("background:transparent")
        grid_layout.addWidget(self.plot_label)

        # 设置图像展示区域
        self.display_plot.setFixedSize(500, 375)
        self.display_plot.setStyleSheet("background:#F0F8FF")
        grid_layout.addWidget(self.display_plot)

    def button_click(self):

        # 设置选择的文件夹起始位置
        directory = QFileDialog.getOpenFileName(None, 'Select Dataset', './series/', '*.txt')
        while directory[0] == '':
            break
        else:
            print(directory)
            self.display_bar.setFont(QFont('Microsoft YaHei', 10))
            self.display_bar.setText(directory[0])
            self.display_txt.setFont(QFont('Microsoft YaHei', 10))
            line = open(directory[0], 'r').readline().strip()  # 读取时间序列
            self.display_txt.setText(line)  # 写入文本框
            series = generate_time_series()  # 字符型转浮点型
            plt.plot(series, color='blue')  # 画图
            plt.title('Original Time Series')  # 命名
            plt.savefig('./plot/Figure_dataset.png')  # 保存
            img = QPixmap('./plot/Figure_dataset.png').scaled(self.display_plot.width(),
                                                              self.display_plot.height()
                                                              )  # 裁剪
            self.display_plot.setPixmap(img)  # 展示


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = LoadTab()
    ex.show()
    sys.exit(app.exec_())
