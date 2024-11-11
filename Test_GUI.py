import sys

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from Main_func import generate_time_series

matplotlib.use('Agg')


class TestTab(QWidget):
    def __init__(self):
        super().__init__()
        self.output_label = QLabel('Prediction :')
        self.output_bar = QTextEdit()
        # self.test_console = QTextEdit()
        self.test_btn = QPushButton('Test the Model')
        self.plot_label = QLabel('Plot the Result :')
        self.display_plot = QLabel()
        self.layout = QGridLayout()  # 网格布局
        self.setLayout(self.layout)  # 设置面板内表单布局
        self.setStyleSheet('background-color:#F0F8FF;')  # 设置tab的面板
        # self.test()
        self.test_tab_gui()

    def test_tab_gui(self):
        # 设置标签
        self.output_label.setFont(QFont('Microsoft YaHei', 15))
        self.output_label.sizeHint()
        self.output_label.setFixedSize(500, 30)
        self.output_label.setStyleSheet("background:transparent")
        self.layout.addWidget(self.output_label)

        # 设置显示预测结果的窗口
        self.output_bar.sizeHint()
        self.output_bar.setFixedSize(500, 60)
        self.output_bar.setPlaceholderText('Predict Result:')
        self.output_bar.setFont(QFont('Microsoft YaHei', 12))
        self.output_bar.setFocusPolicy(QtCore.Qt.NoFocus)  # 设置表格禁止编辑
        self.output_bar.setLineWrapMode(QTextEdit.NoWrap)
        self.layout.addWidget(self.output_bar)

        # 设置图像展示标签
        self.plot_label.setFont(QFont('Microsoft YaHei', 15))
        self.plot_label.sizeHint()
        self.plot_label.setFixedSize(500, 20)
        self.plot_label.setStyleSheet("background:transparent")
        self.layout.addWidget(self.plot_label)

        # 设置图像展示区域
        self.display_plot.setFixedSize(500, 375)
        self.display_plot.setStyleSheet("background:#F0F8FF")
        self.layout.addWidget(self.display_plot)

        # 设置测试的按钮
        self.test_btn.setFont(QFont('Microsoft YaHei', 15))
        self.test_btn.sizeHint()
        self.test_btn.setFixedSize(500, 30)
        self.test_btn.setStyleSheet(
            "QPushButton{background-color:#228B22}"  # 按键背景色
            "QPushButton:hover{color:white}"  # 光标移动到上面后的前景色
            "QPushButton{border-radius:6px}"  # 圆角半径
            "QPushButton:pressed{background-color:rgb(180,180,180);border: None;}"  # 按下时的样式
        )
        self.test_btn.setAutoRepeat(True)  # 设置按钮可重复执行
        self.test_btn.clicked.connect(self.button_click)
        self.layout.addWidget(self.test_btn)  # 按钮布局

    # def test(self):
    #     # 设置控制台
    #     self.test_console.setFixedSize(500, 70)
    #     self.test_console.setFont(QFont('Microsoft YaHei', 12))
    #     self.test_console.setPlaceholderText('Console Log:')
    #     self.test_console.setFocusPolicy(QtCore.Qt.NoFocus)  # 设置表格禁止编辑
    #     # self.test_console.setStyleSheet("padding:1px;border:1px")
    #     # self.test_console.setStyleSheet("background:transparent")
    #     self.layout.addWidget(self.test_console)

    def button_click(self):
        print('Click the Button!')
        series = generate_time_series()
        model = joblib.load('./model/dcfs.pkl')
        output = model.dcfs_test()
        t = np.load('./parameter/period.npy')
        plt.figure()
        plt.plot(series[t-1:3011], color='red')
        plt.plot(output, color='blue')
        plt.savefig('./plot/Figure_predict.png')
        result = []
        for re in output:
            result.append(re)
        print('Complete the All Layers Calculation')
        self.output_bar.setText(str(result))
        img = QPixmap('./plot/Figure_predict.png').scaled(self.display_plot.width(), self.display_plot.height())  # 裁剪
        self.display_plot.setPixmap(img)  # 展示


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TestTab()
    ex.show()
    sys.exit(app.exec_())
