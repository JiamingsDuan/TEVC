import os
import sys

import joblib
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from dcfs.dcfs import DCFS

# 【0】场景的宏
width = 300
height = 300
# 【1】加载数据集
dataset = np.load('./dataset/dataset.npy')
# 【2】加载参数
parameter = np.load('./parameter/parameter.npy')
# 【3】参数设置
fuzzy_set_number = parameter[0]
layer_number = parameter[1]
window_size = parameter[2]
sliding_step = parameter[3]
# 【4】训练集
train_data = np.array(dataset[0:2000, :])  # 前2000行12列
# 【5】测试集
test_data = np.array(dataset[2000:3000, :])  # 后1000行12列
# 【6】Model
model = DCFS(window_size, sliding_step, layer_number, fuzzy_set_number, train_data, test_data)


class TrainTab(QWidget):

    def __init__(self):
        super().__init__()
        self.running_bar = QLabel()
        self.train_btn = QPushButton('Train the Model')  # 按钮
        self.train_data_gui()

    def train_data_gui(self):
        # 设置tab的面板
        layout = QGridLayout()  # 网格布局
        self.setLayout(layout)  # 设置面板内表单布局
        self.setStyleSheet('background:transparent')

        # 设置状态框
        self.running_bar.setFixedSize(500, 375)
        self.running_bar.setStyleSheet('background:transparent')
        self.running_bar.setStyleSheet('padding:0px;border:0px')
        layout.addWidget(self.running_bar)

        # 设置按钮
        self.train_btn.setFixedSize(500, 30)
        self.train_btn.setFont(QFont('Microsoft YaHei', 15))
        self.train_btn.setStyleSheet("QPushButton{background-color:#228B22}"  # 按键背景色
                                     "QPushButton:hover{color:white}"  # 光标移动到上面后的前景色
                                     "QPushButton{border-radius:6px}"  # 圆角半径
                                     "QPushButton:pressed{background-color:#9ACD32;border: None;}"  # 按下时的样式
                                     )
        self.train_btn.clicked.connect(self.button_click)
        layout.addWidget(self.train_btn)

    def button_click(self):
        print('Clicked the Button!')
        # self.running_bar.setPixmap(QPixmap('./Icon/train.png'))
        output = model.dcfs_train()  # 模型训练
        print(output)
        self.running_bar.setPixmap(QPixmap(''))  # 移除label上的图片
        joblib.dump(model, './model/dcfs.pkl')  # 保存模型
        plt.plot(output, color='blue')
        plt.savefig('./plot/Figure_train.png')
        img = QPixmap('./plot/Figure_train.png').scaled(self.running_bar.width(),
                                                        self.running_bar.height())  # 裁剪

        self.running_bar.setPixmap(img)  # 展示


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TrainTab()
    ex.show()
    sys.exit(app.exec_())
