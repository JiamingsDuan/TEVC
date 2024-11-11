import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

# 导入子模块
from Load_GUI import LoadTab
from Set_GUI import SetTab
from Deal_GUI import DealTab
from Train_GUI import TrainTab
from Test_GUI import TestTab


# 定义控制台输出流
class ConsoleStream(QObject):
    train_text = pyqtSignal(str)

    def write(self, text):
        self.train_text.emit(str(text))
        QApplication.processEvents()

    def flush(self):
        pass


class Application(QWidget):

    def __init__(self):
        super(Application, self).__init__()

        # 【1】全局初始化
        # 设置窗口宽高都为屏幕的0.5倍
        self.init_gui(0.5)
        self.tab = QTabWidget()
        self.tab.setFixedWidth(600)
        self.tab.setStyleSheet('background:#E0FFFF')
        self.tab.setFont(QFont('Microsoft YaHei', 12))
        layout = QHBoxLayout()
        layout.addWidget(self.tab)
        self.setLayout(layout)

        # 【2】导入选项卡控件
        self.tab.load_tab = LoadTab()
        self.tab.set_tab = SetTab()
        self.tab.deal_tab = DealTab()
        self.tab.train_tab = TrainTab()
        self.tab.test_tab = TestTab()

        # 【3】添加顶层窗口
        self.tab.addTab(self.tab.load_tab, 'Dataset')
        self.tab.addTab(self.tab.deal_tab, 'Prepossessing')
        self.tab.addTab(self.tab.set_tab, 'Setting')
        self.tab.addTab(self.tab.train_tab, 'Training')
        self.tab.addTab(self.tab.test_tab, 'Prediction')

        # 【4】设置控制台
        sys.stdout = ConsoleStream(train_text=self.update_text)
        sys.stderr = ConsoleStream(train_text=self.update_text)
        self.console = QTextEdit()
        self.console.setFixedWidth(300)
        self.console.setPlaceholderText('Python Console:')
        self.console.setFont(QFont('Microsoft YaHei', 12))
        self.console.setStyleSheet("padding:0px;border:0px")
        self.console.setFocusPolicy(QtCore.Qt.NoFocus)
        layout.addWidget(self.console)

    def init_gui(self, rates):
        # 设置窗口大小为屏幕自适应
        desktop = QtWidgets.QApplication.desktop()  # 获取电脑分辨率right
        screen_width = desktop.width() * rates  # 窗口宽
        screen_height = desktop.height() * rates * 1.3  # 窗口高
        self.resize(int(screen_width), int(screen_height))
        self.setWindowTitle('Application')  # 设置标题
        self.setWindowIcon(QIcon('Icon/monkey.jpg'))  # 设置logo
        self.setFont(QFont('Microsoft YaHei', 12))  # 设置全局字体
        self.show()

    # 控制窗口显示在屏幕中心
    def center(self):
        window = self.frameGeometry()  # 获得窗口
        window_center = QDesktopWidget().availableGeometry().center()
        window.moveCenter(window_center)  # 显示到屏幕中心
        self.move(window.topLeft())

    def update_text(self, text):
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()

    def shutdown_event(self, event):
        sys.stdout = sys.__stdout__
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Application()
    ex.show()
    sys.exit(app.exec_())
