import numpy as np
from dcfs.wm import WM


class DCFS(object):
    # 初始化算法相关参数
    def __init__(self,
                 window_size,
                 sliding_step,
                 layer_number,
                 fuzzy_set_number,
                 train_data,
                 test_data):
        self.window_size = window_size
        self.sliding_step = sliding_step
        self.layer_number = layer_number
        self.fuzzy_set_number = fuzzy_set_number
        self.train_data = train_data
        self.test_data = test_data
        self.wm_fuzzy_system_layer = []

    # 训练深度卷积模糊系统
    def dcfs_train(self):
        input_x = self.train_data[:, 0:(np.size(self.train_data, 1) - 1)]  # 返回train_data的前11列将其设为input_x(2000行11列)
        input_y = self.train_data[:, (np.size(self.train_data, 1) - 1)]  # 返回train_data的最后1列并将其转置，再设为input_y(一行，2000个数)
        for l in range(0, self.layer_number):  # layer_number = 5 , 遍历这5层
            feature_number = np.size(input_x, 1)  # 计算input_x的列数，即为特征数，11个
            fuzzy_system_number = feature_number - self.window_size + self.sliding_step  # 计算fs分类器的个数 11-3+1=9
            convolution_x = self.generate_convolution_data(input_x,
                                                           fuzzy_system_number)  # 卷积操作，将input_x进行卷积处理，得到分别给每个fs分类器的输入数据，为9个2000行3列的矩阵
            wm_fuzzy_system = []  # 准备空列表
            next_input_x = np.zeros((np.size(input_x, 0), fuzzy_system_number))  # 生成2000行9列的由0.组成的矩阵，用来接收fs分类器的输出结果
            for f in range(0, fuzzy_system_number):  # 依次遍历每个fs分类器
                wfs = WM(convolution_x[f], input_y, self.fuzzy_set_number)  # 用WM类创建对象wfs
                wfs.wm_train()  # 调用WM的训练函数，对象是当前fs分类器对应的卷积后的数据convolution_x[f]，返回被数据覆盖的cell的参数c
                wfs.extrapolate_cell()  # 将由样本生成的规则扩充至所有cell，即计算所有cells的参数c
                if l < (self.layer_number - 1):  # 当前层数处于前4层时
                    wfs.wm_test()  # 进行测试，返回当前层当前fs分类器的输出值
                    next_input_x[:, f] = wfs.output[:, 0]  # 将output(2000行1列)赋给next_input_x(2000行9列)的对应列
                    # 以第一层为例，每个fs遍历完后，next_input_x中的每一列都对应着每个fs的输出
                wm_fuzzy_system.append(wfs)  # 将当前对象wfs存入，每个fs对应一个wfs
            self.wm_fuzzy_system_layer.append(wm_fuzzy_system)  # 将一层的wfs放入wm_fuzzy_system_layer
            if l < (self.layer_number - 1):
                input_x = next_input_x  # 将该层的输出作为下一层的输入
            print("Complete the %d layer training！" % l)
        return input_x

    def dcfs_test(self):
        # 将训练数据和测试数据拼接在一起
        input_x_test = np.array(self.train_data[:, 0:(np.size(self.train_data, 1) - 1)])  # 选取train_data的0-11列
        input_x_test = np.vstack((input_x_test, self.test_data[:, 0:(
                    np.size(self.train_data, 1) - 1)]))  # 将train_data的0-11列和test_data的0-11列上下拼接，3000行1列为总的测试数据
        for l in range(0, self.layer_number):  # 按层遍历wm_fuzzy_system_layer
            fuzzy_system_number = len(self.wm_fuzzy_system_layer[l])  # 获取当前层(第一层)中的fs分类器的个数，第一层为9个
            convolution_x_test = self.generate_convolution_data(input_x_test,
                                                                fuzzy_system_number)  # 将测试数据按照fs分类器个数进行卷积操作
            next_input_x_test = np.zeros(
                (np.size(input_x_test, 0), fuzzy_system_number))  # 生成3000行9列的矩阵用于存放该层的输出，且作为下一层的输入
            for f in range(0, len(self.wm_fuzzy_system_layer[l])):  # 依次遍历该层的每个fs分类器
                self.wm_fuzzy_system_layer[l][f].input_x = convolution_x_test[f]  # 将卷积后的测试数据传入给对应的fs分类器
                self.wm_fuzzy_system_layer[l][f].sample_number = np.size(self.wm_fuzzy_system_layer[l][f].input_x,
                                                                         0)  # 将样本个数作为参数(3000个)也一并传入
                self.wm_fuzzy_system_layer[l][f].output = np.zeros(
                    (self.wm_fuzzy_system_layer[l][f].sample_number, 1))  # 为其准备矩阵用于存放输出，3000行1列
                self.wm_fuzzy_system_layer[l][f].wm_test()  # 对该层存储的对象调用测试功能，得到输出
                if l < (self.layer_number - 1):  # 前4层时
                    next_input_x_test[:, f] = self.wm_fuzzy_system_layer[l][f].output[:, 0]  # 将该层的输出复制给next_input_x_test
                else:  # 最后一层时
                    return self.wm_fuzzy_system_layer[l][f].output[:, 0]  # 返回该层(第5层)的输出
            input_x_test = next_input_x_test  # 将存储该层输出的矩阵next_input_x_test作为下一层的输入
            print("Complete the %d layer calculation！" % l)

    # 通过卷积形成输入数据
    def generate_convolution_data(self, input_x, fuzzy_system_number):  # 卷积数据生成
        sample_number = np.size(input_x, 0)  # 待卷积数据的样本数量，计算input_x的行数，2000个
        convolution_data = np.zeros((fuzzy_system_number, sample_number,
                                     self.window_size))  # 生成一个三维矩阵，可理解为9个由0.组成的2000行3列的矩阵，分别给9个fs分类器，用于存放其卷积数据
        for i in range(0, fuzzy_system_number):  # 分别遍历每个fs分类器下的convolution_data
            convolution_data[i] = input_x[:, (i * self.sliding_step):(
                        i + self.window_size)]  # 对2000行11列的训练数据进行滑窗选取，滑窗步长为1，窗口大小为3，将其存储为convolution_data。
            # convolution_data[0]为训练数据的前3列，convolution_data[1]为训练数据的2,3,4列
        return convolution_data  # 返回给每个fs分类器的输入数据
