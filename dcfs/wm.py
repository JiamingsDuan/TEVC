import numpy as np

class WM(object):
    
    #初始化函数
    def __init__(self, input_x, input_y, fuzzy_set_number): # 因为是训练，所以要把y也传入
        #WM算法所需的参数
        self.input_x = input_x #卷积后的输入数据，2000行3列
        self.input_y = input_y #输出数据，2000个数
        self.sample_number = np.size(input_x, 0) #样本数量，2000
        self.feature_number = np.size(input_x, 1) #特征数量，3
        self.fuzzy_set_number = fuzzy_set_number #每个特征上定义模糊集的数量，20个
        self.cell_number = self.fuzzy_set_number**self.feature_number # 每个样本3个特征，每个特征上有20个模糊集，张成20**3个cells
        self.fuzzy_set_range = np.zeros((self.feature_number, 2)) # 3行2列的由0.组成的矩阵，用于存储每个特征对应的模糊集的范围
        self.then_center = np.zeros((1, self.cell_number)) # 1行20**3列 ，每个样本3个特征，每个特征上有20个模糊集，张成20**3个cells
        self.then_y = np.zeros((1, self.cell_number)) # 1行20**3列
        self.output = np.zeros((self.sample_number, 1)) # 输出，2000行1列
        
    #WM方法的训练阶段
    def wm_train(self):   
        #找到每个特征上的最大值和最小值，每一位上的baseCount
        for j in range(0, self.feature_number): # 遍历每个特征的所有样本，寻找其最小值和最大值，用于确定模糊集的endpoints
            self.fuzzy_set_range[j, 0] = np.min(self.input_x[:, j]) #选取特征j，计算其上的最小值 
            self.fuzzy_set_range[j, 1] = np.max(self.input_x[:, j]) #选取特征j，计算其上的最大值           
            # 第1行为第一个特征的最小值和最大值，......

        #从训练样本数据生成规则
        for i in range(0, self.sample_number):  #遍历2000个样本
            
            active_index_grade = np.zeros((self.feature_number, 2)) #用于存放该样本在每个特征上激活的模糊集index及其对应的隶属度
            for j in range(0, self.feature_number):  #遍历特征
                for k in range(0, self.fuzzy_set_number):  #遍历该特征上的模糊集，20个，目的是求该值在每个模糊集上的隶属度，并取隶属度的最大值和所处的模糊集
                    membership_degree = self.membership(j, k, self.input_x[i, j]) # 计算隶属度，参数分别为当前特征的序号，当前模糊集的序号，输入数据中的第i个样本的第j个特征的值
                    if membership_degree > active_index_grade[j, 1]: # 如果当前求得的隶属度大于active_index_grade中存储的隶属度(初始值为0)
                        active_index_grade[j, 0] = k # 记录当前模糊集的序号
                        active_index_grade[j, 1] = membership_degree # 记录当前隶属度
            cell_index_grade = self.get_cell_index_grade(active_index_grade) # 计算当前样本的cell_index_grade，通过计算其每个特征所处的模糊集和隶属度
            cell_index = int(cell_index_grade[0]) # 将所处cell转化成整型
            cell_grade = cell_index_grade[1] # 取隶属度相乘后的值
            self.then_y[0, cell_index] = self.then_y[0, cell_index] + cell_grade # 在cell空间中找到当前cell的位置，记录其cell_grade，step2.4计算weight
            self.then_center[0, cell_index] = self.then_center[0, cell_index] + self.input_y[i] * cell_grade; # 找到当前cell的位置，记录input_y[i] * cell_grade，step2.4计算weight-output
        
        #计算THEN部分的中心值
        for c in range(0, self.cell_number):
            if self.then_y[0, c] != 0:  # weight不为0
                self.then_center[0, c] = self.then_center[0, c] / self.then_y[0, c] # 计算当被数据覆盖的位置的参数c，即为C(0)

        
    #WM方法的测试阶段，实为得到输出，对应论文中的公式7
    def wm_test(self):
        active_fuzzy_index = np.zeros((self.feature_number, 2)) # 构建3行2列的用于存放该样本在每个特征上激活的模糊集index，因为最多能激活两个模糊集
        active_grade = np.zeros((self.feature_number, 2)) # 构建3行2列的用于存放该样本在每个特征上激活的模糊集的隶属度
        active_combination = self.active_combination() # 激活组合矩阵为8行3列，每行内容为行号的二进制表示

        for i in range(0, self.sample_number): # 遍历每个样本，2000个
            for j in range(0, self.feature_number): # 遍历该样本的每个特征，3个
                flag = 0
                for k in range(0, self.fuzzy_set_number): # 遍历该特征上的模糊集，20个
                    membership_degree = self.membership(j, k, self.input_x[i, j]) # 计算该样本的该特征在该模糊集上的隶属度
                    if membership_degree > 0: # 如果该模糊集上隶属度大于0
                        active_fuzzy_index[j, flag] = k # 记录当前模糊集index于active_fuzzy_index第j行第flag(初始值为0)个位置
                        active_grade[j, flag] = membership_degree # 记录当前模糊集隶属度于active_grade第j行第flag(初始值为0)个位置
                        flag = flag + 1 # flag的最大值为1，即最多只能同时属于两个模糊集
            # 将该样本的三个特征全部遍历完后，矩阵active_fuzzy_index记录着每个特征所处的模糊集，active_combination记录着对应的隶属度
            active_index = np.zeros((self.feature_number, 2)) # 3行2列，用来存放最终确定的模糊集和对应隶属度
            for j in range(0, self.feature_number): # 按行遍历active_fuzzy_index
                active_index[j, 0] = active_fuzzy_index[j, 0] # 将模糊集序号记录进active_index第j行第1个位置
                active_index[j, 1] = active_index[j, 0] + 1 # 将active_index第j行第2个位置记录为模糊集的相邻序号
                if active_index[j, 0] == (self.fuzzy_set_number - 1): # 如果第一个模糊集序号已经为19(最后一个模糊集)
                    active_index[j, 1] = active_index[j, 0] # 则第二个位置也是19
            
            fenzi = 0 # 分子初始值
            fenmu = 0 # 分母初始值
            for c in range(0, np.size(active_combination, 0)): # 遍历三个特征相乘的所有可能，本为3**20次，但由于模糊集的定义，每个特征最多只能属于2个模糊集，故实际可能性为3**2
                cell_index = 0 # 初始值
                grade = 1 # 初始值
                # 执行输出公式
                for j in range(0, self.feature_number): # 3个特征，每个样本由三个特征输出一个结果
                    grade = grade * active_grade[j, int(active_combination[c, j])]
                    cell_index = cell_index + active_index[j, int(active_combination[c, j])] * (self.fuzzy_set_number ** (self.feature_number - j - 1))
                fenzi = fenzi + self.then_center[0, int(cell_index)] * grade
                fenmu = fenmu + grade
            self.output[i, 0] = fenzi / fenmu # 为每个样本的输出

    #对可能的激活模糊集进行组合
    def active_combination(self):
        combination_number = 2**self.feature_number # 2**3种可能的激活可能
        combination = np.zeros((combination_number, self.feature_number)) # 8行3列，组合后的矩阵
        for i in range(0, combination_number): # 按行遍历combination
            c = "".join([str((i >> y) & 1) for y in range(self.feature_number - 1, -1, -1)])
            for j in range(0, len(c)):
                combination[i, j] = int(c[j])
        return combination  # 返回的矩阵每行为行号的二进制表示。用于选取所有的可能8种
    
    #将由样本生成的规则扩充至所有cell
    def extrapolate_cell(self):
        then_y_extrapolate = np.zeros((1, self.cell_number)) # 1行20**3列
        then_center_extrapolate = np.zeros((1, self.cell_number)) # 1行20**3列
        zero_cell_index = list(np.where(self.then_y == 0)[1]) # 返回weight=0的位置的下标 ，即寻找(按列)没有参数c的cell的位置 
        while len(zero_cell_index) > 0: # 计算一共有几个cell是没有参数c的
            
            for z in range(0, len(zero_cell_index)): #遍历待扩充的cell，当前为第1个
                fuzzy_set_index = self.cell2fuzzy_index(zero_cell_index[z]) # 实参为没有参数c的位置的下标，此步计算出该cell对应的模糊集的位置(3维坐标，因为特征有3个)
                neighbors = self.get_adjacent_cell(fuzzy_set_index) # 实参为上步计算出cell对应的模糊集的位置，此步便计算出该cell的邻接cells
                neighbor_non_zero_number = 0 # 邻接cell的个数，初始值为0
                for n in range(0, len(neighbors)): # 循环遍历所有邻接cells
                    n_cell_index = self.fuzzy2cell_index(neighbors[n]) # 将模糊集坐标转换成对应的cell_index
                    if self.then_y[0, n_cell_index] == 0: # 如果当前cell的weight=0，即当前cell无参数c
                        continue # 跳过，找下一个
                    else: # 若当前cell有参数c
                        neighbor_non_zero_number = neighbor_non_zero_number + 1 # 记录有参数c的邻接cell的个数
                        then_y_extrapolate[0, zero_cell_index[z]] = then_y_extrapolate[0, zero_cell_index[z]] + self.then_y[0, n_cell_index] # 将当前遍历到的邻接cell的weight都加到当前这个待扩充的无参数c的cell中(放入候补扩充list)
                        then_center_extrapolate[0, zero_cell_index[z]] = then_center_extrapolate[0, zero_cell_index[z]] + self.then_center[0, n_cell_index] # 将当前遍历到的邻接cell的参数c都加到当前这个待扩充的无参数c的cell中(放入候补扩充list)
                if neighbor_non_zero_number == 0: # 若邻接cell都没有参数c
                    continue # 跳出，重新寻找下一个待扩充的cell
                else: # 若该待扩充的cell添加了邻接cell的参数c后
                    then_y_extrapolate[0, zero_cell_index[z]] = then_y_extrapolate[0, zero_cell_index[z]] / neighbor_non_zero_number # 更新weight，为累加的weight的平均值
                    then_center_extrapolate[0, zero_cell_index[z]] = then_center_extrapolate[0, zero_cell_index[z]] / neighbor_non_zero_number # 更新其参数c，为累加的参数c的平均值
                if self.then_y[0, zero_cell_index[z]] == 0 and then_y_extrapolate[0, zero_cell_index[z]] != 0: # 如果原来的weight=0，新的weight不为0时
                    self.then_y[0, zero_cell_index[z]] = then_y_extrapolate[0, zero_cell_index[z]] # 将候补list的中的weight更新到weight空间中
                    self.then_center[0, zero_cell_index[z]] = then_center_extrapolate[0, zero_cell_index[z]] # 将候补list的中的参数c更新到参数c空间中
                else:
                    print (zero_cell_index[z])
            zero_cell_index = list(np.where(self.then_y == 0)[1]) # 再寻找一次，确定所有cell都有参数c后，才能退出循环
    
    
    #获取cell的邻接cell
    def get_adjacent_cell(self, active_index):
        neighbors = []        
        for i in range(0, len(active_index)): # 循环三次，返回邻接cell的坐标值
            neighbor_plus = active_index[:i] + [active_index[i] + 1] + active_index[i + 1:]
            neighbor_minus = active_index[:i] + [active_index[i] - 1] + active_index[i + 1:]
            neighbor_plus_sorted = list(neighbor_plus)
            neighbor_minus_sorted = list(neighbor_minus)
            neighbor_plus_sorted.sort()
            neighbor_minus_sorted.sort()
            if neighbor_plus_sorted[len(neighbor_plus_sorted) - 1] <= (self.fuzzy_set_number - 1):
                neighbors.append(neighbor_plus)
            if neighbor_minus_sorted[0] >= 0:
                neighbors.append(neighbor_minus)
        return neighbors # 返回该cell的所有邻接cells

    
    #求cell的index到对应的active_index
    def cell2fuzzy_index(self, cell_index):  # 10进制转成20进制
        fuzzy_set_index = [] # 准备空列表
        while True: # 进入循环
            shang = cell_index // self.fuzzy_set_number # 求商
            yu = cell_index % self.fuzzy_set_number # 取余
            fuzzy_set_index = fuzzy_set_index + [yu]
            if shang == 0:
                break
            cell_index = shang
        fuzzy_set_index.reverse()
        fuzzy_set_index = list([0] * (self.feature_number - len(fuzzy_set_index)) + fuzzy_set_index) # 补0，让转换后的模糊集序号变为3位数
        return fuzzy_set_index # 返回该cell对应的模糊集序号

        
    #求fuzzy_set_index对应的cell的index
    def fuzzy2cell_index(self, fuzzy_set_index):
        cell_index = 0 # cell序号初始值为0
        for i in range(0, len(fuzzy_set_index)):
            cell_index = cell_index + fuzzy_set_index[i] * (self.fuzzy_set_number**(self.feature_number - i - 1)) # 根据该式计算由模糊集坐标表示的cell对应的在cell空间的序号
        return cell_index # 返回该值
    
    
    #输入模糊规则激活的模糊集index，返回对应的cell的index
    def get_cell_index_grade(self, active_index_grade): # 对active_index_grade进行操作
        cell_index = 0
        cell_grade = 1
        for j in range(0, self.feature_number): # 遍历每个特征对应的模糊集和隶属度
            cell_index = cell_index + active_index_grade[j, 0] * self.fuzzy_set_number**(self.feature_number - j -1) # 当前cell在cells空间中的位置序号
            cell_grade = cell_grade * active_index_grade[j, 1] # 记录对应模糊集的隶属度，并与之后的隶属度进行相乘
        return cell_index, cell_grade # 返回位置序号和隶属度乘积
    
    
    #计算隶属度
    def membership(self, feature_index, fuzzy_set_index, feature_value):
        membership_degree = 0  # 设初始隶属度为0
        feature_min = self.fuzzy_set_range[feature_index, 0] # 查找该特征的最小值
        feature_max = self.fuzzy_set_range[feature_index, 1] # 查找该特征的最大值
        width = (feature_max - feature_min) / (self.fuzzy_set_number - 1) # 计算该特征上划分的20个模糊集的每个区间的宽度
        if fuzzy_set_index == 0: # 若此时遍历到第1个模糊集
            if feature_value < feature_min: # 若当前样本当前特征的值小于该特征的最小值
                membership_degree = 1 # 隶属度为1
            if feature_value >= feature_min and feature_value < (feature_min + width): # 若该值落在模糊集的第一个区间
                membership_degree = (feature_min - feature_value + width) / width
            if feature_value > (feature_min + width): # 若该值不在当前区间
                membership_degree = 0 # 则隶属度为0
        if fuzzy_set_index > 0 and fuzzy_set_index < (self.fuzzy_set_number - 1): # 若当前遍历到第2个-第19个模糊集
            if feature_value < (feature_min + (fuzzy_set_index - 1) * width) and feature_value > (feature_min + (fuzzy_set_index + 1) * width):
                membership_degree = 0
            if feature_value >= (feature_min + (fuzzy_set_index - 1) * width) and feature_value < (feature_min + fuzzy_set_index * width):
                membership_degree = (feature_value - feature_min - (fuzzy_set_index - 1) * width) / width
            if feature_value >= (feature_min + fuzzy_set_index * width) and (feature_value <= feature_min + (fuzzy_set_index + 1) * width):
                membership_degree = (-feature_value + feature_min + (fuzzy_set_index + 1) * width) / width
        if fuzzy_set_index == (self.fuzzy_set_number - 1): # 若当前遍历到最后一个模糊集
            if feature_value < (feature_max - width):
                membership_degree = 0
            if feature_value >= (feature_max - width) and feature_value < feature_max:
                membership_degree = (-feature_max + feature_value + width) / width
            if feature_value >= feature_max:
                membership_degree = 1
        return membership_degree