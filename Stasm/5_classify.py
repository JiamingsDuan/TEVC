import math
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Euclidean distance function
def count_dist(landmark_x1, landmark_y1, landmark_x2, landmark_y2):
    # len =  math.hypot(x1-x2, y1-y2)
    points_length = math.hypot(landmark_x1 - landmark_x2, landmark_y1 - landmark_y2)
    return points_length


# Proportional function
def count_specific(top, below):
    # value = top / below
    value = format(top / below, '.4f')
    return value


# Median function
def count_mid(x1, y1, x2, y2):
    x0 = (x1 + x2) / 2
    y0 = (y1 + y2) / 2
    return x0, y0


def average(left, right):
    return (left + right) * 0.5


# feature selection
def select(x0, y0, k):
    selects = SelectKBest(score_func=f_classif, k=k)
    selects.fit(x0, y0)
    select_best = selects.get_support(True)
    # print('selected index:', select_best)
    return select_best


# Calculate features
def compute_distance(feature_list):
    # 【1】face
    landmark_x = feature_list[1:154:2]
    landmark_y = feature_list[2:155:2]
    length_of_face = count_dist(landmark_x[14], landmark_y[14], landmark_x[6], landmark_y[6])
    width_of_face = count_dist(landmark_x[1], landmark_y[1], landmark_x[11], landmark_y[11])
    RF = count_specific(width_of_face, length_of_face)  # 宽高比
    cheekbones = count_dist(landmark_x[1], landmark_y[1], landmark_x[11], landmark_y[11])
    eyebrow_mid_x, eyebrow_mid_y = count_mid(landmark_x[21], landmark_y[21], landmark_x[22], landmark_y[22])
    eyelid = count_dist(eyebrow_mid_x, eyebrow_mid_y, landmark_x[62], landmark_y[62])
    WHR = count_specific(cheekbones, eyelid)  # 面部宽高比 width / height
    # 【2】eye
    width_l_eye = count_dist(landmark_x[34], landmark_y[34], landmark_x[30], landmark_y[30])
    width_R_eye = count_dist(landmark_x[40], landmark_y[40], landmark_x[44], landmark_y[44])
    eye_width_ave = average(left=width_l_eye, right=width_R_eye)
    height_l_eye = count_dist(landmark_x[32], landmark_y[32], landmark_x[36], landmark_y[36])
    height_R_eye = count_dist(landmark_x[42], landmark_y[42], landmark_x[46], landmark_y[46])
    eye_height_ave = average(height_l_eye, height_R_eye)
    dist_of_pupil = count_dist(landmark_x[38], landmark_y[38], landmark_x[39], landmark_y[39])
    RHL = count_specific(eye_width_ave, eye_height_ave)  # 眼睛宽高均值比
    REF = count_specific(dist_of_pupil, width_of_face)  # 瞳间距占脸宽
    REL = count_specific(eye_width_ave, width_of_face)  # 眼睛宽比脸宽
    REH = count_specific(eye_height_ave, length_of_face)  # 眼睛高比脸长
    # 【3】nose
    height_of_nose = count_dist(eyebrow_mid_x, eyebrow_mid_y, landmark_x[56], landmark_y[56])
    width_of_outer_nose = count_dist(landmark_x[54], landmark_y[54], landmark_x[58], landmark_y[58])
    width_of_inner_nose = count_dist(landmark_x[55], landmark_y[55], landmark_x[57], landmark_y[57])
    RNO = count_specific(width_of_outer_nose, height_of_nose)  # 鼻子外宽高比
    RNI = count_specific(width_of_inner_nose, height_of_nose)  # 鼻子内宽高比
    RHO = count_specific(width_of_inner_nose, width_of_face)  # 鼻子外宽占脸宽
    RHI = count_specific(width_of_outer_nose, width_of_face)  # 鼻子内宽占脸宽
    # 【4】jaw
    L_mouse_line_lip = count_dist(landmark_x[3], landmark_y[3], landmark_x[70], landmark_y[70])
    R_mouse_line_lip = count_dist(landmark_x[9], landmark_y[9], landmark_x[70], landmark_y[70])
    mouse_line_lip = average(L_mouse_line_lip, R_mouse_line_lip)
    tip_of_chin = count_dist(landmark_x[6], landmark_y[6], landmark_x[70], landmark_y[70])  # TOC
    jaw_dist_01 = count_dist(landmark_x[7], landmark_y[7], landmark_x[5], landmark_y[5])  # JAW1
    jaw_dist_02 = count_dist(landmark_x[8], landmark_y[8], landmark_x[4], landmark_y[4])  # JAW2
    jaw_dist_03 = count_dist(landmark_x[9], landmark_y[9], landmark_x[3], landmark_y[3])  # JAW3
    RML = count_specific(mouse_line_lip, width_of_face)  # 唇中线比脸宽
    RTC = count_specific(tip_of_chin, length_of_face)  # 下唇中心到下巴比脸长
    RJ1 = count_specific(jaw_dist_01, width_of_face)  # 中唇线比脸宽
    RJ2 = count_specific(jaw_dist_02, width_of_face)  # 下巴中线比脸宽
    RJ3 = count_specific(jaw_dist_03, width_of_face)  # 下巴底线比脸宽
    # 【5】eyebrow
    width_l_eyebrow = count_dist(landmark_x[18], landmark_y[18], landmark_x[21], landmark_y[21])
    width_R_eyebrow = count_dist(landmark_x[22], landmark_y[22], landmark_x[25], landmark_y[25])
    height_l_eyebrow = count_dist(landmark_x[16], landmark_y[16], landmark_x[20], landmark_y[20])
    height_R_eyebrow = count_dist(landmark_x[23], landmark_y[23], landmark_x[27], landmark_y[27])
    inner_eyebrow_dist = count_dist(landmark_x[22], landmark_y[22], landmark_x[21], landmark_y[21])
    outer_eyebrow_dist = count_dist(landmark_x[18], landmark_y[18], landmark_x[25], landmark_y[25])
    L_eyebrow_mid_x, L_eyebrow_mid_y = count_mid(landmark_x[18], landmark_y[18], landmark_x[21], landmark_y[21])
    R_eyebrow_mid_x, R_eyebrow_mid_y = count_mid(landmark_x[22], landmark_y[22], landmark_x[25], landmark_y[25])
    dist_of_mid_eyebrow = count_dist(L_eyebrow_mid_x, L_eyebrow_mid_y, R_eyebrow_mid_x, R_eyebrow_mid_y)
    width_of_eyebrow = average(width_l_eyebrow, width_R_eyebrow)
    height_of_eyebrow = average(height_l_eyebrow, height_R_eyebrow)
    RUI = count_specific(inner_eyebrow_dist, outer_eyebrow_dist)  # 眉内间距比外间距
    RBL = count_specific(height_of_eyebrow, length_of_face)  # 眉毛高比脸长
    RBH = count_specific(width_of_eyebrow, width_of_face)  # 眉毛宽比脸宽
    RWH = count_specific(width_of_eyebrow, height_of_eyebrow)  # 眉毛宽高比
    RME = count_specific(dist_of_mid_eyebrow, width_of_face)  # 眉心间距比脸宽
    # 【6】mouse
    width_of_mouse = count_dist(landmark_x[59], landmark_y[59], landmark_x[65], landmark_y[65])
    height_of_mouse = count_dist(landmark_x[62], landmark_y[62], landmark_x[74], landmark_y[74])
    nose_to_mouse = count_dist(landmark_x[52], landmark_y[52], landmark_x[67], landmark_y[67])
    nose_to_jaw = count_dist(landmark_x[52], landmark_y[52], landmark_x[6], landmark_y[6])
    mouse_to_jaw = count_dist(landmark_x[70], landmark_y[70], landmark_x[6], landmark_y[6])
    RWM = count_specific(width_of_mouse, height_of_mouse)  # 嘴巴宽高比
    RMW = count_specific(width_of_mouse, width_of_face)  # 嘴巴宽比脸宽
    RMH = count_specific(height_of_mouse, length_of_face)  # 嘴巴高比脸长
    RNM = count_specific(nose_to_mouse, length_of_face)  # 鼻子到嘴巴占脸长
    RNF = count_specific(nose_to_jaw, length_of_face)  # 鼻子到下巴占脸长
    RMF = count_specific(mouse_to_jaw, length_of_face)  # 嘴到下巴占脸长
    # 【7】forehead
    eyebrow_top_mid_x, eyebrow_top_mid_y = count_mid(landmark_x[16], landmark_y[16], landmark_x[23], landmark_y[23])
    height_of_forehead = count_dist(eyebrow_top_mid_x, eyebrow_top_mid_y, landmark_x[14], landmark_y[14])
    RFL = count_specific(height_of_forehead, length_of_face)

    '''
    L_eyebrow_to_pupil = count_dist(L_eyebrow_mid_x, L_eyebrow_mid_y, landmark_x[38], landmark_y[38])
    R_eyebrow_to_pupil = count_dist(R_eyebrow_mid_x, R_eyebrow_mid_y, landmark_x[39], landmark_y[39])
    RVEL = count_specific(L_eyebrow_to_pupil, length_of_face)  # 左眉到左瞳孔占脸长
    RVER = count_specific(R_eyebrow_to_pupil, length_of_face)  # 右眉到右瞳孔占脸长

    L_eyebrow_to_nose = count_dist(landmark_x[21], landmark_y[21], landmark_x[57], landmark_y[57])
    R_eyebrow_to_nose = count_dist(landmark_x[22], landmark_y[22], landmark_x[58], landmark_y[58])
    RVLN = count_specific(L_eyebrow_to_nose, length_of_face)  # 左眉到鼻子占脸长
    RVRN = count_specific(R_eyebrow_to_nose, length_of_face)  # 右眉到鼻子占脸长

    L_eyebrow_to_mouse = count_dist(landmark_x[21], landmark_y[21], landmark_x[75], landmark_y[75])
    R_eyebrow_to_mouse = count_dist(landmark_x[22], landmark_y[22], landmark_x[73], landmark_y[73])
    RVLM = count_specific(L_eyebrow_to_mouse, length_of_face)  # 左眉到嘴占脸长
    RVRM = count_specific(R_eyebrow_to_mouse, length_of_face)  # 右眉到嘴占脸长

    L_eyebrow_to_jaw = count_dist(landmark_x[21], landmark_y[21], landmark_x[6], landmark_y[6])
    R_eyebrow_to_jaw = count_dist(landmark_x[22], landmark_y[22], landmark_x[6], landmark_y[6])
    RVLF = count_specific(L_eyebrow_to_jaw, length_of_face)  # 左眉到下巴占脸长
    RVRF = count_specific(R_eyebrow_to_jaw, length_of_face)  # 右眉到下巴占脸长

    L_pupil_to_nose = count_dist(landmark_x[38], landmark_y[38], landmark_x[58], landmark_y[58])
    R_pupil_to_nose = count_dist(landmark_x[39], landmark_y[39], landmark_x[54], landmark_y[54])
    RELN = count_specific(L_pupil_to_nose, length_of_face)  # 左瞳孔到鼻子占脸长
    RERN = count_specific(R_pupil_to_nose, length_of_face)  # 右瞳孔到鼻子占脸长

    L_pupil_to_mouse = count_dist(landmark_x[38], landmark_y[38], landmark_x[59], landmark_y[59])
    R_pupil_to_mouse = count_dist(landmark_x[39], landmark_y[39], landmark_x[65], landmark_y[65])
    RELM = count_specific(L_pupil_to_mouse, length_of_face)  # 左瞳孔到嘴占脸长
    RERM = count_specific(R_pupil_to_mouse, length_of_face)  # 右瞳孔到嘴占脸长

    L_pupil_to_jaw = count_dist(landmark_x[38], landmark_y[38], landmark_x[5], landmark_y[5])
    R_pupil_to_jaw = count_dist(landmark_x[39], landmark_y[39], landmark_x[7], landmark_y[7])
    RELF = count_specific(L_pupil_to_jaw, length_of_face)  # 左瞳孔到下巴占脸长
    RERF = count_specific(R_pupil_to_jaw, length_of_face)  # 右瞳孔到下巴占脸长
    '''

    feature = [RF, WHR, RHL, REF, REL, REH, RNO, RNI, RHO, RHI,
               RML, RTC, RJ1, RJ2, RJ3, RUI, RBL, RBH, RWH, RME,
               RWM, RMW, RMH, RNM, RNF, RMF, RFL]

    return feature


# Calculate feature
def compute_feature(feature_list):
    landmark_x = feature_list[1:154:2]
    landmark_y = feature_list[2:155:2]
    length_of_face = count_dist(landmark_x[14], landmark_y[14], landmark_x[6], landmark_y[6])
    width_of_face = count_dist(landmark_x[1], landmark_y[1], landmark_x[11], landmark_y[11])
    RF = count_specific(width_of_face, length_of_face)  # 长宽比

    # 【00】脸
    # fWHR：宽度——两颧骨宽度；高度——上嘴唇和眉毛之间的距离
    cheekbones = count_dist(landmark_x[1], landmark_y[1], landmark_x[11], landmark_y[11])
    eyebrow_bone_mid_x, eyebrow_bone_mid_y = count_mid(landmark_x[21], landmark_y[21], landmark_x[22], landmark_y[22])
    eyelid = count_dist(eyebrow_bone_mid_x, eyebrow_bone_mid_y, landmark_x[62], landmark_y[62])
    WHR = count_specific(eyelid, cheekbones)  # 面部宽高比

    # 【01】眼睛
    length_l_eye = count_dist(landmark_x[34], landmark_y[34], landmark_x[30], landmark_y[30])
    length_R_eye = count_dist(landmark_x[40], landmark_y[40], landmark_x[44], landmark_y[44])
    width_l_eye = count_dist(landmark_x[32], landmark_y[32], landmark_x[36], landmark_y[36])
    width_R_eye = count_dist(landmark_x[42], landmark_y[42], landmark_x[46], landmark_y[46])
    dist_of_pupil = count_dist(landmark_x[38], landmark_y[38], landmark_x[39], landmark_y[39])

    L_inner_to_pupil = count_dist(landmark_x[30], landmark_y[30], landmark_x[38], landmark_y[38])  # LIP
    L_top_right_to_pupil = count_dist(landmark_x[31], landmark_y[31], landmark_x[38], landmark_y[38])  # LTRP
    L_top_to_pupil = count_dist(landmark_x[32], landmark_y[32], landmark_x[38], landmark_y[38])  # LTP
    L_top_left_to_pupil = count_dist(landmark_x[33], landmark_y[33], landmark_x[38], landmark_y[38])  # LTLP
    L_outer_to_pupil = count_dist(landmark_x[34], landmark_y[34], landmark_x[38], landmark_y[38])  # LOP
    L_bot_left_to_pupil = count_dist(landmark_x[35], landmark_y[35], landmark_x[38], landmark_y[38])  # LBLP
    L_bot_to_pupil = count_dist(landmark_x[36], landmark_y[36], landmark_x[38], landmark_y[38])  # LBP
    L_bot_right_to_pupil = count_dist(landmark_x[37], landmark_y[37], landmark_x[38], landmark_y[38])  # LBRP

    R_inner_to_pupil = count_dist(landmark_x[44], landmark_y[44], landmark_x[39], landmark_y[39])  # RIP
    R_top_right_to_pupil = count_dist(landmark_x[43], landmark_y[43], landmark_x[39], landmark_y[39])  # RTRP
    R_top_to_pupil = count_dist(landmark_x[42], landmark_y[42], landmark_x[39], landmark_y[39])  # RTP
    R_top_left_to_pupil = count_dist(landmark_x[41], landmark_y[41], landmark_x[39], landmark_y[39])  # RTLP
    R_outer_to_pupil = count_dist(landmark_x[40], landmark_y[40], landmark_x[39], landmark_y[39])  # ROP
    R_bot_left_to_pupil = count_dist(landmark_x[47], landmark_y[47], landmark_x[39], landmark_y[39])  # RBLP
    R_bot_to_pupil = count_dist(landmark_x[46], landmark_y[46], landmark_x[39], landmark_y[39])  # RBP
    R_bot_right_to_pupil = count_dist(landmark_x[45], landmark_y[45], landmark_x[39], landmark_y[39])  # RBRP

    REL = count_specific(width_l_eye, length_l_eye)  # 左眼宽高比
    RER = count_specific(width_R_eye, length_R_eye)  # 右眼宽高比
    REF = count_specific(dist_of_pupil, width_of_face)  # 瞳间距占脸宽
    RFEL = count_specific(width_l_eye, length_of_face)  # 左眼宽占脸长
    RFER = count_specific(width_R_eye, length_of_face)  # 右眼宽占脸长
    REFL = count_specific(length_l_eye, width_of_face)  # 左眼长占脸宽
    REFR = count_specific(length_R_eye, width_of_face)  # 右眼长占脸宽

    LIP = count_specific(L_inner_to_pupil, length_l_eye)
    LTRP = count_specific(L_top_right_to_pupil, length_l_eye)
    LTP = count_specific(L_top_to_pupil, length_l_eye)
    LTLP = count_specific(L_top_left_to_pupil, length_l_eye)
    LOP = count_specific(L_outer_to_pupil, length_l_eye)
    LBLP = count_specific(L_bot_left_to_pupil, length_l_eye)
    LBP = count_specific(L_bot_to_pupil, length_l_eye)
    LBRP = count_specific(L_bot_right_to_pupil, length_l_eye)

    RIP = count_specific(R_inner_to_pupil, length_R_eye)
    RTRP = count_specific(R_top_right_to_pupil, length_R_eye)
    RTP = count_specific(R_top_to_pupil, length_R_eye)
    RTLP = count_specific(R_top_left_to_pupil, length_R_eye)
    ROP = count_specific(R_outer_to_pupil, length_R_eye)
    RBLP = count_specific(R_bot_left_to_pupil, length_R_eye)
    RBP = count_specific(R_bot_to_pupil, length_R_eye)
    RBRP = count_specific(R_bot_right_to_pupil, length_R_eye)

    # 【02】鼻子
    mid_eyebrow_x, mid_eyebrow_y = count_mid(landmark_x[21], landmark_y[21], landmark_x[22], landmark_y[22])
    length_of_nose = count_dist(mid_eyebrow_x, mid_eyebrow_y, landmark_x[56], landmark_y[56])
    width_of_outer_nose = count_dist(landmark_x[54], landmark_y[54], landmark_x[58], landmark_y[58])
    width_of_inner_nose = count_dist(landmark_x[55], landmark_y[55], landmark_x[57], landmark_y[57])
    L_nose_mid = count_dist(landmark_x[50], landmark_y[50], landmark_x[52], landmark_y[52])  # LNmid
    R_nose_mid = count_dist(landmark_x[48], landmark_y[48], landmark_x[52], landmark_y[52])  # RNmid
    C_nose_mid = count_dist(landmark_x[49], landmark_y[49], landmark_x[52], landmark_y[52])  # CNmid
    L_nostril_top = count_dist(landmark_x[51], landmark_y[51], landmark_x[52], landmark_y[52])  # LNT
    L_nostril_left = count_dist(landmark_x[57], landmark_y[57], landmark_x[52], landmark_y[52])  # LNL
    R_nostril_top = count_dist(landmark_x[53], landmark_y[53], landmark_x[52], landmark_y[52])  # RNT
    R_nostril_right = count_dist(landmark_x[55], landmark_y[55], landmark_x[52], landmark_y[52])  # RNR
    C_nose_base = count_dist(landmark_x[56], landmark_y[56], landmark_x[52], landmark_y[52])  # CNB
    L_nose_side = count_dist(landmark_x[58], landmark_y[58], landmark_x[52], landmark_y[52])  # LNS
    R_nose_side = count_dist(landmark_x[54], landmark_y[54], landmark_x[52], landmark_y[52])  # RNS
    RN = count_specific(width_of_outer_nose, length_of_nose)  # 鼻子宽高比
    RNL = count_specific(length_of_nose, length_of_face)  # 鼻子长占脸长
    ROH = count_specific(width_of_inner_nose, width_of_face)  # 鼻子外宽占脸宽
    RIH = count_specific(width_of_outer_nose, width_of_face)  # 鼻子内宽占脸宽
    LNmid = count_specific(L_nose_mid, length_of_nose)
    RNmid = count_specific(R_nose_mid, length_of_nose)
    CNmid = count_specific(C_nose_mid, length_of_nose)
    LNT = count_specific(L_nostril_top, width_of_outer_nose)
    LNL = count_specific(L_nostril_left, width_of_outer_nose)
    RNT = count_specific(R_nostril_top, width_of_outer_nose)
    RNR = count_specific(R_nostril_right, width_of_outer_nose)
    CNB = count_specific(C_nose_base, length_of_nose)
    LNS = count_specific(L_nose_side, width_of_outer_nose)
    RNS = count_specific(R_nose_side, width_of_outer_nose)

    # 【03】下巴
    L_mouse_line_lip = count_dist(landmark_x[3], landmark_y[3], landmark_x[70], landmark_y[70])
    L_jaw_top = count_dist(landmark_x[4], landmark_y[4], landmark_x[70], landmark_y[70])  # LJT
    L_jaw_bot = count_dist(landmark_x[5], landmark_y[5], landmark_x[70], landmark_y[70])  # LJB
    tip_of_chin = count_dist(landmark_x[6], landmark_y[6], landmark_x[70], landmark_y[70])  # TOC
    R_jaw_top = count_dist(landmark_x[8], landmark_y[8], landmark_x[70], landmark_y[70])  # RJT
    R_jaw_bot = count_dist(landmark_x[7], landmark_y[7], landmark_x[70], landmark_y[70])  # RJB
    jaw_dist_01 = count_dist(landmark_x[7], landmark_y[7], landmark_x[5], landmark_y[5])  # JAW1
    jaw_dist_02 = count_dist(landmark_x[8], landmark_y[8], landmark_x[4], landmark_y[4])  # JAW2
    jaw_dist_03 = count_dist(landmark_x[9], landmark_y[9], landmark_x[3], landmark_y[3])  # JAW3
    LML = count_specific(L_mouse_line_lip, width_of_face)
    LJT = count_specific(L_jaw_top, width_of_face)
    LJB = count_specific(L_jaw_bot, width_of_face)
    TOC = count_specific(tip_of_chin, eyelid)
    RJT = count_specific(R_jaw_top, width_of_face)
    RJB = count_specific(R_jaw_bot, width_of_face)
    JAW1 = count_specific(jaw_dist_01, width_of_face)
    JAW2 = count_specific(jaw_dist_02, width_of_face)
    JAW3 = count_specific(jaw_dist_03, width_of_face)

    # 【04】眉毛
    length_l_eyebrow = count_dist(landmark_x[18], landmark_y[18], landmark_x[21], landmark_y[21])
    length_R_eyebrow = count_dist(landmark_x[22], landmark_y[22], landmark_x[25], landmark_y[25])
    width_l_eyebrow = count_dist(landmark_x[16], landmark_y[16], landmark_x[20], landmark_y[20])
    width_R_eyebrow = count_dist(landmark_x[23], landmark_y[23], landmark_x[27], landmark_y[27])
    inner_eyebrow_dist = count_dist(landmark_x[22], landmark_y[22], landmark_x[21], landmark_y[21])
    outer_eyebrow_dist = count_dist(landmark_x[18], landmark_y[18], landmark_x[25], landmark_y[25])
    RVL = count_specific(width_l_eyebrow, length_l_eyebrow)  # 左眉宽高比
    RVR = count_specific(width_R_eyebrow, length_R_eyebrow)  # 右眉宽高比
    RIF = count_specific(inner_eyebrow_dist, width_of_face)  # 内眉间距
    ROF = count_specific(outer_eyebrow_dist, width_of_face)  # 外眉间距
    RVFL = count_specific(length_l_eyebrow, width_of_face)  # 左眉长占脸宽
    RVFR = count_specific(length_R_eyebrow, width_of_face)  # 右眉长占脸宽
    RFVL = count_specific(width_l_eyebrow, length_of_face)  # 左眉宽占脸长
    RFVR = count_specific(width_R_eyebrow, length_of_face)  # 右眉宽占脸长

    # 【05】嘴
    length_of_mouse = count_dist(landmark_x[59], landmark_y[59], landmark_x[65], landmark_y[65])
    width_of_mouse = count_dist(landmark_x[62], landmark_y[62], landmark_x[74], landmark_y[74])
    L_mouth_cup = count_dist(landmark_x[61], landmark_y[61], landmark_x[67], landmark_y[67])
    R_mouth_cup = count_dist(landmark_x[63], landmark_y[63], landmark_x[67], landmark_y[67])
    C_mouth_cup = count_dist(landmark_x[62], landmark_y[62], landmark_x[67], landmark_y[67])
    L_mouth_bot = count_dist(landmark_x[75], landmark_y[75], landmark_x[70], landmark_y[70])
    R_mouth_bot = count_dist(landmark_x[73], landmark_y[73], landmark_x[70], landmark_y[70])
    C_mouth_bot = count_dist(landmark_x[74], landmark_y[74], landmark_x[70], landmark_y[70])
    RM = count_specific(width_of_mouse, length_of_mouse)  # 嘴巴长宽比
    RML = count_specific(length_of_mouse, width_of_face)  # 嘴巴长占脸宽
    RMW = count_specific(width_of_mouse, length_of_face)  # 嘴巴宽占脸长
    LMC = count_specific(L_mouth_cup, length_of_mouse)
    RMC = count_specific(R_mouth_cup, length_of_mouse)
    CMC = count_specific(C_mouth_cup, width_of_mouse)
    LMB = count_specific(L_mouth_bot, length_of_mouse)
    RMB = count_specific(R_mouth_bot, length_of_mouse)
    CMB = count_specific(C_mouth_bot, width_of_mouse)

    # 【06】额头
    forehead_to_nose = count_dist(landmark_x[14], landmark_y[14], landmark_x[52], landmark_y[52])
    forehead_to_mouse = count_dist(landmark_x[14], landmark_y[14], landmark_x[70], landmark_y[70])
    RFN = count_specific(forehead_to_nose, length_of_face)  # 额头到鼻尖占脸长
    RFM = count_specific(forehead_to_mouse, length_of_face)  # 额头到嘴巴占脸长

    attribute = [RF, REL, RER, REF, REFL, RFER, RFEL, REFL, REFR,
                 LIP, LTRP, LTP, LTLP, LOP, LBLP, LBP, LBRP,
                 RIP, RTRP, RTP, RTLP, ROP, RBLP, RBP, RBRP,
                 RN, RNL, ROH, RIH, LNmid, RNmid, CNmid,
                 LNT, LNL, RNT, RNR, CNB, LNS, RNS, LML,
                 LJT, LJB, TOC, RJT, RJB, JAW1, JAW2, JAW3,
                 RIF, ROF, RVFL, RVFR, RFVL, RFVR, RVL, RVR,
                 RM, RML, RMW, LMC, RMC, CMC, LMB, RMB, CMB, RFN, RFM,
                 WHR]

    return attribute


# features landmarks
landmark_csv_path = 'csv_data/Landmark_529.csv'
landmark_frame = pd.read_csv(landmark_csv_path)
# personality traits
score_csv_path = 'csv_data/zong_dawu_529.csv'
score = pd.read_csv(score_csv_path)[['N', 'E', 'O', 'A', 'C']]
# students'genders
big_five_path = 'csv_data/zong_dawu_529.csv'
big_five = pd.read_csv(big_five_path)
rows = score.shape[0]
# initialize a empty frame
Dataset = pd.DataFrame(columns=list(range(0, len(compute_distance(list(landmark_frame.iloc[0, :]))))),
                       index=list(range(0, rows)))
# fill in the features
for row in range(0, rows):
    stu_landmark = list(landmark_frame.iloc[row, :])
    features_list = compute_distance(stu_landmark)
    features = [float(i) for i in features_list]
    Dataset.iloc[row] = features
# 5-classification labels
trait_list = []
for row in range(0, rows):
    five_trait = list(score.iloc[row, :])
    trait_index = five_trait.index(max(five_trait)) + 1
    trait_list.append(trait_index)
# encoding personalities'traits into 0~1
gender_list = []
for gen in big_five['性别'].tolist():
    if gen == '男':
        gen = 0
    else:
        gen = 1
    gender_list.append(gen)
# add personality trait
Dataset.insert(0, 'T', value=trait_list)
# add students'gender
Dataset.insert(0, 'Gender', value=gender_list)
M_dataset = Dataset.drop(index=Dataset.loc[(Dataset['Gender'] == 0)].index).reset_index(drop=True)
# obtain the female's dataset
F_dataset = Dataset.drop(index=Dataset.loc[(Dataset['Gender'] == 1)].index).reset_index(drop=True)


# select the dataset if gender==1 choose female if gender==0 choose male other is all data
def gender_divide(gender):
    if gender == 1 or gender == 0:
        data = Dataset.drop(index=Dataset.loc[(Dataset['Gender'] == gender)].index).reset_index(drop=True)
    else:
        data = Dataset
    return data


# save data into csv file
Dataset.to_csv('csv_data/features_27.csv', sep=',', encoding='utf-8', index=False, float_format='%.4f')
# obtain the dataset
dataset = gender_divide(2)
# obtain the attributes
attributes = dataset.iloc[:, -27:]
# obtain the num of feature
col = attributes.shape[1]
# obtain the labels
labels = dataset.iloc[:, 1]
X = attributes.values
y = labels.values

for step in range(5, col+1):
    selector = SelectKBest(score_func=f_classif, k=step)
    selector.fit(X, y)
    select_best_index = selector.get_support(True)
    X0 = attributes.iloc[:, select_best_index].values
    y0 = y
    X_train, X_test, y_train, y_test = train_test_split(X0, y0, test_size=0.25, random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    classifier = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accurate = accuracy_score(y_test, y_pred)
    print('accurate:', '%.2f' % accurate, 'feature_num:k=', step)
