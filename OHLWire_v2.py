import numpy as np
from scipy.special import iv as besseli
from scipy.special import kv as besselk
import math
from Node import Node

"""
系统参数
"""
frq_default = np.logspace(0, 9, 37)
Lseg = 20#

"""
常数
"""

class Ground:
    def __init__(self, sig, mur, epr, gnd_model, ionisation_intensity, ionisation_model):
        """
        sig(float):电导率
        mur(float):相对磁导率
        epr(float):相对介电常数
        gnd_model(str):接地模型
        ionisation_intensity(str):电离强度
        ionisation_model(str):电离模型
        """
        self.sig = sig
        self.mur = mur
        self.epr = epr
        self.gnd_model = gnd_model
        self.ionisation_intensity = ionisation_intensity
        self.ionisation_model = ionisation_model


def OHL_mutual_inductance_calculate(r, height, end_node_y):
    """
    【函数功能】电感电容矩阵参数计算
    【入参】
    end_node_y (numpy.ndarray,n*1): n条线的第二个节点的y值
    height(numpy.ndarray,n*1):n条线高
    r (numpy.ndarray,n*1): n条线的半径

    【出参】
    Lm(numpy.ndarray:n*n)：n条线互感矩阵
    """
    mu0 = 4 * np.pi * 1e-7
    km = mu0 / (2 * np.pi)
    Ncon = np.array([r]).reshape(-1).shape[0]
    out = np.log(2 * height / r)
    Lm = np.diag(out.reshape(-1))
    for i in range(Ncon - 1):
        for j in range(i + 1, Ncon):
            d = abs(end_node_y[i] - end_node_y[j])
            h1 = height[i]
            h2 = height[j]
            Lm[i, j] = 0.5 * np.log((d ** 2 + (h1 + h2) ** 2) / (d ** 2 + (h1 - h2) ** 2))
            Lm[j, i] = np.copy(Lm[i, j])
    Lm = km * Lm
    return Lm


def OHL_inductance_calculate(l, Lm):
    """
    【函数功能】电感矩阵参数计算
    【入参】
    Lm(numpy.ndarray:n*n)：n条线互感矩阵
    l(numpy.ndarray,n*1): n条线的电感

    【出参】
    L(numpy.ndarray:n*n)：n条线的电感矩阵
    """
    L = Lm + np.diag(l)
    return L


def OHL_capcitance_calculate(Lm):
    """
    【函数功能】电感矩阵参数计算
    【入参】
    Lm(numpy.ndarray:n*n)：n条线互感矩阵

    【出参】
    C(numpy.ndarray:n*n)：n条线电容矩阵
    """
    Vair = 3e8
    C = np.linalg.inv(Lm) / Vair ** 2
    return C


def OHL_wire_impedance_calculate(r, mur, sig, epr, frq=frq_default):
    """
    【函数功能】导线阻抗参数计算
    【入参】
    r (numpy.ndarray,n*1): n条线线的半径
    sig (numpy.ndarray,n*1): n条线线的电导率
    mur (numpy.ndarray,n*1): n条线线的磁导率
    epr (numpy.ndarray,n*1): n条线线的相对介电常数
    frq(numpy.ndarray，1*Nf):Nf个频率组成的频率矩阵

    【出参】
    Zc(numpy.ndarray:n*n*Nf)：n条线在Nf个频率下的阻抗矩阵
    """
    mu0 = 4 * np.pi * 1e-7
    ep0 = 8.854187818e-12
    Emax = 350
    Ncon = np.array([r]).reshape(-1).shape[0]
    Nf = np.array([frq]).reshape(-1).shape[0]
    Zc = np.zeros((Ncon, Ncon, Nf), dtype='complex')
    omega = 2 * np.pi * frq
    for i in range(Nf):
        gamma = np.sqrt(1j * mu0 * mur * omega[i] * sig + 1j * omega[i] * ep0 * epr)
        Ri = r * gamma
        I0i = besseli(0, Ri)
        I1i = besseli(1, Ri)
        out = gamma / (2 * np.pi * r * sig)
        low = np.where(abs(Ri) < Emax)
        out[low] = 1j * mu0 * mur[low] * omega[i] * I0i[low] / (2 * np.pi * Ri[low] * I1i[low])
        Zc[:, :, i] = np.diag(out)
    return Zc


def OHL_ground_impedance_calculate(sig_g, mur_g, epr_g, r, offset, height, frq=frq_default):
    """
    【函数功能】大地阻抗参数计算
    【入参】
    offset (numpy.ndarray,n*1): n条线的偏置
    r (numpy.ndarray,n*1): n条线的半径
    sig_g (float): 大地的电导率
    mur_g (float): 大地的磁导率
    epr_g (float): 大地的相对介电常数
    frq(numpy.ndarray，1*Nf):Nf个频率组成的频率矩阵

    【出参】
    Zg(numpy.ndarray:n*n*Nf)：n条线对应的大地阻抗矩阵
    """
    mu0 = 4 * np.pi * 1e-7
    ep0 = 8.854187818e-12
    Sig_g = sig_g
    Mur_g = mur_g * mu0
    Eps_g = epr_g * ep0
    Ncon = np.array([r]).reshape(-1).shape[0]
    Nf = np.array([frq]).reshape(-1).shape[0]
    Zg = np.zeros((Ncon, Ncon, Nf), dtype='complex')
    omega = 2 * np.pi * frq
    gamma = np.sqrt(1j * Mur_g * omega * (Sig_g + 1j * omega * Eps_g))
    km = 1j * omega * Mur_g / 4 / np.pi
    for i in range(Ncon):
        for j in range(i, Ncon):
            d = abs(offset[i] - offset[j])
            h1 = height[i]
            h2 = height[j]
            Zg[i, j, :] = km * np.log(((1 + gamma * (h1 + h2) / 2) ** 2 + (d * gamma / 2) ** 2) / (
                    (gamma * (h1 + h2) / 2) ** 2 + (d * gamma / 2) ** 2))
            Zg[j, i, :] = np.copy(Zg[i, j, :])
    for i in range(Ncon):
        h = height[i]
        Zg[i, i, :] = km * np.log(((1 + gamma * h) ** 2) / ((gamma * h) ** 2))
    return Zg


def OHL_impedance_calculate(r, mur, sig, epr, offset, height, sig_g, mur_g, epr_g, Lm, frq=frq_default):
    """
    【函数功能】阻抗矩阵参数计算
    【入参】
    height(numpy.ndarray,n*1):n条线高
    offset (numpy.ndarray,n*1): n条线的偏置
    r (numpy.ndarray,n*1): n条线的半径
    sig (numpy.ndarray,n*1): n条线的电导率
    mur (numpy.ndarray,n*1): n条线的磁导率
    epr (numpy.ndarray,n*1): 线n条的相对介电常数
    sig_g (float): 大地的电导率
    mur_g (float): 大地的磁导率
    epr_g (float): 大地的相对介电常数
    Lm(numpy.ndarray:n*n)：n条线的互感矩阵
    frq(numpy.ndarray，1*Nf):Nf个频率组成的频率矩阵

    【出参】
    Z(numpy.ndarray:n*n)：n条线的阻抗矩阵
    """
    Zc = OHL_wire_impedance_calculate(r, mur, sig, epr, frq)
    Zg = OHL_ground_impedance_calculate(sig_g, mur_g, epr_g, r, offset, height, frq)
    Nf = np.array([frq]).reshape(-1).shape[0]
    Z = Zc + Zg
    for i in range(Nf):
        Z[:, :, i] += 1j * 2 * np.pi * frq[i] * Lm
    return Z


def OHL_resistance_calculate(R):
    """
    【函数功能】电阻矩阵参数计算
    【入参】
    R (numpy.ndarray,n*1): n条线的电阻
    """
    return np.diag(R)


def OHL_parameters_calculate(OHL, frq=frq_default):
    """
    【函数功能】线路参数计算
    【入参】
    OHL (OHLWire): 管状线段对象
    GND(Ground):大地对象

    【出参】
    R (numpy.ndarray,n*n): n条线的电阻矩阵
    Z(numpy.ndarray:n*n)：n条线的阻抗矩阵
    L(numpy.ndarray:n*n)：n条线的电感矩阵
    C(numpy.ndarray:n*n)：n条线的电容矩阵
    """
    OHL_r = OHL.get_r()
    OHL_height = OHL.get_height()
    Lm = OHL_mutual_inductance_calculate(OHL_r, OHL_height, OHL.get_end_node_y())
    L = OHL_inductance_calculate(OHL.get_L(), Lm)
    C = OHL_capcitance_calculate(Lm)
    Z = OHL_impedance_calculate(OHL_r, OHL.get_mur(), OHL.get_sig(), OHL.get_epr(), OHL.get_offset(), OHL_height,
                                OHL.ground.sig, OHL.ground.mur, OHL.ground.epr, Lm, frq)
    R = OHL_resistance_calculate(OHL.get_R())
    return R, L, Z, C



class Wire:
    def __init__(self, name: str, start_node: Node, end_node: Node, offset: float, r: float, R: float, L: float,
                 sig: float, mur: float, epr: float, VF):
        """
        初始化管状线段对象

        参数说明:
        name (str): 线的名称
        start_node (Node): 线的第一个节点
        end_node (Node): 线的第二个节点
        offset (float): 线的偏置
        r (float): 线的半径
        R (float): 线的电阻
        L (float): 线的电感
        sig (float): 线的电导率
        mur (float): 线的磁导率
        epr (float): 线的相对介电常数
        VF (int): 线的向量拟合矩阵
        inner_num (int): 线的内部导体数量
        """
        self.name = name
        self.start_node = start_node
        self.end_node = end_node
        self.offset = offset
        self.r = r
        self.R = R
        self.L = L
        self.sig = sig
        self.mur = mur
        self.epr = epr
        self.VF = VF
        self.inner_num = 1
        self.height = (end_node.z + start_node.z) / 2

    def length(self):
        dx = self.end_node.x - self.start_node.x
        dy = self.end_node.y - self.start_node.y
        dz = self.end_node.z - self.start_node.z
        return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def display(self):
        """
        打印线段对象信息。
        """
        print(
            f"Wire(name='{self.name}', start_node={self.start_node}, end_node={self.end_node}, offset={self.offset}, r={self.r}, R={self.R}, L={self.L}, sig={self.sig}, mur={self.mur}, epr={self.epr}, inner_num={self.inner_num}, VF_matrix is not showned here.)\n")

    def __repr__(self):
        """
        返回线段对象的字符串表示形式。
        """
        return f"Wire(name='{self.name}', start_node={self.start_node}, end_node={self.end_node}, offset={self.offset}, r={self.r}, R={self.R}, L={self.L}, sig={self.sig}, mur={self.mur}, epr={self.epr}, inner_num={self.inner_num}, VF_matrix is not showned here.)"


class OHLWire:
    def __init__(self, name: str, Cir_No, Phase, phase_num: int, ground: Ground):
        """
        初始化架空线对象。
        name (str): 线的名称
        Cir_No (int): 线圈回路号
        Phase (str): 线圈相线
        phase_num (int): 线圈相数
        ground(Ground类)：大地类
        """
        self.name = name
        self.wires = []
        self.Cir_No = Cir_No
        self.Phase = Phase
        self.phase_num = phase_num
        self.ground = ground
        self.R = np.zeros((phase_num, phase_num))
        self.L = np.zeros((phase_num, phase_num))
        self.C = np.zeros((phase_num, phase_num))
        self.Z = np.zeros((phase_num, phase_num))

    def add_wire(self, wire: Wire):
        """
        向架空线中添加线段。

        Args:
            wire (Wire): 要添加的线段。
        """
        if len(self.wires) >= self.phase_num:
            raise ValueError("TubeWire can only have {} inner wires, but {} is added.".format(self.phase_num,
                                                                                              len(self.wires) + 1))
        self.wires.append(wire)

    def get_r(self):
        """
        返回芯线集合的半径矩阵。

        返回:
        radii (numpy.narray, n*1): n条芯线的半径矩阵,每行为某一条芯线的半径
        """
        r = np.zeros((len(self.wires), 1))
        for i, wire in enumerate(self.wires):
            r[i] = wire.r
        return r

    def get_end_node_y(self):
        """
        返回芯线末端y值矩阵。

        返回:
        end_node_y (numpy.narray, n*1): n条芯线的末端y值
        """
        end_node_y = np.zeros((len(self.wires), 1))
        for i, wire in enumerate(self.wires):
            end_node_y[i] = wire.end_node.y
        return end_node_y

    def get_sig(self):
        """
        返回芯线电导率矩阵。

        返回:
        sig (numpy.narray, n*1): n条芯线的电导率
        """
        sig = np.zeros((len(self.wires), 1))
        for i, wire in enumerate(self.wires):
            sig[i] = wire.sig
        return sig

    def get_mur(self):
        """
        返回芯线磁导率。

        返回:
        mur (numpy.narray, n*1): n条芯线的磁导率
        """
        mur = np.zeros((len(self.wires), 1))
        for i, wire in enumerate(self.wires):
            mur[i] = wire.mur
        return mur

    def get_epr(self):
        """
        返回芯线相对介电常数。

        返回:
        epr (numpy.narray, n*1): n条芯线的相对介电常数
        """
        epr = np.zeros((len(self.wires), 1))
        for i, wire in enumerate(self.wires):
            epr[i] = wire.epr
        return epr

    def get_offset(self):
        """
        返回芯线偏置矩阵。

        返回:
        offset (numpy.narray, n*1): n条线的偏置
        """
        offset = np.zeros((len(self.wires), 1))
        for i, wire in enumerate(self.wires):
            offset[i] = wire.offset
        return offset

    def get_height(self):
        """
        返回芯线高度矩阵。

        返回:
        height (numpy.narray, n*1): n条线的高度
        """
        height = np.zeros((len(self.wires), 1))
        for i, wire in enumerate(self.wires):
            height[i] = wire.height
        return height

    def get_L(self):
        """
        返回线电感。

        返回:
        L (numpy.narray, n*1): n条芯线的电感
        """
        L = np.zeros((len(self.wires), 1))
        for i, wire in enumerate(self.wires):
            L[i] = wire.L
        return L

    def get_R(self):
        """
        返回架空线电阻。

        返回:
        R (numpy.narray, n*1): n条芯线的相对介电常数
        """
        R = np.zeros((len(self.wires), 1))
        for i, wire in enumerate(self.wires):
            R[i] = wire.R
        return R



if __name__ == '__main__':
    import pandas as pd

    file_name = 'C:\\Users\\demo\\Desktop\\PolyU\\电路参数生成矩阵\\Tower_V9h\\Tower_V9h\\DATA_InputFile_P2\\Input_Span1.xlsx'
    data = pd.read_excel(file_name, index_col=None, header=None)
    x1 = data.iloc[12:16, 5].to_numpy(dtype='float')
    y1 = data.iloc[12:16, 6].to_numpy(dtype='float')
    z1 = data.iloc[12:16, 7].to_numpy(dtype='float')
    x2 = data.iloc[12:16, 8].to_numpy(dtype='float')
    y2 = data.iloc[12:16, 9].to_numpy(dtype='float')
    z2 = data.iloc[12:16, 10].to_numpy(dtype='float')
    start_node_1 = Node('start_node_1', x1[0], y1[0], z1[0])
    end_node_1 = Node('end_node_1', x2[0], y2[0], z2[0])
    start_node_2 = Node('start_node_2', x1[1], y1[1], z1[1])
    end_node_2 = Node('end_node_2', x2[1], y2[1], z2[1])
    start_node_3 = Node('start_node_3', x1[2], y1[2], z1[2])
    end_node_3 = Node('end_node_3', x2[2], y2[2], z2[2])

    offset = data.iloc[12:16, 11].to_numpy(dtype='float')
    r = data.iloc[12:16, 12].to_numpy(dtype='float')
    R = data.iloc[12:16, 13].to_numpy(dtype='float')
    l = data.iloc[12:16, 14].to_numpy(dtype='float')
    sig = data.iloc[12:16, 15].to_numpy(dtype='float')
    mur = data.iloc[12:16, 16].to_numpy(dtype='float')
    epr = data.iloc[12:16, 17].to_numpy(dtype='float')

    wires_r_1 = r[0]
    wires_offset_1 = offset[0]
    wires_sig_1 = sig[0]
    wires_mur_1 = mur[0]
    wires_epr_1 = epr[0]
    wires_R_1 = R[0]
    wires_l_1 = l[0]

    wires_r_2 = r[1]
    wires_offset_2 = offset[1]
    wires_sig_2 = sig[1]
    wires_mur_2 = mur[1]
    wires_epr_2 = epr[1]
    wires_R_2 = R[1]
    wires_l_2 = l[1]

    wires_r_3 = r[2]
    wires_offset_3 = offset[2]
    wires_sig_3 = sig[2]
    wires_mur_3 = mur[2]
    wires_epr_3 = epr[2]
    wires_R_3 = R[2]
    wires_l_3 = l[2]

    wire1 = Wire('wire_1', start_node_1, end_node_1, wires_offset_1, wires_r_1, wires_R_1, wires_l_1,
                     wires_sig_1, wires_mur_1, wires_epr_1, 0)
    wire2 = Wire('wire_2', start_node_2, end_node_2, wires_offset_2, wires_r_2, wires_R_2, wires_l_2,
                 wires_sig_2, wires_mur_2, wires_epr_2, 0)
    wire3 = Wire('wire_3', start_node_3, end_node_3, wires_offset_3, wires_r_3, wires_R_3, wires_l_3,
                     wires_sig_3, wires_mur_3, wires_epr_3, 0)

    sig_g = data.iloc[10, 5]
    mur_g = data.iloc[10, 6]
    epr_g = data.iloc[10, 7]

    GND = Ground(sig_g, mur_g, epr_g, 0, 0, 0)

    OHL = OHLWire('OHL', 0, 0, 3, GND)
    OHL.add_wire(wire1)
    OHL.add_wire(wire2)
    OHL.add_wire(wire3)



    R1, L1, Z1, C1 = OHL_parameters_calculate(OHL, frq=frq_default)

    print(1)