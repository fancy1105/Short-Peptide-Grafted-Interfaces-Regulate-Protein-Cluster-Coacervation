#  2 files need to be prepared to use cluster_analyse.py
#  1-dump1.200000000.lammpstrj (complete lammps dumping file)
#  2-no_spacer (delete spacer bead using OVITO visualization software)
#  output-cluster_info_full (new lammps dumping file with extra cluster id in every bead info for further analyse)


import numpy as np
import numba as nb

box_x = 600
box_y = 600
box_z = 600

start_dump = 0
final_dump = 200
fus_chain_number = 1000
fus_chain_length = 50
fus_total_sticker_valence = 6
surface_sticker_number = 150
surface_length = 16
surface_sticker_total_valence = 6
bead_info = 6  # id type xyz mol
bead_number1 = fus_chain_number * fus_chain_length
bead_number2 = surface_sticker_number * surface_length
read_lammpstrj_1 = np.zeros(shape=[(final_dump - start_dump) * bead_number1, bead_info])
read_lammpstrj_3 = np.zeros(shape=[(final_dump - start_dump) * fus_chain_number * fus_total_sticker_valence, bead_info])
read_lammpstrj_4 = np.zeros(shape=[(final_dump - start_dump) * surface_sticker_number * surface_sticker_total_valence, bead_info])
dump_file_for_cluster = r"immobile_150\no_spacer"
output_filename1 = r"immobile_150\cluster_info_full"
dump_file = r"immobile_150\dump1.200000000.lammpstrj"
# output_filename2 = "D:/WORK/wetting_dewetting/23-9-5/bulk/36/multimer_info"
for i in range(start_dump, final_dump):
    print(i)
    read_lammpstrj_1[i * bead_number1:(i + 1) * bead_number1, :] = np.loadtxt(
        dump_file,
        skiprows=9 + i * (bead_number1 + bead_number2 + 9) * every_Nth_frame, max_rows=bead_number1, encoding='UTF-8')
    read_lammpstrj_3[
    i * fus_chain_number * fus_total_sticker_valence:(i + 1) * fus_chain_number * fus_total_sticker_valence,
    :] = np.loadtxt(dump_file_for_cluster, skiprows=9 + i * (
            fus_chain_number * fus_total_sticker_valence + surface_sticker_number * surface_sticker_total_valence + 9) * every_Nth_frame,
                    max_rows=fus_chain_number * fus_total_sticker_valence, encoding='UTF-8')

read_lammpstrj_2 = np.zeros(shape=[(final_dump - start_dump) * bead_number2, bead_info])
for i in range(start_dump, final_dump):
    print(i)
    read_lammpstrj_2[i * bead_number2:(i + 1) * bead_number2, :] = np.loadtxt(
        dump_file,
        skiprows=9 + bead_number1 + i * (bead_number1 + bead_number2 + 9) * every_Nth_frame, max_rows=bead_number2,
        encoding='UTF-8')
    read_lammpstrj_4[i * surface_sticker_number * surface_sticker_total_valence:(
                                                                                        i + 1) * surface_sticker_number * surface_sticker_total_valence,
    :] = np.loadtxt(
        dump_file_for_cluster,
        skiprows=9 + fus_chain_number * fus_total_sticker_valence + i * (
                fus_chain_number * fus_total_sticker_valence + surface_sticker_number * surface_sticker_total_valence + 9) * every_Nth_frame,
        max_rows=surface_sticker_number * surface_sticker_total_valence,
        encoding='UTF-8')


class Quick_Find():
    def __init__(self, n):
        self.count = n
        self._parent = np.array([i for i in range(n)])
        self._weight = np.array([1 for i in range(n)])

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP == rootQ:
            return
        # 轻根到重根，为了平衡,把轻根的根节点放到重根下
        if self._weight[rootP] > self._weight[rootQ]:
            self._parent[rootQ] = rootP
            self._weight[rootP] += self._weight[rootQ]
        else:
            self._parent[rootP] = rootQ
            self._weight[rootQ] += self._weight[rootP]
        self.count -= 1

    def is_connected(self, p, q):
        return self.find(p) == self.find(q)

    def find(self, x):
        while self._parent[x] != x:
            # 路径压缩
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def get_count(self):
        return self.count

    def hash_dict(self):
        hash1 = {}  # 定义一个字典
        for i in range(len(self._parent)):
            parent = self.find(i)  # 查找每个粒子的根节点
            if parent in hash1:
                hash1[parent].append(i)
            else:
                hash1[parent] = [i]
        sort_id = []
        keys = []
        for j in hash1:
            sort_id.append(j)
            keys.append(len(hash1[j]))
        keys_sort = np.argsort(np.array(keys))  # np.argsort返回的是元素值从小到大排序后的索引值的数组
        result = [hash1[sort_id[x]] for x in keys_sort]
        result.reverse()
        return result


@nb.jit()
def ppp_distance(bead1, bead2):
    if bead1[3] - bead2[3] > box_x / 2:
        dx = box_x - (bead1[3] - bead2[3])
    elif bead1[3] - bead2[3] < -(box_x / 2):
        dx = bead1[3] - bead2[3] + box_x
    else:
        dx = bead1[3] - bead2[3]
    if bead1[4] - bead2[4] > box_y / 2:
        dy = box_y - (bead1[4] - bead2[4])
    elif bead1[4] - bead2[4] < -(box_y / 2):
        dy = bead1[4] - bead2[4] + box_y
    else:
        dy = bead1[4] - bead2[4]
    if bead1[5] - bead2[5] > box_z / 2:
        dz = box_z - (bead1[5] - bead2[5])
    elif bead1[5] - bead2[5] < -(box_z / 2):
        dz = bead1[5] - bead2[5] + box_z
    else:
        dz = bead1[5] - bead2[5]
    ds = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
    return ds


@nb.jit()
def ppf_distance(bead1, bead2):
    if bead1[3] - bead2[3] > box_x / 2:
        dx = box_x - (bead1[3] - bead2[3])
    elif bead1[3] - bead2[3] < -(box_x / 2):
        dx = bead1[3] - bead2[3] + box_x
    else:
        dx = bead1[3] - bead2[3]
    if bead1[4] - bead2[4] > box_y / 2:
        dy = box_y - (bead1[4] - bead2[4])
    elif bead1[4] - bead2[4] < -(box_y / 2):
        dy = bead1[4] - bead2[4] + box_y
    else:
        dy = bead1[4] - bead2[4]
    dz = bead1[5] - bead2[5]
    ds = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
    return ds


@nb.jit()
def check_binding_ab(chain1, chain2):
    temp_check = 0
    for i in range(0, len(chain1)):
        for j in range(0, len(chain2)):
            if chain1[i, 2] == 1 and chain2[j, 2] == 2:
                if ppf_distance(chain1[i], chain2[j]) < 2:
                    temp_check = temp_check + 1

    if temp_check == 0:
        return 0  # 表示两条链之间没有成键
    elif temp_check != 0:
        return 1  # 表示两条链之间有成键


@nb.jit()
def check_binding_aB(chain1, chain2):
    temp_check = 0
    for i in range(0, len(chain1)):
        for j in range(0, len(chain2)):
            if chain1[i, 2] == 1 and chain2[j, 2] == 4:
                if ppf_distance(chain1[i], chain2[j]) < 2:
                    temp_check = temp_check + 1

    if temp_check == 0:
        return 0  # 表示两条链之间没有成键
    elif temp_check != 0:
        return 1  # 表示两条链之间有成键


@nb.jit()
def binding_matrix_ab(type_a_chain_list, type_b_chain_list):  # 统计每帧中，a链和b链的交联情况，有交联则相应矩阵元为1，无则为0
    temp_chain_binding_matrix = np.zeros(shape=[fus_chain_number, fus_chain_number])
    for i in range(0, fus_chain_number):
        for j in range(0, fus_chain_number):
            if check_binding_ab(type_a_chain_list[fus_total_sticker_valence * i:fus_total_sticker_valence * (i + 1)],
                                type_b_chain_list[
                                fus_total_sticker_valence * j:fus_total_sticker_valence * (j + 1)]) == 1:
                temp_chain_binding_matrix[i, j] = 1

    return temp_chain_binding_matrix


@nb.jit()
def binding_matrix_aB(type_a_chain_list, type_b_chain_list):  # 统计每帧中，a链和b链的交联情况，有交联则相应矩阵元为1，无则为0
    temp_chain_binding_matrix = np.zeros(shape=[fus_chain_number, surface_sticker_number])
    for i in range(0, fus_chain_number):
        for j in range(0, surface_sticker_number):
            if check_binding_aB(type_a_chain_list[fus_total_sticker_valence * i:fus_total_sticker_valence * (i + 1)],
                                type_b_chain_list[
                                (surface_sticker_total_valence * j):(surface_sticker_total_valence * (j + 1))]) == 1:
                temp_chain_binding_matrix[i, j] = 1

    return temp_chain_binding_matrix


# a = binding_matrix(sticker_a[10], sticker_b[10])


# print(binding_matrix(sticker_a[10], sticker_b[10]))

@nb.jit()
def find_cluster(chain_binding_matrix_ab,
                 chain_binding_matrix_ac):  # 输出每个cluster所包含的链id的信息(注意这里的链id因为列表索引的原因都是减1的,真正的链id要在输出值上加1),并按cluster从大到小排列
    quickfind = Quick_Find(fus_chain_number + surface_sticker_number)

    for i in range(0, fus_chain_number):
        for j in range(0, fus_chain_number):
            if chain_binding_matrix_ab[i, j] != 0:
                quickfind.union(i, j)
        for k in range(0, surface_sticker_number):
            if chain_binding_matrix_ac[i, k] != 0:
                quickfind.union(i, k + fus_chain_number)

    cluster_list = quickfind.hash_dict()

    # del quickfind
    return cluster_list



# print(b[0])
# @nb.jit()
def output_lammpstrj_cluster(dump11, cluster_list):  # 添加某一帧的cluster信息在轨迹文件的最后一列
    print('output', dump11)
    temp_new_lammpstrj = np.zeros(shape=[bead_number1 + bead_number2, bead_info + 1])
    temp_new_lammpstrj[:bead_number1, :bead_info] = read_lammpstrj_1[dump11 * bead_number1:(dump11 + 1) * bead_number1]
    temp_new_lammpstrj[bead_number1:bead_number1 + bead_number2, :bead_info] = read_lammpstrj_2[dump11 * bead_number2:(
                                                                                                                              dump11 + 1) * bead_number2]
    for i in range(0, len(cluster_list)):
        for j in range(0, len(cluster_list[i])):
            for k in range(0, fus_chain_number):
                if temp_new_lammpstrj[k * fus_chain_length, 1] == cluster_list[i][j] + 1:
                    for i1 in range(0, fus_chain_length):
                        temp_new_lammpstrj[k * fus_chain_length + i1, 6] = i + 1
            for m in range(0, surface_sticker_number):
                if temp_new_lammpstrj[fus_chain_number * fus_chain_length + m * surface_length, 1] == cluster_list[i][
                    j] + 1:
                    for i2 in range(0, surface_length):
                        temp_new_lammpstrj[
                            fus_chain_number * fus_chain_length + m * surface_length + i2, 6] = (i + 1)

    return temp_new_lammpstrj


# c = output_lammpstrj_cluster(10, b)
# print(output_lammpstrj_cluster(10, b))

new_lammpstrj = np.zeros(shape=[final_dump - start_dump, bead_number1 + bead_number2, bead_info + 1])
for i in range(0, final_dump - start_dump):
    # print(i)
    new_lammpstrj[i] = output_lammpstrj_cluster(i,
                                                find_cluster(
                                                    binding_matrix_ab(
                                                        read_lammpstrj_3[
                                                        i * fus_chain_number * fus_total_sticker_valence:(
                                                                                                                 i + 1) * fus_chain_number * fus_total_sticker_valence],
                                                        read_lammpstrj_3[
                                                        i * fus_chain_number * fus_total_sticker_valence:(
                                                                                                                 i + 1) * fus_chain_number * fus_total_sticker_valence]),
                                                    binding_matrix_aB(
                                                        read_lammpstrj_3[
                                                        i * fus_chain_number * fus_total_sticker_valence:(
                                                                                                                 i + 1) * fus_chain_number * fus_total_sticker_valence],
                                                        read_lammpstrj_4[
                                                        i * surface_sticker_number * surface_sticker_total_valence:(
                                                                                                                           i + 1) * surface_sticker_number * surface_sticker_total_valence])))

f = open(f'{output_filename1}', 'w',
         encoding='UTF-8')  # 存在该目录下
for i in range(0, final_dump - start_dump):
    # print(i)
    f.write(
        f'ITEM: TIMESTEP\n{int(i * 1)}\nITEM: NUMBER OF ATOMS\n{int(bead_number1 + bead_number2)}\nITEM: BOX BOUNDS pp pp pp\n0.0000000000000000e+00 4.0000000000000000e+02\n0.0000000000000000e+00 4.0000000000000000e+02\n0.0000000000000000e+00 4.1000000000000000e+02\nITEM: ATOMS id mol type x y z c\n')
    for j in range(0, bead_number1 + bead_number2):
        f.write(
            f'{int(new_lammpstrj[i, j, 0])} {int(new_lammpstrj[i, j, 1])} {int(new_lammpstrj[i, j, 2])} {new_lammpstrj[i, j, 3]} {new_lammpstrj[i, j, 4]} {new_lammpstrj[i, j, 5]} {new_lammpstrj[i, j, 6]}\n')
