import numpy as np
import time

class VanEmdeBoasTree: #van Emde Boas木の初期化

    def __init__(self, valueRange): #領域サイズを設定し、木を初期化する

        bitSize = np.ceil(np.log2(valueRange)) #ビットサイズ
        self.valueRange =  1 << int(bitSize)
        self.min = None #領域内の最小値
        self.max = None #領域内の最大値
        if self.valueRange > 2: #領域の最小サイズ
            halfBitSize = bitSize / 2

            self.summary = None
            self.cluster = {}
            self.lsbSize = int(np.floor(halfBitSize))
            self.lsqrt = 1 << self.lsbSize
            self.hsqrt = 1 << int(np.ceil(halfBitSize))
            self.lsbMask = self.lsqrt - 1

    def clusterIndex(self, x): #クラスターのインデックスを返す

        return x >> self.lsbSize 

    def valueIndex(self, x): #xを格納するクラスターを返す

        return x & self.lsbMask

    def value(self, clusterIndex, valueIndex): #インデックスとクラスターに関連する値を返す

        return (clusterIndex << self.lsbSize) + valueIndex

    def contains(self, x): #xが木に格納されているかをチェック

        if self.min == x or self.max == x:
            return True

        if self.min is None or x < self.min or x > self.max or self.valueRange == 2:
            return False

        xCluster = self.cluster.get(self.clusterIndex(x), None)
        return (xCluster is not None) and xCluster.contains(self.valueIndex(x))

    def predecessor(self, x): #predecessor関数

        if self.valueRange == 2:
            return 0 if (x == 1 and self.min == 0) else None

        if self.max is not None and x > self.max:
            return self.max

        xClusterIndex = self.clusterIndex(x)
        xCluster = self.cluster.get(xClusterIndex, None)
        xIndex = self.valueIndex(x)

        if xCluster is not None and xCluster.min < xIndex:
            return self.value(xClusterIndex, xCluster.predecessor(xIndex))

        if self.summary is not None:
            predClusterIndex = self.summary.predecessor(xClusterIndex)
            if predClusterIndex is not None:
                return self.value(predClusterIndex, self.cluster[predClusterIndex].max)

        return self.min if (self.min is not None and x > self.min) else None

    def successor(self, x): #successor関数

        if self.valueRange == 2:
            return 1 if (x == 0 and self.max == 1) else None

        if self.min is not None and x < self.min:
            return self.min

        xClusterIndex = self.clusterIndex(x)
        xCluster = self.cluster.get(xClusterIndex, None)
        xIndex = self.valueIndex(x)

        if xCluster is not None and xCluster.max > xIndex:
            return self.value(xClusterIndex, xCluster.successor(xIndex))

        if self.summary is not None:
            succClusterIndex = self.summary.successor(xClusterIndex)
            if succClusterIndex is not None:
                return self.value(succClusterIndex, self.cluster[succClusterIndex].min)

        return self.max if (self.max is not None and x < self.max) else None

    
    def insert(self, x): #insert関数

        if x == self.min or x == self.max:
            return

        if self.min is None:
            self.min = self.max = x
            return

        if self.min == self.max:
            if x < self.min:
                self.min = x
            elif x > self.max:
                self.max = x
            return

        if self.valueRange == 2:
            return

        if x < self.min:
            self.min, x = x, self.min

        elif x > self.max:
            self.max, x = x, self.max

        xClusterIndex = self.clusterIndex(x)
        xCluster = self.cluster.get(xClusterIndex, None)
        xIndex = self.valueIndex(x)

        if xCluster is None:
            if self.summary is None:
                self.summary = VanEmdeBoasTree(self.hsqrt)
            self.summary.insert(xClusterIndex)

            xCluster = VanEmdeBoasTree(self.lsqrt)
            self.cluster[xClusterIndex] = xCluster

        xCluster.insert(xIndex)

    
    def delete(self, x): #削除関数
        """
        Deletes x from the veb-tree.
        """
        if self.min is None or x < self.min or x > self.max:
            return

        if self.min == self.max:
            self.min = self.max = None
            return

        if self.valueRange == 2:
            if x == 0:
                self.min = 1
            else:
                self.max = 0
            return

        if x == self.min:
            if self.summary is None:
                self.min = self.max
                return

            xClusterIndex = self.summary.min
            xCluster = self.cluster[xClusterIndex]
            xIndex = xCluster.min

            self.min = self.value(xClusterIndex, xIndex)

        elif x == self.max:
            if self.summary is None:
                self.max = self.min
                return

            xClusterIndex = self.summary.max
            xCluster = self.cluster[xClusterIndex]
            xIndex = xCluster.max

            self.max = self.value(xClusterIndex, xIndex)

        else:
            xClusterIndex = self.clusterIndex(x)
            xCluster = self.cluster.get(xClusterIndex, None)
            xIndex = self.valueIndex(x)
            if xCluster is None:
                return

        xCluster.delete(xIndex)

        if xCluster.min is None:
            del self.cluster[xClusterIndex]
            self.summary.delete(xClusterIndex)

            if self.summary.min is None:
                self.summary = None

    
    def __iter__(self): #要素を越えて参照

        value = self.min
        while value is not None:
            yield value
            value = self.successor(value)

    
    def __str__(self): #タプル型のデータを返す

        return str(tuple(value for value in self))

    
    def __repr__(self): #木構造全体を表示

        return str(self)


def LIS(π): #LIS探索の心臓部（LISの長さを返す）
    n = len(π)
    B = VanEmdeBoasTree(n)
    k = 0
    for i in range(n):
        x = π[i][2]
        B.insert(x)
        succ = B.successor(x)
        if succ is not None:
            B.delete(succ)
        else:
            k += 1
    return k

def find_LIS_subsequence(π):
    n = len(π)
    B = VanEmdeBoasTree(n)
    predecessors = [-1] * n
    lengths = [0] * n
    best_elements = [-1] * (n + 1)

    for i in range(n):
        x = π[i][2]
        succ = B.successor(x)
        pred = B.predecessor(x)

        # If there is a predecessor, update the length and predecessor
        if pred is not None:
            pred_index = next(idx for idx, val in enumerate(π) if val[2] == pred)
            lengths[i] = lengths[pred_index] + 1
            predecessors[i] = pred_index
        else:
            lengths[i] = 1

        # Insert the current element into the VEB tree
        B.insert(x)

        # Update the best elements list
        if succ is not None:
            B.delete(succ)
            best_elements[lengths[i]] = i
        else:
            best_elements[lengths[i]] = i

    # Reconstruct the LIS based on best_elements and predecessors
    lis_length = max(lengths)
    lis_subsequence = []
    current_index = best_elements[lis_length]

    while current_index != -1:
        lis_subsequence.append(π[current_index])
        current_index = predecessors[current_index]

    lis_subsequence.reverse()
    return lis_subsequence


def process_file_and_number(input_file): #リンゴのナンバリング（これが遅いので改良の余地あり）
    mtrx = []

    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    for d, line in enumerate(lines):
        e, f, g = map(int, line.strip().split(','))
        h = g + 5 * f
        mtrx.append([e, h, None, None])

    # Numbering for y = x - K (decreasing K)
    for d in range(len(mtrx)):
        e, h, _, _ = mtrx[d]
        mtrx[d][2] = h - e
        
    mtrx.sort(key=lambda x: (x[2], x[1]))
    
    for d in range(len(mtrx)):
        mtrx[d][2] = d+1
        

    # Numbering for y = -x + L (increasing L)
    for d in range(len(mtrx)):
        e, h, _, _ = mtrx[d]
        mtrx[d][3] = h + e
        
    mtrx.sort(key=lambda x: (x[3], x[1]))
    
    for d in range(len(mtrx)):
        mtrx[d][3] = d+1

    return find_LIS_subsequence(mtrx)

def route(list_1,list_2):
    x_1 = list_1[0]
    y_1 = list_1[1]
        
    x_2 = list_2[0]
    y_2 = list_2[1]
    
    if x_2 > x_1:
        print("r",end=',')
        list_1[0]+=1
        list_1[1]+=1
        route(list_1,list_2)

    elif x_2 < x_1:
        print("l",end=',')
        list_1[0]-=1
        list_1[1]+=1
        route(list_1,list_2)
        
    elif y_2 > y_1:
        print("s",end=',')
        list_1[1]+=1
        route(list_1,list_2)
    

def route_func(num_list):
    n= len(num_list)
    print(longest_subsequence[0][0],end='')
    for i in range(1, n):
        list_1=num_list[i-1]
        list_2=num_list[i]
        route(list_1,list_2)


# Usage example
input_file = 'input8.txt'

start_time = time.time()

longest_subsequence = process_file_and_number(input_file)

end_time = time.time()


start = [longest_subsequence[0][0], 0, 0, 0]

longest_subsequence.insert(0,start)
    
route_func(longest_subsequence)



print(f"\n実行時間: {end_time - start_time}秒")