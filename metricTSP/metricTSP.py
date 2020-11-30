

class metricTSP:

    def __init__(self, Graph:list):
        self.G = Graph
        print("输出图G")
        print(self.G)
        self.T = []
        self.Gt = self.copyM()  # T对应的图
        self.odds = set()  # 奇数度顶点集 odds  集合｛｝  元素不可重复
        self.H = self.copyM()  # 复制G的结构，基于G构建子图H   奇数度顶点的子图
        self.Mh = []        # 最小费用完美匹配   matching
        self.eulerGraph = []
        self.etour = []
        self.cost = 0

    def isMetric(self):
        """
        metric     一般要求为非负 无向完全图  故求解之前需要测试下是否符合metric的定义
        （1）如果i = j，则w（i，j）= 0，所以对角线为0     无边
        （2）w（i，j）= w（j，i），所以它是对称的->无向，视为无向图
        （3）w（i，j）<= w（i，k）+ w（k，j）  满足  三角形不等式  metric定义
        """
        n = len(self.G)  # 矩阵M行数
        # print(n)
        for i in range(0, n):  # range(0,n)  相当于访问[0,n)
            m = len(self.G[i])  # M[i]这一行的列数
            # print(m)

            if n != m:  # 如果行数和列数不相等，那么不完全 直接返回
                return False
            else:
                for j in range(0, m):
                    # 这里判断三个条件
                    for k in range(0, n):  # k -> [0,n]
                        if i == j and self.G[i][j] != 0:  # 对角线元素不为0
                            return False
                        elif self.G[i][j] != self.G[j][i]:  # 不对称 不是无向图
                            return False
                        else:
                            if self.G[i][j] > self.G[i][k] + self.G[k][j]:  # 不满足三角不等式
                                return False

        # 若metric的三个条件都满足，中途不会提前返回（False），则验证通过，返回True
        return True

    def MST(self):
        """
        最小生成树 Prim算法
        G：图的矩阵 list 数组形式
        """
        # print("G in line 57 is {}".format(G))
        # T = []  # 树边 列表
        V = set()  # set 集合  元素无序  唯一 不可更改   这里看重其   不可重复性
        for i in range(0, len(self.G)):
            V.add(i)  # 顶点值直接用数字标志，当然可以用字符来表示某个顶点，不过比较麻烦
        u = 0  # 第一个选择的顶点为0
        W = {u}  # 加入第一个顶点   W是一个集合

        while W != V:  # 直到所有的节点都加入这个树
            tmpSet = V.difference(W)  # 将两个集合的差返回，用来判断两个集合是否相同
            vMin = 999999999  # 定义一个极大值以来最小值，基操
            a = 0  # a  i
            b = 0  # b  j
            for i in W:  # 已经入树的节点 i
                for j in tmpSet:  # 还未选中的节点j
                    if self.G[i][j] < vMin:
                        a = i
                        b = j
                        vMin = self.G[i][j]  # 更新最小权重  prim算法 选择可选的最小权重边的节点

            W.add(b)  # b接收的就是未加入节点中可加入的新节点
            self.T.append((a, b))  # 加入树边（a,b）

        #print("MST:",self.T)

    def copyM(self):
        """
            copy矩阵的函数
        """
        M = []
        for x in self.G:
            M.append(x.copy())
        return M

    def convertT2G(self):
        """
        代表树T的G的子图   树T 是 图G 的子图
        G:图 矩阵形式 边权
        T:生成树的边
        函数根据图G返回T的矩阵形式
        """
        # Gt = self.copyM(self.G)  # 不对G进行操作，故需要一个copy函数，构建一个tmp矩阵
        n = len(self.Gt)  # 行数 n
        noArc = -1  # 用 -1表示节点间无边相连

        for i in range(0, n):  # 初始化
            for j in range(0, n):  # 矩阵为方阵
                if i != j:
                    self.Gt[i][j] = noArc
                else:
                    self.Gt[i][j] = 0

        for edge in self.T:  # 这里开始根据生成树T构建矩阵
            u = edge[0]
            v = edge[1]
            self.Gt[u][v] = self.G[u][v]
            self.Gt[v][u] = self.G[v][u]
        # return Gt  # 树T对应的矩阵图

    def getOddsOfT(self):
        """
        奇数度顶点列表
        """
        odds1 = set()  # 奇数odds  集合｛｝  元素不可重复
        n = len(self.Gt)
        print("n:",n)
        for i in range(0, n):
            verTOdd = 0  # 度数
            for j in range(0, n):
                if self.Gt[i][j] != -1 and (i != j):  # i j有边 且不是对角线元素，则可以计算度数
                    verTOdd += 1  # 得到i的出度   当然这是无向图，计算出度即可
            if verTOdd % 2 != 0:  # 为奇数，则加入集合
                odds1.add(i)
        #print("odds1",odds1)
        self.odds = odds1

    def getSubH(self):
        """
        构造由图上的奇数度顶点构成的子图，返回子图的列表
        """
        vertex = set()
        n = len(self.G)
        for i in range(0,n):
            vertex.add(i)    # 顶点集合，数字

        others = vertex.difference(self.odds)       # 得到除去奇数度顶点的那些顶点集合
        # print("others in line 140 ",others)
        for v in others:
            for i in range(0,n):
                if i != v:          # 非对角线元素   初始化
                    self.H[i][v] = -1
            for j in range(0,n):
                if j != v:
                    self.H[v][j] = -1

        # return self.H


    # minimumWeightedPerfectMatching   最小费用完美匹配
    def mWPMn(self):
        """
         最小生成树T上的奇数顶点集H具有以下重要性质
         1）来自metric空间
         2）它与图G的最小树T有关
         3）树的奇数度的顶点数为偶数
         4）可以计算完美匹配M
         5）H是由奇数度节点导出的G的子图
         6）因为G是完整的，所以H保留了其顶点之间的连接
         在H上计算最小代价完美匹配M，把M加到T上得到欧拉图
        """
        cobert = set()
        n = len(self.H)
        Mh1 = []

        while cobert != self.odds:
            for u in self.odds:
                vMin = 999999999
                a = 0  # 记录最小值的边（u,v）->(a,b)
                b = 0
                if u not in cobert:
                    for v in range(0, n):
                        if self.H[u][v] != 0 and self.H[u][v] != -1:  # 边权有效
                            if self.H[u][v] < vMin and v not in cobert:
                                vMin = self.H[u][v]
                                a = u  # 计算H中的最小值，并记录边的情况
                                b = v
                    Mh1.append((a, b))  # 将边加入
                    cobert.add(a)
                    cobert.add(b)  # 更新元素，直到访问所有odds节点

        #print("Mh1",Mh1)
        self.Mh = Mh1
        # return Mh  # 返回最小费用完美匹配   matching


    # 得到欧拉图 需要相加操作
    def calEulerGraph(self):
        """
        将最小费用完美匹配Mh+T，得到欧拉图
        """
        self.eulerGraph = self.T + self.Mh
        # return D


    # 计算欧拉图上的欧拉环游
    def eulerTour(self):
        """
        G 矩阵matrix
        D是可能的多重图的顶点列表，它是元组列表
        """
        A = self.eulerGraph.copy()
        Av = self.list_Avalible(A)  # matix  将边与一个boolean值对象，标志边是否被访问过

        e0 = self.eulerGraph[0]  # 第一个边
        u0 = e0[0]  # 第一个顶点
        self.etour = []

        self.find_tour(u0, Av)  # 求欧拉环游


    def list_Avalible(self, A: list):
        Av = []
        for edge in A:
            e = []
            e.append(edge)
            e.append(True)  # 将边与一个boolean值对应，构建而二元组而已，表示这个边是否被访问过
            Av.append(e)

        return Av


    # 邻接表
    def list_vertexA(self, v: int, E: list):
        """
        E 矩阵
        E的对象是另一个list，E[i]结构为[(u,v),Bool]   边是否被访问过的意思

        ek = E[k] 具有布尔值的一条边
        ek 是一个布尔元组列表
        ek[1] 访问布尔值
        edge = ek[0] 访问（u,v）    edge[0]   edge[1]  v
        """

        n = len(E)
        A2 = []
        for i in range(0, n):  # E[i]结构为[(u,v),Bool]
            ek = E[i]
            edge = ek[0]
            a = edge[0]
            b = edge[1]
            if a == v:
                edgeIx = (a, b, i)  # 索引
                A2.append(edgeIx)

            if b == v:
                edgeIx = (b, a, i)
                A2.append(edgeIx)

        return A2


    # cite : https://algorithmist.com/wiki/Euler_tour
    def find_tour(self, u: int, A: list):
        """
        u 开始的节点
        A  邻接表
        tour 保存欧拉环游的结构，列表
        若C是欧拉图的一个循环，则去除C的边后，则子图其连通分量也是欧拉图  性质
        """
        A2 = self.list_vertexA(u, A)  # 构建邻接表结构
        for edge in A2:  # edgeIx = (a,b,i)   A2中元素的结构
            # a = superEdge[0]       #u
            b = edge[1]  # v  边的终点 下一个访问点
            k = edge[2]  # index

            if A[k][1]:
                A[k][1] = False  # 访问过了，不可再访问  [(u,v),Boolean]
                self.find_tour(b, A)  # 递归，从下一个访问点继续访问
        self.etour.append(u)

        # return tour


    def shortCuts(self):
        """
        根据W返回一个列表，保留order次序，其中没有重复元素
        """
        P = []
        for i in self.etour:
            if i not in P:
                P.append(i)

        self.C = P


    def getTourWeight(self):
        """
        P G中环游的list  是顶点的list
        reurn 环游的代价  G 代价矩阵 图矩阵
        """
        n = len(self.C)
        for i in range(0, n):
            if (i + 1) != n:
                u = self.C[i]
                v = self.C[i + 1]
                self.cost += self.G[u][v]

        a = self.C[0]
        z = self.C[-1]
        self.cost += self.G[z][a]

        # return sum

    def getG(self):
        return self.G

    def getMST(self):
        return self.T
    def getGt(self):
        return self.Gt
    def getOdds(self):
        return self.odds
    def getH(self):
        return self.H
    def getPerfectMatch(self):
        return self.Mh
    def getEulerGraph(self):
        return self.eulerGraph
    def getEulerTour(self):
        return self.etour
    def getPathC(self):
        return self.C
    def getMinCost(self):
        return self.cost

def displayM(M:list):
    """
    print matrix M
    """
    for i in M:
        print(i)


'''
下面几个是应对自输入矩阵的构造的函数，用于构造图矩阵
'''
def constructM():
    """
    输入并构建一个完整的方阵，只需要上对角线的值
    """
    n = constructMpt2()

    X = []
    for i in range(0, n):
        X.append([])  # 构建二维列表（二维数组-矩阵）

    # 填充矩阵，初始化
    for i in range(0, n):
        for j in range(0, n):
            X[i].append(-1)

    X = constructMpt3(X, n)

    return finalConf(X)


def constructMpt2(x=-1):
    """
    函数返回有效的n
    n 默认为整数 这是维度
    """
    while True:
        try:
            if x == -1:  # -1第一次调用确定维度
                n = int(input('输入矩阵的尺寸n: '))
                if n > 1:
                    return n
                else:
                    raise ValueError
            else:  # 继续调用输入元素
                z = int(input())
                if z > 0:
                    return z
                else:
                    raise ValueError
        except ValueError:
            print('无效值，重新输入')


def constructMpt3(X: list, n: int) -> list:
    """
    填充对角线，值为0    对称化处理
    """

    for i in range(0, n):
        for j in range(0, n):
            if i == j:  # 对角线元素处理
                X[i][j] = (0)
            elif i < j:
                print('输出值 A[' + str(i) + '][' + str(j) + ']:', end=' ')
                w = constructMpt2(1)
                X[i][j] = w
                X[j][i] = w  # 对称处理
    return X


def finalConf(X: list):
    """
     最终确认矩阵，可以选择是否使用这个或是构建新的
    """
    print('这是输入而确定的矩阵，是否继续？ [Y/N]')
    displayM(X)
    while not False:
        op = input('输入：')
        if op.upper() == 'Y':
            return X
        elif op.upper() == 'N':
            print('输入\'N\',构建新的矩阵')
            constructM()  # 继续构建新的矩阵
        else:
            print('输入无效，继续输入')

def run(mtsp):
    #https://en.wikipedia.org/wiki/Christofides_algorithm
    print('Problema Metric TSP Solver with Christofides algorithm, whose factor is 3/2')
    try:
        if mtsp.isMetric():         # 首先需要图是metric的
            print('-1表示弧不存在，或者正在创建的矩阵的顶点之间没有连接')
            print('这里，图G是metric的，显示如下')
            displayM(mtsp.G)

            print('下面求解图G的最小生成树T')
            mtsp.MST()
            mst = mtsp.getMST()
            print(mst)

            print('T的矩阵形式为')
            mtsp.convertT2G()
            gt = mtsp.getGt()
            displayM(gt)

            print('T的奇数度的顶点集')
            mtsp.getOddsOfT()
            odds = mtsp.getOdds()
            print(odds)

            print('由这些奇数度顶点得到的子图H')
            mtsp.getSubH()
            H = mtsp.getH()
            displayM(mtsp.H)

            print('Mh, H的最小费用完美匹配')
            mtsp.mWPMn()
            M = mtsp.getPerfectMatch()
            print(M)

            print('组合图，得到欧拉图 T + M')
            mtsp.calEulerGraph()
            eulerGraph = mtsp.getEulerGraph()
            print(eulerGraph)

            print('欧拉图的一个欧拉环游')
            mtsp.eulerTour()
            eulerTour = mtsp.getEulerTour()
            print(eulerTour)

            print('通过shortcuts,得到旅行商的环游路径')
            print('图G的汉密尔顿循环')
            mtsp.shortCuts()   # 环游顶点顺序
            path = mtsp.getPathC()
            mtsp.getTourWeight()    # 代价，即费用
            cost = mtsp.getMinCost()
            print(path,'cost = ',cost)
        else:
            raise ValueError
    except ValueError:
        print('G不在metric space，无法计算')

if __name__ == "__main__":
    #测试图，提供两个
    G = [
        [0,1,1,2,1],
        [1,0,1,1,2],
        [1,1,0,1,1],
        [2,1,1,0,1],
        [1,2,1,1,0]
        ]

    M = [
        [0, 12,10,9, 12,11,12],
        [12,0, 8, 12,10,11,12],
        [10,8, 0, 11,3, 8, 9],
        [9, 12,11,0, 11,10,12],
        [12,10,3, 11,0,  6,7],
        [11,11,8, 10,6, 0, 9],
        [12,12,9, 12,7, 9, 0]
        ]
    print('选择测试矩阵')
    print('G =')
    displayM(G)
    print()
    print('M =')
    displayM(M)
    print()
    print('选择G or M，或按x进行自输入：')
    while True:
        op = input('choice:')
        if op.upper() == 'G':
            mtsp = metricTSP(G)
            run(mtsp)
            break
        elif op.upper() == 'M':
            mtsp = metricTSP(M)
            run(mtsp)
            break
        elif op.lower() == 'x':
            X = constructM()
            mtsp = metricTSP(X);
            run(mtsp)
            break
        else:
            print('输入非法，重新输入')


