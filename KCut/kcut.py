# Gomory-Hu Tree
from collections import deque
from sys import maxsize as maxint

BLACK = 2
GRAY = 1
WHITE = 0


class GomoryHuTree:
    def __init__(self, graph):
        self.G = graph
        self.V = len(graph)  # 维度
        self.capacity = {}  # 容量
        for i in range(self.V):
            for j in range(self.V):
                self.capacity[i, j] = graph[i][j]  # 容量
        self.color = {}
        self.pred = {}
        self.tree = {}
        self.flow = {}  # 参量
        self.depth = {}

        self.buildTree()

    def bfs(self,s,t,route):
        #print("route is ",route)
        visited = []
        for i in range(0,self.V):
            visited.append(False)
            self.color[i] = WHITE
        #print(visited)
        visited[s] = True

        queue = deque()
        route[s] = -1

        for i in range(0,self.V):
            #print("flow[s,i]=",self.flow[s,i])
            #print(visited)
            #print(self.flow)
            if self.flow[s,i] != 0 and visited[i] == False:
                queue.append(i)
                route[i] = s
                visited[i] = True
        #print(queue)
        while queue:
            #print("queue is ",queue)
            middleVex = queue.popleft()
            self.color[middleVex] = BLACK
            #print("queue is ",queue)
            #print(middleVex==t)
            if middleVex == t:
                #print("middleVex == t return true")
                return True
            else:
                for i in range(0,self.V):
                    if self.flow[middleVex,i] != 0 and visited[i] == False:
                        queue.append(i)
                        route[i] = middleVex
                        visited[i] = True
                        self.color[i] = GRAY

        return False


    def max_flow(self, source, sink):
        '''
         使用Ford Fulkerson算法求源点和汇点之间的最大流/最小割
        :param source:  s
        :param sink:   t
        :return:  maxflow value
        '''
        max_flow1 = 0
        route = []
        self.f = []   # 当前流
        for i in range(0,self.V):
            route.append(0)

        for i in range(0,self.V):
            self.f.append([])
            for j in range(0,self.V):  # 初始化
                self.flow[i, j] = self.capacity[i,j]
                #self.flow[i, j] = 0
                self.f[i].append(0)
        #print(source,sink)
        while self.bfs(source,sink,route):
            min = 9999999999
            tail = sink
            head = route[sink]
            #print(min,tail,head)
            while head != -1:
                #print("head is ",head)
                if self.flow[head,tail] < min:
                    min = self.flow[head,tail]
                    #print("min is ",min)
                tail = head
                head = route[head]

            #print("line 128")
            tail1 = sink
            head1 = route[tail1]

            while head1 != -1:
                #print("line 133 head1=",head1)
                if self.capacity[head1,tail1] != 0:
                    self.f[head1][tail1] += min
                else:
                    self.f[head1][tail1] -= min

                self.flow[head1,tail1] -= min
                self.flow[tail1,head1] += min

                tail1 = head1
                head1 = route[head1]
            for i in range(0, self.V):
                route.append(0)
            #print(route)
        #print(self.f)
        for i in range(0,self.V):
            max_flow1 += self.f[source][i]
        #print("max_flow is",max_flow1)
        # print("max_flow done!")
        # print(self.capacity)
        # print(self.flow)
        # print(self.pred)

        #print("min_cut is ")
        mincut = self.mincut1(source)
        #print("mincut is ",mincut)
        return max_flow1  # 返回最大流的值


    '''
    求出最小割的其中一种方案
    在进行最大流计算完成之后，从源点进行遍历整张图，将遍历到的点标记为 true ；
    最终结束之后，所有标记为 true 的点和没有标记的点之间就是一个割边。

    因为最大流之后，所有通路在满流的那条通道处断掉了，也就是没有办法继续走下去，
    而这条通道一边标记为了 true 另一边没有被标记，那么他们之间就是一个割边了。

    在残量矩阵中，从s做一次搜索，能达到的顶点集合
    '''

    def mincut(self, s, t):
        '''
        求s-t之间的最小割，首先求最大流，然后调用最小割
        :param s:
        :param t:
        :return:
        '''
        # print(s,t)
        #print("s-t maxflow is ",self.max_flow(s, t))
        #print("t-s maxflow is ",self.max_flow(t, s))
        #maxflow = max(self.max_flow(s, t), self.max_flow(t, s))
        maxflow = self.max_flow(s,t)
        print(s, ",", t, "的最大流：", maxflow)
        # for item in self.flow:
        #     print(self.flow[item])
        # 下面从s进行遍历，得到可访问的节点
        # print(self.G)
        #print(self.flow)
        vertex = set()
        n = len(self.G)
        for i in range(0, n):
            vertex.add(i)  # 顶点集合，数字
        # print("vertex is ",vertex)
        #print("s=",s)
        cut = self.mincut1(s)  # 得到可遍历到的
        print("能访问的节点集合", cut)
        # others = vertex.difference(self.odds)       # 得到除去奇数度顶点的那些顶点集合
        others = vertex.difference(cut)
        print("不能遍历到的节点为：", others)
        print("那么(", s, ",", t, ")的割集为")
        mcut = []
        for i in cut:
            for j in others:
                if self.G[i][j] != 0:
                    edge = "(" + str(i) + "," + str(j) + ")"
                    mcut.append(edge)
        print(mcut)
        return mcut
        # cost = 0
        # for item in others:
        #     # print(item)
        #     for i in range(0,self.V):
        #         if self.G[item][i] != 0:
        #             edge = "("+str(item)+","+str(i)+")"
        #             #print(edge)
        #             cost += self.G[item][i]
        #             mcut.append(edge)
        # print(mcut,"cost",cost)
        #print(self.capacity)
        #print(self.flow)
        '''
        {(0, 0): 0, (0, 1): -13, (0, 2): 0, (0, 3): 0, (0, 4): 0, (0, 5): 0, 
        (1, 0): 13, (1, 1): 0, (1, 2): 0, (1, 3): 0, (1, 4): 0, (1, 5): -13, 
        (2, 0): 0, (2, 1): 0, (2, 2): 0, (2, 3): 0, (2, 4): 13, (2, 5): 0, 
        (3, 0): 0, (3, 1): 0, (3, 2): 0, (3, 3): 0, (3, 4): 0, (3, 5): 0, 
        (4, 0): 0, (4, 1): 0, (4, 2): -13, (4, 3): 0, (4, 4): 0, (4, 5): 13, 
          (5, 0): 0, (5, 1): 13, (5, 2): 0, (5, 3): 0, (5, 4): -13, (5, 5): 0}

        '''

    def mincut1(self, s):
        # source node s
        n = self.V
        visited = []
        for i in range(0, n):
            visited.append(False)
        cut1 = set()
        #print(self.V)
        # for i in range(0,self.V):
        #     for j in range(0,self.V):
        #         print(self.flow[i,j])
        # for item in self.flow:
        #     print(item,self.flow[item])
        # print("flow:",self.flow)
        '''
        {(0, 0): 0, (0, 1): 16, (0, 2): 0, (0, 3): 0, (0, 4): 0, (0, 5): 15, 
        (1, 0): 4, (1, 1): 0, (1, 2): 8, (1, 3): 0, (1, 4): 4, (1, 5): 3, 
        (2, 0): 0, (2, 1): 0, (2, 2): 0, (2, 3): 10, (2, 4): 5, (2, 5): 0, 
        (3, 0): 0, (3, 1): 0, (3, 2): 0, (3, 3): 0, (3, 4): 1, (3, 5): 0, 
        (4, 0): 0, (4, 1): 0, (4, 2): 3, (4, 3): 13, (4, 4): 0, (4, 5): 0, 
        (5, 0): 1, (5, 1): 3, (5, 2): 4, (5, 3): 4, (5, 4): 6, (5, 5): 0}

        '''
        self.dfs(visited, cut1, s)
        # print("mincut done!")
        # print("能访问的节点集合",cut)
        # print(self.flow[0,1])
        return cut1

    def dfs(self, visited, cut, v):
        # print(type(cut))
        #print(v,visited)
        cut.add(v)
        visited[v] = True
        for i in range(0, self.V):
            #  self.flow[u, self.pred[u]] -= increment
            #print("ssssss",self.flow[v, i])
            if self.flow[v, i] != 0 and visited[i] == False:
                self.dfs(visited, cut, i)
    # def gomory(self):
    #     T = {}
    #     for i in range(self.V-1):
    #         for j in range(i+1,self.V):
    #             # 直接求i-j的最大流不就成了
    #             s,t = i,j
    #             print("s,t=",s,t)
    #             T[s,t] = self.max_flow(s,t)
    #     print(T)
    #     '''
    #     {(0, 1): 18, (0, 2): 13, (0, 3): 13, (0, 4): 13, (0, 5): 17,
    #                  (1, 2): 13, (1, 3): 13, (1, 4): 13, (1, 5): 17,
    #                              (2, 3): 14, (2, 4): 15, (2, 5): 13,
    #                                          (3, 4): 14, (3, 5): 13,
    #                                                      (4, 5): 13}
    #
    #     '''
    def buildTree(self):
        """构建 GomoryHuTree"""
        p = []
        f1 = []

        for i in range(self.V):
            p.append(0)
            f1.append(0)
            for j in range(self.V):
                self.tree[i, j] = 0

        for s in range(1, self.V):

            t = p[s]

            min_cut = self.max_flow(s, t)
            #print("min_cut in line 287=",min_cut)
            #print(self.color)
            f1[s] = min_cut  # f1记录最小割值

            for i in range(self.V):
                if i != s and p[i] == t and self.color[i] == BLACK:
                    p[i] = s

            if self.color[p[t]] == BLACK:
                p[s] = p[t]
                p[t] = s
                f1[s] = f1[t]
                f1[t] = min_cut

            if s == self.V - 1:
                for i in range(1, s + 1):
                    self.tree[i, p[i]] = f1[i]

        print("build tree is done!")
        #print(self.color)
        # print(self.)
        self.convertT2G()

    def convertT2G(self):
        '''
        将tree化为矩阵形式，那么之后如果查询最大流值可以通过在该矩阵操作
        具体的查询还未做，利用的性质是：T中从u到v唯一路径上的最小权边为所求G中u-v割的值
        思路：查询一次遍历，使用图的遍历，不过可能需要调整一下
        :return:
        '''
        tm = []
        # for i in range(0,self.V):
        #     tm.append([])
        for i in range(0, self.V):
            tm.append([])
            for j in range(0, self.V):
                tm[i].append(self.tree[i, j])
                # print(tm[i][j])
                # print("self.tree[i,j]",self.tree[i,j])
        for i in range(0, self.V):
            for j in range(0, self.V):
                if tm[i][j] != 0 and tm[j][i] == 0:
                    tm[j][i] = tm[i][j]
                if tm[i][j] == 0 and tm[j][i] != 0:
                    tm[i][j] = tm[j][i]
                # tm[i].append(self.tree[i,j])
                # print(tm[i][j])
                # print("self.tree[i,j]",self.tree[i,j])
            # print(tm[i])
        self.tm = tm  # gomory tree的矩阵形式

    # def prepare(self):
    #     for i in range(self.V):
    #         for j in range(self.V):
    #             self.capacity[i, j] = self.tree[i, j]

    def getMaxFlow(self, u, v):
        '''
        :param u:  s
        :param v:  t
        :return:
        获取最大流  待完善，之后可直接在树上查询，不必每次都求解最大流'''
        return max(self.max_flow(u, v), self.max_flow(v, u))
        # return max(self.tree[u,v], self.tree[v,u])

    def printTree(self):
        print("display tree")
        # n = len(self.tree)
        # print("n=",n)
        '''    tree 的内部结构   tree={}
        {
        (0, 0): 0, (0, 1): 0, (0, 2): 0, (0, 3): 0, (0, 4): 0, (0, 5): 0, 
        (1, 0): 18, (1, 1): 0, (1, 2): 0, (1, 3): 0, (1, 4): 0, (1, 5): 0, 
        (2, 0): 0, (2, 1): 0, (2, 2): 0, (2, 3): 0, (2, 4): 15, (2, 5): 0, 
        (3, 0): 0, (3, 1): 0, (3, 2): 0, (3, 3): 0, (3, 4): 14, (3, 5): 0, 
        (4, 0): 0, (4, 1): 0, (4, 2): 0, (4, 3): 0, (4, 4): 0, (4, 5): 13, 
        (5, 0): 0, (5, 1): 17, (5, 2): 0, (5, 3): 0, (5, 4): 0, (5, 5): 0
        }
        '''
        # print(type(self.tree))   # dict
        for i in self.tree:
            if self.tree[i] != 0:
                print(i, self.tree[i])

    def kcut(self, k):
        '''
        1.求gomory tee
        2.设C为图G中与T关联的n-1个割中最轻的k-1个割的并，输出C
        :return:
        '''
        st_cut = {}
        for i in self.tree:
            if self.tree[i] != 0:
                st_cut[i] = self.tree[i]
        T = sorted(st_cut.items(), key=lambda x: x[1])  # 排序
        # print(T)
        # print("k=", k)
        # print(T[:k])
        kcut = []
        for item in T[:k-1]:
            # print(item[0][0])
            # ((4, 5), 13)
            # ((3, 4), 14)
            kcut.append(self.mincut(item[0][0], item[0][1]))
        #print("kcut is ", kcut)
        # 下面对kcut进行合并处理
        result = []
        for i in range(self.V):
            result.append([])
            for j in range(self.V):
                result[i].append(0)
        #print(result)
        # 得到一个二维矩阵
        for i in kcut:
            #print(i)
            '''
            ['(2,1)', '(2,5)', '(3,5)', '(4,1)', '(4,5)']
            ['(3,2)', '(3,4)', '(3,5)']
            '''
            for j in i:
                #print(j[1],j[3])
                index_i = int(j[1])
                index_j = int(j[3])
                #print(index_i,type(index_i))

                result[index_i][index_j] = 1;
                result[index_j][index_i] = 1;
        #print(result,len(result),len(result[0]))
        rcut = []
        cost = 0
        for i in range(len(result)):
            for j in range(i+1,len(result[i])):
                if result[i][j] !=0:
                    edge = "("+str(i)+","+str(j)+")"
                    #print(edge)
                    #print(i,j)
                    rcut.append(edge)
                    cost += self.G[i][j]
        return rcut,cost

def displayG(G):
    print("图G为：")
    for i in G:
        print(i)
if __name__ == "__main__":
    '''
    0   10  0   0   0   8
    10  0   4   0   2   3
    0   4   0   5   4   2
    0   0   5   0   7   2
    0   2   4   7   0   3
    8   3   2   2   3   0
    '''
    graph = [[0, 10, 0, 0, 0, 8],
             [10, 0, 4, 0, 2, 3],
             [0, 4, 0, 5, 4, 2],
             [0, 0, 5, 0, 7, 2],
             [0, 2, 4, 7, 0, 3],
             [8, 3, 2, 2, 3, 0]]
    displayG(graph)
    # 构建 GomoryHuTree
    tree = GomoryHuTree(graph)

    # 输出树
    tree.printTree()
    # # 查询s、t的最小割值
    # print(tree.getMaxFlow(3, 0))
    # print(tree.getMaxFlow(0, 3))
    # #输出边(0, 3)
    # print(tree[0, 3])
    print("示例，求解0-3割：")
    tree.mincut(0, 3)

    k = int(input("下面开始求k割："))
    kcut,cost = tree.kcut(k)
    print("kcut is ",kcut)
    print("cost is ",cost,"（该值正常小于等于两个割的和）")