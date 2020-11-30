def bfs(self, s, t, route):
    '''
    用于求解最大流的广度优先遍历
    :param self:
    :param s:   源点
    :param t:   终点
    :param route:    访问路径
    :return:
    '''
    # print("route is ",route)
    visited = []
    for i in range(0, self.V):
        visited.append(False)
        self.color[i] = WHITE    # 标志边是否被访问过，以及访问次数  白：0，黑：2，灰：1
    visited[s] = True
    queue = deque()    # 库函数，一个双向队列
    route[s] = -1

    for i in range(0, self.V):
        if self.flow[s, i] != 0 and visited[i] == False:
            queue.append(i)
            route[i] = s
            visited[i] = True
    while queue:
        middleVex = queue.popleft()
        self.color[middleVex] = BLACK

        if middleVex == t:
            return True
        else:
            for i in range(0, self.V):
                if self.flow[middleVex, i] != 0 and visited[i] == False:
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
    self.f = []    # 当前流
    for i in range(0, self.V):
        route.append(0)

    for i in range(0, self.V):
        self.f.append([])
        for j in range(0, self.V):  # 初始化
            self.flow[i, j] = self.capacity[i, j]
            # self.flow[i, j] = 0
            self.f[i].append(0)
    # print(source,sink)
    while self.bfs(source, sink, route):
        min = 9999999999
        tail = sink
        head = route[sink]
        while head != -1:
            if self.flow[head, tail] < min:
                min = self.flow[head, tail]
            tail = head
            head = route[head]

        tail1 = sink
        head1 = route[tail1]

        while head1 != -1:
            if self.capacity[head1, tail1] != 0:
                self.f[head1][tail1] += min
            else:
                self.f[head1][tail1] -= min

            self.flow[head1, tail1] -= min
            self.flow[tail1, head1] += min

            tail1 = head1
            head1 = route[head1]
        for i in range(0, self.V):
            route.append(0)

    for i in range(0, self.V):
        max_flow1 += self.f[source][i]
    mincut = self.mincut1(source)
    return max_flow1  # 返回最大流的值
def buildTree(self):
    '''
    构建 GomoryHuTree
    '''
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
    self.convertT2G()
'''
求出最小割的其中一种方案: 在进行最大流计算完成之后，从源点进行遍历整张图，将遍历到的点标记为 true ；
最终结束之后，所有标记为 true 的点和没有标记的点之间就是一个割边。因为最大流之后，
所有通路在满流的那条通道处断掉了，也就是没有办法继续走下去，而这条通道一边标记为了 true,另一边没有被标记，
那么他们之间就是一个割边了。故而在残量矩阵中，从s做一次搜索即可
'''
def mincut(self, s, t):
    '''
    求s-t之间的最小割，首先求最大流，然后调用最小割
    :param s:
    :param t:
    :return:
    '''
    maxflow = self.max_flow(s,t)
    print(s, ",", t, "的最大流：", maxflow)
    vertex = set()
    n = len(self.G)
    for i in range(0, n):
        vertex.add(i)  # 顶点集合，数字
    cut = self.mincut1(s)  # 得到可遍历到的
    print("能访问的节点集合", cut)
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

def mincut1(self, s):
    # 源结点s
    n = self.V
    visited = []
    for i in range(0, n):
        visited.append(False)
    cut1 = set()
    self.dfs(visited, cut1, s)
    return cut1

def dfs(self, visited, cut, v):
    '''
    深度优先遍历，与求解割对应
    '''
    cut.add(v)
    visited[v] = True
    for i in range(0, self.V):
        if self.flow[v, i] != 0 and visited[i] == False:
            self.dfs(visited, cut, i)

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
    T = sorted(st_cut.items(), key=lambda x: x[1])  # 根据valuee值对tree排序
    kcut = []
    for item in T[:k-1]:   # 求解前k个最轻的割
        kcut.append(self.mincut(item[0][0], item[0][1]))
    # 下面对kcut进行合并处理
    result = []
    for i in range(self.V):
        result.append([])
        for j in range(self.V):
            result[i].append(0)
    # 得到一个二维矩阵
    for i in kcut:    # 有各种边，合并的话需要删除重复的边
        for j in i:
            index_i = int(j[1])
            index_j = int(j[3])

            result[index_i][index_j] = 1;
            result[index_j][index_i] = 1;
    rcut = []       #result cut
    cost = 0
    for i in range(len(result)):
        for j in range(i+1,len(result[i])):
            if result[i][j] !=0:
                edge = "("+str(i)+","+str(j)+")"
                rcut.append(edge)
                cost += self.G[i][j]
    return rcut,cost     # 返回结果kcut集合，以及总代价