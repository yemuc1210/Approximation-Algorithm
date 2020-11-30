import itertools
from heapq import *

class PriorityQueue:
    def __init__(self):
        self._pq = []
        self._entry_map = {}
        self._counter = itertools.count()

    def addtask(self, task, priority = 0):
        '''添加任务，或者是更新一个已存在任务的优先级
        '''
        if task in self._entry_map:     # 若任务已经存在
            self.removetask(task)
        count = next(self._counter)
        #print((count))
        entry = [priority, count, task]     #三元组为 优先权，计数和任务
        #print(entry)
        self._entry_map[task] = entry
        #print(self._entry_map)
        heappush(self._pq, entry)     #将元素push到堆上，保持堆的性质不变
        #print(self._pq)

    def removetask(self, task):
        '''标记一个任务，为移除状态'''
        entry = self._entry_map.pop(task)   # 弹出即移除
        entry[-1] = 'removed'    # entry = [priority, count, task]

    def poptask(self):
        '''移除并返回最低优先权的任务'''
        while self._pq:
            priority, count, task = heappop(self._pq)    #从堆中移出
            if task != 'removed':
                del self._entry_map[task]
                return task

    def __len__(self):
        return len(self._entry_map)

import math
MAXPRIORITY = math.inf        #定义一个优先权上限可以用math库中的常数，inf的类型为float

class SetCover:
    def __init__(self,S,W):
        self.S = S
        self.W = W
        self.selected = list()
        self.cost = 0

    def SetCovel_Solver(self):
        '''
            贪婪集合覆盖算法：选择成本效益最小的集合，比如说这个集合为S：min(w[S]/|S-C|，其中C为当前的被覆盖的元素集合    C = C U S
            使用优先权队列进行选择最具成本效益的集合
            输入:
            udict - universe U, which contains the <elem, setlist>. (dict)
            S - sets的集合，list类型
            w - S中对应的权重，list类型

            输出:
            selected: 按顺序选定的集合ids。(list)
            cost: 选择集合的总代价
            '''

        udict = {}
        # selected = list()
        scopy = []  # s的copy，因为s会不断被修改
        for index, item in enumerate(S):
            scopy.append(set(item))
            for j in item:
                if j not in udict:
                    udict[j] = set()
                udict[j].add(index)

        pq = PriorityQueue()  # 这里声明优先权队列
        #cost = 0  # 初始代价为0
        coverednum = 0
        for index, item in enumerate(scopy):  # 将集合添加到优先权队列
            if len(item) == 0:
                pq.addtask(index, MAXPRIORITY)
            else:
                pq.addtask(index, float(self.W[index]) / len(item))
        while coverednum < len(udict):
            a = pq.poptask()  # 获取最具成本效益的集合
            self.selected.append(a)  # a: 集合 id
            self.cost += self.W[a]  # 更新cost
            # print("cost now is :{}".format(cost))
            coverednum += len(scopy[a])
            # 更新包含最新被包含元素的集合
            # print(scopy)
            # print(S)
            # print(a)
            # print(scopy[a])
            for m in scopy[a]:  # m: element   a是集合id
                # print("m:{}".format(m))   # 被选中集合的元素
                for n in udict[m]:  # n: 集合 id
                    # print("n:{}".format(n))
                    if n != a:
                        scopy[n].discard(m)
                        if len(scopy[n]) == 0:
                            pq.addtask(n, MAXPRIORITY)
                        else:  # 选取集合n，对每个e∈S-C，规定优先权为α=w[n]/|S-C|
                            pq.addtask(n, float(self.W[n]) / len(scopy[n]))
            scopy[a].clear()
            pq.addtask(a, MAXPRIORITY)

        return self.selected, self.cost


if __name__ == "__main__":
    S = [[1,2,3],
         [3,6,7,10],
         [8],
         [9,5],
         [4,5,6,7,8],
         [4,5,9,10],]           # 集合的collecton
    Cost = [1, 2, 3, 4, 3, 5]      # 每个集合的权重
    setCover = SetCover(S, Cost)
    print("集合U，以及对应的权重为")
    #print(S[0])
    for i in range(len(S)):
        print(S[i],"cost=",Cost[i])
    selected, cost = setCover.SetCovel_Solver()
    print("选择的集合id为 :{}".format(selected))
    for i in selected:
        print("{0}，对应的cost is {1} ".format(S[i],Cost[i]))
    print("总代价为:{}".format(cost))
