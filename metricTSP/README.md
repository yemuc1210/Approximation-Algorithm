# Metric TSP
This is an approximation algorithm to solve metric TSP

Approximation based on a minimum spanning tree

    factor is 2

Imporving

    factor is 3/2
    Algorithm 3.10 (Metric TSP – factor 3/2)
    1. Find an MST of G, say T.
    2. Compute a minimum cost perfect matching, M, on the set of odd-degree vertices of T. Add M to T and obtain an Eulerian graph.
    3. Find an Euler tour, T , of this graph.
    4. Output the tour that visits vertices of G in order of their first appearance in T . Let C be this tour.
    
    在最小生成树T的奇数度顶点集上计算最小费用完美匹配M。把M加到T上，得到欧拉图。
    找这个图的欧拉环游，按照顶点在欧拉环游中首次出现的次序访问G中顶点，得到环游C，记为结果

