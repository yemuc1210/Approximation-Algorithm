# K Cut Problem
##1. problem definiton
    Problem 4.2 (Minimum k-cut) A set of edges whose removal leaves k connected components is called a k-cut. 
    The k-cut problem asks for a minimum weight k-cut.
##2. Algorithm
###(1)gomory-hu tree
Let **T** be a tree on vertex set **V** ; the edges of T need not be in **E**. Let **e** be an edge in T. 
Its removal from T creates two connected components. Let **S** and **S'** be the vertex sets of these components. 
The cut defined in graph G by the partition **(S, S')** is the cut associated with e in G. 
Define a weight function w on the edges of T. Tree T will be said to be a **Gomory–Hu tree** for G if 

    1. for each pair of vertices u, v ∈ V , the weight of a minimum u–v cut in G is the same as that in T.
    2. for each edge e ∈ T, w(e) is the weight of the cut associated with e in G
    
###(2)algorithm
    '''
    1. Compute a Gomory–Hu tree T for G.
    2. Output the union of the lightest k − 1 cuts of the n − 1 cuts associated with edges of T in G; let C be this union.
    '''
###(3)Theorem
    Algorithm achieves an approximation factor of 2 − 2/k.