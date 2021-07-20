def merge_cluster(n, connect):
    print(connect)

    clusters = []
    for con in connect:
        print(f"LR: {con}, cluster {clusters}")
        left,right = con[0],con[1]
        if not clusters:
            clusters.append([left,right])
            continue
        idx_left = [idx for idx,group in enumerate(clusters) if left in group]
        idx_right = [idx for idx,group in enumerate(clusters) if right in group]
        if idx_left == idx_right and idx_left!=[]:
            pass
        elif idx_left != [] and idx_right!=[]:
            a,b = clusters[idx_left[0]], clusters[idx_right[0]]
            clusters.remove(a)
            clusters.remove(b)
            clusters.append(a+b)

        elif idx_left != [] :
            clusters[idx_left[0]].append(right)
        else:
            clusters[idx_right[0]].append(left)
    
        
        # print(clusters,connect)

    # print(' ')

    l = len(clusters)

    if l>1:
        return False
    return True
                
            
        

def criticalConnections( n, connections):
    """
    :type n: int
    :type connections: List[List[int]]
    :rtype: List[List[int]]
    """
    outputs =[]

    for kdx,m in enumerate(connections) :
        # connections.remove(m)
        dcopy = list(connections).copy()
        dcopy.remove(m)
        res = merge_cluster(n,dcopy)
        if not res:
            outputs.append(m)
            print(outputs)
        print(f"this is {connections}")
        # connections.insert(kdx,m)
    return outputs
criticalConnections(4, [[0,1],[1,2],[2,0],[1,3]])