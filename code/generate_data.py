import numpy as np
def generate_data(C_numbers):
    N = sum(C_numbers)
    adj_matrix = np.zeros((N,N))
    nodes = set(i for i in range(N))
    communities = []
    for c in C_numbers:
        comm = np.random.choice(list(nodes), c, replace=False)
        communities.append(comm)
        nodes = nodes.difference(comm)
    for community in communities:
        for i in range(len(community)):
            for j in range(i, len(community)):
                weight = max(0, 0.2*np.random.randn()+0.7)
                adj_matrix[community[i]][community[j]] = weight
                adj_matrix[community[j]][community[i]] = weight
    return adj_matrix




if __name__ == '__main__':
    c_numbers = [10, 23, 29, 33, 35, 38, 42, 102]
    generate_data(c_numbers)