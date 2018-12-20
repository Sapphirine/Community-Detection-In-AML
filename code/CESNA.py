import numpy as np
import colorsys
from itertools import permutations
###############################################################
#               Main function of CESNA.                       #
###############################################################
class Graph:
    def __init__(self, adj_matrix, attributes):
        self.adj_matrix = adj_matrix # Generated dataset with serveral node groups
        self.attributes = attributes # Generated node attributes
        self.E = self.number_of_edges() # Number of Edges

    def number_of_edges(self):
        return np.sum(self.adj_matrix != 0)

    def get_node_number(self):
        return len(self.adj_matrix) # Number of nodes

    def get_attributes_number(self):
        # assume a node id is 1.
        return len(self.attributes[1]) # Number of attributes

    def get_number_of_edges(self):
        return self.number_of_edges

    def get_out_degree(self, u):
        return np.sum((self.adj_matrix != 0)[u])

    def get_in_degree(self, u):
        return np.sum((self.adj_matrix != 0)[:, u])

    def get_degree(self, u):
        return self.get_in_degree(u) + self.get_out_degree(u)

    def get_neighbour(self, u): # Number of neighbour
        # Make sure u is a node in G.#
        neighbour = set()
        for v in range(self.get_node_number()):
            if self.adj_matrix[u][v] != 0 or self.adj_matrix[v][u]:
                neighbour.add(v)
        neighbour.add(u)
        return neighbour 

    # Function of Conductance Computation
    def get_conductance(self, S):
        cut = edges = edges_=0
        for u in range(self.get_node_number()):
            for v in range(self.get_node_number()):
                if self.adj_matrix[u][v] != 0 or self.adj_matrix[v][u] != 0:
                    if u in S and v in S and self.adj_matrix[u][v]:
                        edges += 1
                    elif u not in S and v not in S:
                        edges_ += 1
                    else:
                        cut += 1
        vol = edges*2 + cut
        vol_ = edges_*2 + cut
        if min(vol, vol_) == 0:
            phi = 0
        else:
            phi = cut/min(vol, vol_)
        return phi
def NeighborComInit(G, F, C):
    NPhi=[]
    for u in range(G.get_node_number()):
        degree = G.get_degree(u)
        if degree < 0:               # according to paper, do not include nodes with a small degree
            phi = 1.0
        else:
            neighbour = G.get_neighbour(u)
            phi = G.get_conductance(neighbour)
        NPhi.append((u, phi))
    NPhi.sort(key=lambda x: x[1])
    print("conductance computation completed")
    c = 0
    invalid_u = set()
    for r in NPhi:
        u = r[0]
        if u in invalid_u:
            continue
        F[u][c] = 1
        for v in G.get_neighbour(u):
            F[v][c] = 1
            invalid_u.add(v)
        c += 1
        if c >= C:
            break
    if c < C:
        print(str(C-c)+" communities needs to be filled randomly")
    for c_ in range(c, C):
        Comsz = 10
        us = np.random.randint(0, G.get_node_number(), Comsz)
        for u in us:
            F[u][c_]=np.random.rand()

# Function of updating G matrix
def updata_W(G, F, W, l, alpha, t, tol):
    F_ = np.insert(F, len(F[0]), 1, axis=1)
    X = G.attributes
    error = float("inf")
    while alpha > tol:
        Q = 1 / (1 + np.exp(-W.dot(F_.T)).T)
        gradient = (X - Q).T.dot(F_) - 0.2*np.sign(W)
        W += alpha*gradient
        alpha *= t

# Function of updating F matrix
def updata_F(G, F, W, C, alpha, t, tol):
    X = G.attributes
    weight = G.adj_matrix
    while alpha > tol:
        F_ = np.insert(F, len(F[0]), 1, axis=1)
        Q = 1 / (1 + np.exp(-W.dot(F_.T)).T)
        # print(F)
        # print(-W.dot(F_.T))
        gradient_x = (X-Q).dot(W)[:, :-1]
        mu = F.dot(F.T)
        gradient_y = (weight-F.dot(F.T)).dot(F)-F*(weight.diagonal()-mu.diagonal())[:,np.newaxis]
        gradient = gradient_y + gradient_x
        F += alpha*gradient
        F[F < 0] = 0
        alpha *= t

# Function of CESNA algorithm
def CESNA(G, C, l, alpha1, alpha2, t):
    # G: graph
    # C: number of communities
    N = G.get_node_number()
    K = G.get_attributes_number()
    F = np.zeros([N, C])
    #NeighborComInit(G, F, C)
    F += np.random.rand(N,C) # Initiate F matrix
    W = np.random.randn(K, C+1) # Initiate W matrix
    # Updating logistic parameters
    for i in range(100):
        updata_W(G, F, W, l, alpha1, t, tol=0.001) # Update F matrix
        updata_F(G, F, W, C, alpha2, t, tol=0.001) # Update W matrix
    updata_F(G, F, W, l, alpha1, t, tol=0.001) # Update F matrix at last
    return F, W


# Function of Save Json File
def save_json(G, F, attributes):
    C = len(F[0])
    colors = []
    for c in range(C):
        color = np.array(colorsys.hls_to_rgb(c/C, 0.8, 0.8))*255
        colors.append(color)
    colors = np.array(colors)
    with open('graph.json', 'w') as file:
        s = ''
        s += '{\n' + '  "nodes": [\n'
        for u in range(G.get_node_number()-1):
            #u_color = np.sum(colors*((F[u])/sum(F[u]))[:, np.newaxis], axis=0).astype(np.int)
            u_color = np.sum(colors*(F[u]==np.max(F[u]))[:, np.newaxis], axis=0).astype(np.int)
            u_color_code = "#"+"{:02x}".format(u_color[0])+"{:02x}".format(u_color[1])+"{:02x}".format(u_color[2])
            #print(u_color_code)
            u_attributes = str(attributes[u])
            s += """    {"id": """ + str(u)+ """, "color": """ + '"' + u_color_code + '"' + """, "attr": """ + '"' + u_attributes+ '"' + "},\n"
        u = G.get_node_number()-1
        #u_color = np.sum(colors * ((F[u]) / sum(F[u]))[:, np.newaxis], axis=0).astype(np.int)
        u_color = np.sum(colors * (F[u] == np.max(F[u]))[:, np.newaxis], axis=0).astype(np.int)
        u_color_code = "#" + "{:02x}".format(u_color[0]) + "{:02x}".format(u_color[1]) + "{:02x}".format(u_color[2])
        u_attributes = str(attributes[u])
        #print(u_color_code)
        s += """    {"id": """ + str(
            u) + """, "color": """ + '"' + u_color_code + '"' + """, "attr": """ + '"' + u_attributes + '"' + "}\n"
        s += "  ],\n"
        s += """  "links": [\n"""
        for u in range(G.get_node_number()):
            for v in range(G.get_node_number()):
                if G.adj_matrix[u][v] != 0:
                    s += """    {"source": """ + str(u) + """, "target": """+str(v)+""", "value":""" + str(G.adj_matrix[u][v]) + "},\n"
        s = s[0:-2] + '\n'
        file.write(s)
        file.write('  ]\n')
        file.write('}')

# Function of generating dataset
def generate_data(C_numbers):
    N = sum(C_numbers)
    adj_matrix = np.zeros((N,N))
    nodes = set(i for i in range(N))
    communities = []
    # for c in C_numbers:
    #     comm = np.random.choice(list(nodes), c, replace=False)
    #     communities.append(comm)
    #     nodes = nodes.difference(comm)
    s = 0
    for c in c_numbers:
        communities.append(np.arange(c)+s)
        s += c
    for community in communities:
        for i in range(len(community)):
            for j in range(i, len(community)):
                if np.random.rand() > 0.9:
                    weight = 0.2*np.random.randn()+1
                    adj_matrix[community[i]][community[j]] = weight
                    adj_matrix[community[j]][community[i]] = weight
    for u in nodes:
        for v in nodes:
            if np.random.rand() > 0.99:
                weight = max(0,0.02 * np.random.randn() + 0.1)
                adj_matrix[u][v] += weight
                adj_matrix[u][v] += weight
    return adj_matrix

# Function of generating node attributes
def gen_attributes_matrix(group_list):
    c = len(group_list)
    total = sum(group_list)
    t = np.array([[0.95,0.98,0.08,0.93,0.90,0.13,0.92,0.02],
                [0.98, 0.02, 0.96, 0.01, 0.09, 0.03, 0.06, 0.93],
                [0.02,0.89,0.03,0.05,0.03,0.09,0.07,0.97],
                [0.03,0.08,0.93,0.95,0.05,0.04,0.02,0.08],
                [0.92,0.04,0.02,0.09,0.98,0.03,0.06,0.10],
                [0.07,0.94,0.06,0.02,0.92,0.09,0.96,0.05],
                [0.10, 0.09, 0.01, 0.04, 0.08, 0.91, 0.96, 0.03],
                [0.06,0.07,0.04,0.98,0.05,0.94,0.03,0.06]])
    t = t[0:c]
    attribute1 = []
    attribute2 = []
    attribute3 = []
    attribute4 = []
    attribute5 = []
    attribute6 = []
    attribute7 = []
    attribute8 = []
    for n in range(0,len(t)):
        attribute1.extend(np.random.binomial(1, t[n][0], size=group_list[n]).tolist())
        attribute2.extend(np.random.binomial(1, t[n][1], size=group_list[n]).tolist())
        attribute3.extend(np.random.binomial(1, t[n][2], size=group_list[n]).tolist())
        attribute4.extend(np.random.binomial(1, t[n][3], size=group_list[n]).tolist())
        attribute5.extend(np.random.binomial(1, t[n][4], size=group_list[n]).tolist())
        attribute6.extend(np.random.binomial(1, t[n][5], size=group_list[n]).tolist())
        attribute7.extend(np.random.binomial(1, t[n][6], size=group_list[n]).tolist())
        attribute8.extend(np.random.binomial(1, t[n][7], size=group_list[n]).tolist())
    attribute = np.array([attribute1,attribute2,attribute3,attribute4,attribute5,attribute6,attribute7,attribute8])
    return(attribute.T)

# Algorithm Performance Evaluation: Cross-Entropy Loss
def entropy(C_, C):
    C_ = C_/np.sum(C_)
    C = C/np.sum(C)
    perm = list(permutations(C_))
    min_entropy = float("inf")
    for p in perm:
        entropy = 0
        for r in zip(p, C):
            entropy += -r[0]*np.log(r[1])
        if entropy < min_entropy:
            min_entropy = entropy
    return min_entropy

# Algorithm Performance Evaluation: Error Rate
def ErrorRate(labels, c_numbers):
    n = len(labels)
    ground_truth = []
    index = 0
    for c_number in c_numbers:
        ground_truth += [index]*c_number
        index += 1
    correct_group = 0
    for u in range(n):
        for v in range(n):
            if u != v:
                if (labels[u] == labels[v]) == (ground_truth[u]==ground_truth[v]):
                    correct_group += 1
    return 1 - correct_group / (n*(n-1))


# Main Function to process through the algorithm and get the results
if __name__ == '__main__':
    c_numbers = [21, 23, 29, 33, 35, 38, 42, 52]
    # c_numbers = [22, 31]
    # c_numbers=[21, 33, 45, 53]
    C = len(c_numbers)
    # C = 8
    adj_matrix = generate_data(c_numbers)
    attributes = gen_attributes_matrix(c_numbers)
    #attributes = np.zeros([sum(c_numbers), 1])
    G = Graph(adj_matrix, attributes)
    l = 1
    alpha1 = 0.5
    alpha2 = 0.05
    t = 0.9
    F, W = CESNA(G, C, l, alpha1, alpha2, t)
    print(F)
    print(W)
    print(F.dot(F.T))
    error = adj_matrix-F.dot(F.T)
    F_ = np.insert(F, len(F[0]), 1, axis=1)
    print(1 / (1 + np.exp(-W.dot(F_.T)).T))
    labels = np.zeros(F.shape[0])
    for u in range(F.shape[0]):
        labels[u] = np.argmax(F[u])
    C_ = np.zeros(C)
    for i in labels:
        C_[int(i)] += 1
    print(labels)
    print(C_)
    print(entropy(C_, c_numbers))
    print(ErrorRate(labels, c_numbers))
    save_json(G, F, attributes)
    # np.savetxt("F.csv", F, delimiter=",")
