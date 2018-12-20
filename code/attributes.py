import numpy as np

def gen_attributes_matrix(group_list):
    c = len(group_list)
    total = sum(group_list)
    t = np.array([[0.95,0.98,0.08,0.93,0.90,0.13,0.92,0.02],
                [0.10,0.09,0.01,0.04,0.08,0.91,0.96,0.03],
                [0.02,0.89,0.03,0.05,0.03,0.09,0.07,0.97],
                [0.03,0.08,0.93,0.95,0.05,0.04,0.02,0.08],
                [0.92,0.04,0.02,0.09,0.98,0.03,0.06,0.10],
                [0.07,0.94,0.06,0.02,0.92,0.09,0.96,0.05],
                [0.98,0.02,0.96,0.01,0.09,0.03,0.06,0.93],
                [0.06,0.07,0.04,0.98,0.05,0.94,0.03,0.06]])
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

if __name__ == '__main__':
    group_list = [110,123,129,133,135,138,142,202]
    attributes = gen_attributes_matrix(group_list)
    print(attributes)
    print(len(attributes))
    print(len(attributes[0]))
