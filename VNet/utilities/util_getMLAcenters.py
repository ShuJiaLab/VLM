from torch import cat,IntTensor
from csv import reader as csvreader

def get_lenslet_centers(filename):
    x,y = [], []
    with open(filename,'r') as f:
        reader = csvreader(f,delimiter='\t')
        for row in reader:
            x.append(int(row[0]))
            y.append(int(row[1]))
    lenslet_coords = cat((IntTensor(x).unsqueeze(1),
                                IntTensor(y).unsqueeze(1)),1)
    return lenslet_coords