from numpy import random


f = open("data.txt", "a")
n = 1000
f.write(str(n))

connections = {}


for i in range(n):
    
    x = random.randint(n)
    y = random.randint(n)
    weight = random.randint(10)
    
    if(x == y):
        if(x > 1):
            y = x - 1
        elif(x < n):
            y = x + 1

    if(weight == 0):
        weight = 1
    
    connections[str(x) + "," + str(y)] = weight
    connections[str(y) + "," + str(x)] = weight


data = []

for i in range(n):
    
    new = []

    for j in range(n):

        new.append(0)

    data.append(new)


for key in connections:
    values = key.split(',')
    x = int(values[0])
    y = int(values[1])

    data[x][y] = connections.get(key)
    data[y][x] = connections.get(key)


for i in range(n):
    for j in range(n):
        f.write(" " + str(data[i][j]))


f.close()

