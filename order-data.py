def sortByYear(e):
    attr = e.split(',')
    year = int(attr[3])
    return year

def sortById(e):
    attr = e.split(',')
    id = int(attr[2])
    return id
def parse():
    f = open('batter_stats.csv', 'r')
    list = []
    for line in f:
        line = line[:-1]
        list.append(line)
    header = list.pop(0)
    list.sort(key=sortByYear)
    list.sort(key=sortById)
    o = open('batter_stats_sorted.csv', 'a')
    o.write(header + '\n')
    for line in list:
        o.write(line + '\n')
    o.close()

parse()


    
