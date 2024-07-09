from data.heat_chart import heat_chart

def get_players(year):
    hm = heat_chart(2022, neighbors = 50)
    f = open('data/batter_stats/batter_stats_sorted.csv', 'r')
    next(f)
    for line in f:
        line = line[:-1]
        l = line.split(',')
        y = int(l[3])
        if year == y:
            id = l[2]
            path = 'data/' + str(year) + '_pbp/' + id + '.csv'
            pbp = open(path, 'r')
            plays = 0
            hits = 0
            next(pbp)
            for play in pbp:
                p = play.split(',')
                ev = p[5].strip()
                la = p[6].strip()
                if la == 'None' or ev == 'None':
                    continue
                if hm.is_hit(int(round(float(ev))), int(round(float(la)))):
                    hits += 1
                plays += 1
            pbp.close()
            expected_hits = hits/plays
            line += ','+str(expected_hits) + '\n'
            write(line, year)

def write(line, year):
    path = 'data/training_data/' + str(year) + '_with_hit_predictions_K=50.csv'
    f = open(path, 'a')
    f.write(line)
    f.close()
                
get_players(2022)

