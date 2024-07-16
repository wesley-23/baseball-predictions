from data.heat_chart import heat_chart

m = heat_chart(2021, neighbors = 50, kernel = 'ekernel')
m.create_heat_chart()
print(m.is_hit(77, 25))


