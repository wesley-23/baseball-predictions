from data.heat_chart import heat_chart

m = heat_chart(2021, to = 2022)
m.create_heat_chart()
print(m.is_hit(77, 25))
