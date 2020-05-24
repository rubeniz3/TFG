# Analisis exploratorio de los datos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_excel('dataset/LD2011_2014.xlsx')
df.head()

df_original = df.copy()

# Analisis inicial
df.shape
df['MT_362'] = df['MT_362']/100
df.dtypes
df.dtypes.isnull().any()

# Inicio de las medidas, no todos los clientes inician las medidas en la misma fecha
df_start_date = df.ne(0).idxmax().drop(labels = 'Unnamed: 0')
df_start_date[:] = df.loc[df_start_date.values, 'Unnamed: 0'][:]

# Configurar index del dataset, elegimos la serie temporal como indice
df.index = df['Unnamed: 0']
df.index.name = 'date'
df = df.drop('Unnamed: 0', axis = 1)
df.head()


# ----------------------------------------------------------------------------------
# Visualización gráfica

# Analisis mensual
# Muestreamos por mes, sumando todas las columnas y dividiendo entre 4 para hallar los kWh
df = df/4
df_total_per_month = df['2011':'2014'].resample('M').sum().sum(axis = 1)
df_total_per_month = (df_total_per_month)/1e6
#df_total_per_month

#Numero de hogares medidos por mes
df_num_hogares = pd.Series(index = df_total_per_month.index)

for i in df_num_hogares.index:
    df_num_hogares[i] = df_start_date[df_start_date[:] <= i].size



# Consumo total en toda la serie temporal
df_total_per_month_graph = df_total_per_month.copy()
df_total_per_month_graph.index = df_total_per_month_graph.index.strftime('%b-%y')
df_total_per_month_graph.plot.bar(rot = 45, figsize = (20,10))
plt.ylabel("Consumo total medido (en GWh)", fontsize = 18)
plt.xlabel("Mes-Año" , fontsize = 18)
plt.title("Consumo total en toda la serie temporal", fontsize = 24)
plt.show()

# Número de clientes / medidas por mes
df_num_hogares_graph = df_num_hogares.copy()
df_num_hogares_graph.index = df_num_hogares_graph.index.strftime('%b-%y')
df_num_hogares_graph.plot.bar(figsize = (20,10), color = 'darkred', rot = 45)
plt.ylabel("Número de medidas", fontsize = 18)
plt.xlabel("Mes-Año" , fontsize = 18)
plt.title("Número de clientes / medidas por mes", fontsize = 24)
plt.show()

# Consumo total medio por cliente al mes
df_consumo_total_medio_por_hogar_graph = (df_total_per_month_graph[:]/df_num_hogares_graph[:])*1000
df_consumo_total_medio_por_hogar_graph.plot.bar(rot = 45, figsize = (20,10), color = 'limegreen')
plt.ylabel("Consumo total medio (en MWh)", fontsize = 18)
plt.xlabel("Mes-Año" , fontsize = 18)
plt.title("Consumo total medio por cliente al mes", fontsize = 24)
plt.show()

# Consumo total medio por cliente al mes normalizado
df_total_per_month_norm = df['2011':'2014'].resample('M').sum()
df_total_per_month_norm[:] = df_total_per_month_norm[:]/df_total_per_month_norm.max()
df_total_per_month_norm = df_total_per_month_norm.sum(axis = 1)
df_consumption_per_client_norm = df_total_per_month_norm/df_num_hogares

df_consumption_per_client_norm_graph = df_consumption_per_client_norm.copy()
df_consumption_per_client_norm_graph.index = df_consumption_per_client_norm_graph.index.strftime('%b-%y')
(df_consumption_per_client_norm_graph*100).plot.bar(rot = 45, figsize = (20,10), color = 'limegreen')
plt.ylabel("Variación sobre el consumo máximo (%)", fontsize = 18)
plt.xlabel("Mes-Año" , fontsize = 18)
plt.title("Consumo total medio por cliente al mes normalizado", fontsize = 24)
plt.show()




# Análisis semanal

#Semanal por día
df_total_per_day = df['2011':'2014'].resample('D').sum().sum(axis = 1)
days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday']
df_total_per_day_sum = df_total_per_day.groupby(df_total_per_day.index.weekday_name).sum().reindex(days)/1e6 #Valores en GWh

df_total_per_day_sum.plot.bar(rot = 45)
plt.ylabel("Consumo total (GWh)", fontsize = 12)
plt.xlabel("Día de la semana" , fontsize = 12)
plt.title("Consumo total por día", fontsize = 20)
plt.show()


#Evolucion anual
days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday']
df_weekdays_2011 = df_total_per_day['2011'].groupby(df_total_per_day['2011'].index.weekday_name).sum().reindex(days)/1e6
df_weekdays_2012 = df_total_per_day['2012'].groupby(df_total_per_day['2012'].index.weekday_name).sum().reindex(days)/1e6
df_weekdays_2013 = df_total_per_day['2013'].groupby(df_total_per_day['2013'].index.weekday_name).sum().reindex(days)/1e6
df_weekdays_2014 = df_total_per_day['2014'].groupby(df_total_per_day['2014'].index.weekday_name).sum().reindex(days)/1e6


plt.subplot(221)
df_weekdays_2011.plot.bar(rot = 45, figsize = (20,10))
plt.ylabel('GWh')
plt.title('2011', fontsize = 16)
plt.xlabel("")
plt.subplot(222)
df_weekdays_2012.plot.bar(rot = 45, figsize = (20,10))
plt.ylabel('GWh')
plt.title('2012', fontsize = 16)
plt.xlabel("")
plt.subplot(223)
df_weekdays_2013.plot.bar(rot = 45, figsize = (20,10))
plt.ylabel('GWh')
plt.title('2013', fontsize = 16)
plt.xlabel("")
plt.subplot(224)
df_weekdays_2014.plot.bar(rot = 45, figsize = (20,10))
plt.ylabel('GWh')
plt.title('2014', fontsize = 16)
plt.xlabel("")

plt.subplots_adjust(top = 5, bottom = 4)
plt.show()



# Semanal por día medio
df_total_per_day_medio = df_total_per_day.copy()
for i in df_total_per_day_medio.index:
    df_total_per_day_medio[i] = df_total_per_day_medio[i] / df_num_hogares[(df_num_hogares.index.year == i.year) & (df_num_hogares.index.month == i.month)]

df_total_per_day_medio_weekday = df_total_per_day_medio.groupby(df_total_per_day_medio.index.weekday_name).sum().reindex(days)/1e6

# Consumo total medio por cliente según el día
df_total_per_day_medio_weekday.plot.bar(rot = 45, figsize = (10,5), color = 'limegreen')
plt.xlabel("Día de la semana", fontsize = 14)
plt.title("Consumo total medio por cliente según el día")
plt.ylabel("Consumo en GWh")
plt.show()



# Semanal por día normalizado
df_total_per_day_norm = df['2011':'2014'].resample('D').sum()
df_total_per_day_norm[:] = df_total_per_day_norm[:]/df_total_per_day_norm.max()
df_total_per_day_norm = df_total_per_day_norm.sum(axis=1)
for i in df_total_per_day_norm.index:
    df_total_per_day_norm[i] = df_total_per_day_norm[i] / df_num_hogares[(df_num_hogares.index.year == i.year) & (df_num_hogares.index.month == i.month)]

# Variacion del consumo por día (%)
(df_total_per_day_norm*100).plot()
plt.title("Variación del consumo por día")
plt.xlabel("Fecha")
plt.ylabel("Variación consumo (%)")
plt.show()

df_total_per_day_norm_weekday = df_total_per_day_norm.groupby(df_total_per_day_medio.index.weekday_name).mean().reindex(days)
(df_total_per_day_norm_weekday*100).plot.bar(rot = 45, figsize = (10,5), color = 'limegreen')
plt.xlabel("Día de la semana", fontsize = 14)
plt.title("Variación del consumo según el día")
plt.ylabel("Variación del consumo (%)")
plt.show()







# Consumo horario
# Consumo total según la hora del día
df_total_per_hour = df.sum(axis = 1).groupby(df.index.time).sum()/1e6
df_total_per_hour_graph = df_total_per_hour.copy()
df_total_per_hour_graph.index = pd.to_datetime(df_total_per_hour_graph.index, format = "%H:%M:%S").strftime('%H:%M')
df_total_per_hour_graph.plot(figsize = (20,10), linewidth = 4)
plt.title("Consumo total según la hora del día", fontsize = 32)
plt.ylabel("Consumo en GWh", fontsize = 16)
plt.xlabel("Hora", fontsize = 16)
plt.grid()
plt.show()


# Consumo total Mayo 2012
(df['2012-05'].sum(axis = 1)/1e3).plot(figsize = (20,10))
plt.title("Consumo Total Mayo 2012", fontsize = 32)
plt.ylabel("Consumo (MWh)", fontsize = 16)
plt.xlabel("Dia del mes", fontsize = 16)
plt.grid()
plt.show()


# Consumo total medio segun la hora del día
df_per_hour_medio = df.copy()
df_per_hour_medio = df_per_hour_medio['2011':'2014'].sum(axis=1)
for i in df_per_hour_medio.index:
    df_per_hour_medio[i] = df_per_hour_medio[i] / df_num_hogares[(df_num_hogares.index.year == i.year) & (df_num_hogares.index.month == i.month)]



# Consumo total medio por cliente segun la hora del día
df_per_hour_medio_graph = df_per_hour_medio.groupby(df_per_hour_medio.index.time).sum()/1e3
df_per_hour_medio_graph.index = pd.to_datetime(df_per_hour_medio_graph.index, format = "%H:%M:%S").strftime('%H:%M')
df_per_hour_medio_graph.plot(figsize = (20,10), linewidth = 4, color = 'limegreen')
plt.title("Consumo total medio por cliente", fontsize = 32)
plt.xlabel("Hora", fontsize = 16)
plt.ylabel("Consumo (MWh)", fontsize = 16)
plt.grid()
plt.show()





# Consumo horario normalizado
df_total_per_hour_norm = df['2011':'2014']
df_total_per_hour_norm = df_total_per_hour_norm [:] / df_total_per_hour_norm[:].max()
df_total_per_hour_norm = df_total_per_hour_norm.sum(axis = 1)
for i in df_total_per_hour_norm.index:
    df_total_per_hour_norm[i] = df_total_per_hour_norm[i] / df_num_hogares[(df_num_hogares.index.year == i.year) & (df_num_hogares.index.month == i.month)]


# Variación del consumo medio por cliente
df_total_per_hour_norm_graph = df_total_per_hour_norm.groupby(df_per_hour_medio.index.time).sum()/1e3
df_total_per_hour_norm_graph.index = pd.to_datetime(df_total_per_hour_norm_graph.index, format = "%H:%M:%S").strftime('%H:%M')
(df_total_per_hour_norm_graph*100).plot(figsize = (20,10), linewidth = 4, color = 'limegreen')
plt.title("Variacion del consumo medio por cliente (%)", fontsize = 32)
plt.xlabel("Hora", fontsize = 16)
plt.ylabel("Variación del consumo (%)", fontsize = 16)
plt.grid()
plt.show()






# Dias especiales

df_total_per_day = df['2011':'2014'].resample('D').sum()
df_total_per_day[:] = df_total_per_day[:]/df_total_per_day.max()
df_total_per_day = df_total_per_day.sum(axis = 1)
for i in df_total_per_day.index:
    df_total_per_day[i] = df_total_per_day[i] / df_num_hogares[(df_num_hogares.index.year == i.year) & (df_num_hogares.index.month == i.month)]

moving_avg = df_total_per_day.rolling(31).mean()

# Variación porcentual del consumo
(df_total_per_day*100).plot(figsize = (15,10), label = "Valores medidos")
(moving_avg*100).plot(color = 'red', linewidth = 3, label = "Media móvil")
plt.legend(fontsize = 16)
plt.ylabel('Variación del consumo (%)')
plt.grid(which = 'both')
plt.title("Variación porcentual del consumo", fontsize = 24)
plt.show()

df_total_per_day[df_total_per_day[:] < 0.8*moving_avg[:]]
df_total_per_day[df_total_per_day[:] > 1.10*moving_avg[:]]