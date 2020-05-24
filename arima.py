import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

%matplotlib inline
plt.rcParams["figure.figsize"] = [20, 10]

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)




# Importamos la serie a usar: df_consumption_per_client_norm
from statsmodels.tsa.seasonal import seasonal_decompose
df_descomposicion = seasonal_decompose(df_consumption_per_client_norm, model='multiplicative')
df_descomposicion.plot()
plt.show()

# Media móvil con ventana = 12 meses
rolmean = df_consumption_per_client_norm.rolling(12).mean()*100
rolstd = df_consumption_per_client_norm.rolling(12).std()*100

# Plot rolling statistic
df_consumption_per_client_norm = df_consumption_per_client_norm*100
original = df_consumption_per_client_norm.plot(color = 'blue', label = 'Original')
mean = rolmean.plot(color = 'red', label = 'Media móvil')
std = rolstd.plot(color = 'black', label = 'Desviación estándar')
plt.legend(loc='best', fontsize = 16)
plt.title('Media móvil y desviación estándar', fontsize = 24)
plt.xlabel("")
plt.ylabel("Variación porcentual (%)", fontsize = 24)
plt.show(block = False)

# Dickey-Fuller Test de la serie original
from statsmodels.tsa.stattools import adfuller

print("Resultados del  Dickey-Fuller Test:")
dftest = adfuller(df_consumption_per_client_norm)
dfoutput = pd.Series(dftest[0:4], index=['Estadísticas', 'p-value', '#Lags Used', 'Número de observaciones'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)' %key] = value
    
print(dfoutput)


# Tomamos logaritmo
df_consumption_per_client_norm_logScale = np.log(df_consumption_per_client_norm)

movingAverage = df_consumption_per_client_norm_logScale.rolling(12).mean()
movingSTD = df_consumption_per_client_norm_logScale.rolling(12).std()

df_consumption_per_client_norm_logScale.plot(label = "Logaritmo de la serie original")
movingAverage.plot(color = 'red', label = "Media móvil")
plt.xlabel("")
plt.title("Logaritmo de la serie original", fontsize = 24)
plt.legend(loc='best', fontsize = 20)
plt.show(block = False)

datasetLogScaleMinusMovingAverage = df_consumption_per_client_norm_logScale - movingAverage
datasetLogScaleMinusMovingAverage.head(12)
datasetLogScaleMinusMovingAverage.dropna(inplace = True)


from statsmodels.tsa.stattools import adfuller
def test_stationarity(df):
    
    # Rolling statistics
    movingAverage = df.rolling(window = 12).mean()
    movingSTD = df.rolling(window = 12).std()
 
    orig = df.plot(color = 'blue', label = 'Original')
    mean = movingAverage.plot(color = 'red', label = 'Media móvil')
    std = movingSTD.plot(color = 'black', label = 'Desviación estándar')
    plt.legend(loc = 'best', fontsize = 20)
    plt.title('Media móvil y desviación estándar', fontsize = 24)
    plt.show(block = False)
    
    # ADFT
    print('Results of Dickey-Fuller Test: ')
    dftest = adfuller(df, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Estadísticas', 'p-value', '#Lags Used', 'Número de observaciones'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' %key] = value
    print(dfoutput)

test_stationarity(datasetLogScaleMinusMovingAverage)


# Shifting
datasetLogDiffShifting = df_consumption_per_client_norm_logScale - df_consumption_per_client_norm_logScale.shift()
datasetLogDiffShifting.dropna(inplace = True)
test_stationarity(datasetLogDiffShifting)




# ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(datasetLogDiffShifting, nlags = 20)
lag_pacf = pacf(datasetLogDiffShifting, nlags = 20, method = 'ols')

# Plot ACF
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y = 0, linestyle = '--', color = 'gray')
plt.axhline(y = -1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle = '--', color = 'gray')
plt.axhline(y = 1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle = '--', color = 'gray')
plt.title('Autocorrelation function (ACF)', fontsize = 24)

# Plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y = 0, linestyle = '--', color = 'gray')
plt.axhline(y = -1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle = '--', color = 'gray')
plt.axhline(y = 1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle = '--', color = 'gray')
plt.title('PartialAutocorrelation function (PACF)', fontsize = 24)
plt.tight_layout()
plt.show()



# Implementamos el modelo ARIMA
from statsmodels.tsa.arima_model import ARIMA

#AR MODEL
model = ARIMA(df_consumption_per_client_norm_logScale, order=(12,1,2))
results_ARIMA = model.fit(disp = 1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues, color = 'red')
plt.title('RSS: %.4f'%sum((results_ARIMA.fittedvalues - datasetLogDiffShifting)**2), fontsize = 24)



# Hacemos predicciones
results_ARIMA.plot_predict(1,96)
plt.title("Predicción mensual futura", fontsize = 24)
plt.legend(fontsize = 20)
plt.show()
x = results_ARIMA.forecast(steps=120)






# -------------------------------------------------------------------------------------
# -----------------------------------ANÁLISIS DIARIO-----------------------------------
df_total_per_day_norm = df['2011': '2014'].resample('D').sum()
df_total_per_day_norm = df_total_per_day_norm/df_total_per_day_norm.max()

df_total_per_day_norm = df_total_per_day_norm.sum(axis=1)

for i in df_total_per_day_norm.index:
    df_total_per_day_norm[i] = df_total_per_day_norm[i] / df_num_hogares[(df_num_hogares.index.year == i.year) & (df_num_hogares.index.month == i.month)]


df_total_per_day_norm = df_total_per_day_norm*100
df_total_per_day_norm.shape


df_total_per_day_norm.plot()
plt.title("Evolución del consumo eléctrico medio", fontsize = 24)
plt.xlabel("")
plt.ylabel("Variación del consumo eléctrico (%)", fontsize = 20)
plt.show()


from statsmodels.tsa.seasonal import seasonal_decompose
df_descomposicion = seasonal_decompose(df_total_per_day_norm, model='multiplicative')
df_descomposicion.plot()
plt.xlabel("")
plt.show()


# Media móvil con ventana = 365 dias
rolmean = df_total_per_day_norm.rolling(365).mean()
rolstd = df_total_per_day_norm.rolling(365).std()


# Plot rolling statistic
original = df_total_per_day_norm.plot(color = 'blue', label = 'Original')
mean = rolmean.plot(color = 'red', label = 'Media móvil')
std = rolstd.plot(color = 'black', label = 'Desviación estándar')
plt.legend(loc='best', fontsize = 16)
plt.title('Media móvil y desviación estándar', fontsize = 24)
plt.xlabel("")
plt.ylabel("Variación porcentual (%)", fontsize = 24)
plt.show(block = False)



# Dickey-Fuller Test de la serie original
from statsmodels.tsa.stattools import adfuller

print("Resultados del  Dickey-Fuller Test:")
dftest = adfuller(df_total_per_day_norm)
dfoutput = pd.Series(dftest[0:4], index=['Estadísticas', 'p-value', '#Lags Used', 'Número de observaciones'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)' %key] = value
    
print(dfoutput)


# Tomamos logaritmo
df_total_per_day_norm_logScale = np.log(df_total_per_day_norm)

movingAverage = df_total_per_day_norm_logScale.rolling(365).mean()
movingSTD = df_total_per_day_norm_logScale.rolling(365).std()

df_total_per_day_norm_logScale.plot(label = "Logaritmo de la serie original")
movingAverage.plot(color = 'red', label = "Media móvil")
plt.xlabel("")
plt.title("Logaritmo de la serie original", fontsize = 24)
plt.legend(loc='best', fontsize = 20)
plt.show(block = False)

dayLogScaleMinusMovingAverage = df_total_per_day_norm_logScale - movingAverage
dayLogScaleMinusMovingAverage.head(12)
dayLogScaleMinusMovingAverage.dropna(inplace = True)


from statsmodels.tsa.stattools import adfuller
def test_stationarity(df):
    
    # Rolling statistics
    movingAverage = df.rolling(window = 365).mean()
    movingSTD = df.rolling(window = 365).std()
 
    orig = df.plot(color = 'blue', label = 'Original')
    mean = movingAverage.plot(color = 'red', label = 'Media móvil')
    std = movingSTD.plot(color = 'black', label = 'Desviación estándar')
    plt.legend(loc = 'best', fontsize = 20)
    plt.title('Media móvil y desviación estándar', fontsize = 24)
    plt.show(block = False)
    
    # ADFT
    print('Results of Dickey-Fuller Test: ')
    dftest = adfuller(df, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Estadísticas', 'p-value', '#Lags Used', 'Número de observaciones'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' %key] = value
    print(dfoutput)


test_stationarity(dayLogScaleMinusMovingAverage)



# Shifting
dayLogDiffShifting = df_total_per_day_norm_logScale - df_total_per_day_norm_logScale.shift()
dayLogDiffShifting.dropna(inplace = True)
test_stationarity(dayLogDiffShifting)



# ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(dayLogDiffShifting, nlags = 20)
lag_pacf = pacf(dayLogDiffShifting, nlags = 20, method = 'ols')

# Plot ACF
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y = 0, linestyle = '--', color = 'gray')
plt.axhline(y = -1.96/np.sqrt(len(dayLogDiffShifting)), linestyle = '--', color = 'gray')
plt.axhline(y = 1.96/np.sqrt(len(dayLogDiffShifting)), linestyle = '--', color = 'gray')
plt.title('Autocorrelation function (ACF)', fontsize = 24)

# Plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y = 0, linestyle = '--', color = 'gray')
plt.axhline(y = -1.96/np.sqrt(len(dayLogDiffShifting)), linestyle = '--', color = 'gray')
plt.axhline(y = 1.96/np.sqrt(len(dayLogDiffShifting)), linestyle = '--', color = 'gray')
plt.title('PartialAutocorrelation function (PACF)', fontsize = 24)
plt.tight_layout()
plt.show()


from statsmodels.tsa.arima_model import ARIMA

#AR MODEL
model = ARIMA(df_total_per_day_norm_logScale, order=(7,1,2))
results_ARIMA = model.fit(disp = 1)
plt.plot(dayLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues, color = 'red')
plt.title('RSS: %.4f'%sum((results_ARIMA.fittedvalues - dayLogDiffShifting)**2), fontsize = 24)
plt.show()


 # Predecimos Dic2014-Ene2015
results_ARIMA.plot_predict(1430,1492)
plt.title("Predicción diaria futura", fontsize = 24)
plt.legend(fontsize = 20)
plt.show()
x = results_ARIMA.forecast(steps=120)


# ---------------------------------------------------------------------------------------------
# -----------------------------------ANÁLISIS HORARIO------------------------------------------

# En GWh
df_total_per_hour = df.sum(axis = 1)
df_total_per_hour['2014-12'].plot()
plt.show()

df_total_per_hour = df_total_per_hour.resample('h').sum()
df_total_per_hour_dic2014 = df_total_per_hour['2014-12']

from statsmodels.tsa.seasonal import seasonal_decompose
df_descomposicion = seasonal_decompose(df_total_per_hour_dic2014, model='multiplicative')
df_descomposicion.plot()
plt.xlabel("")
plt.show()

# Media móvil con ventana = 24 horas
rolmean = df_total_per_hour_dic2014.rolling(24).mean()
rolstd = df_total_per_hour_dic2014.rolling(24).std()

# Plot rolling statistic
original = df_total_per_hour_dic2014.plot(color = 'blue', label = 'Original')
mean = rolmean.plot(color = 'red', label = 'Media móvil')
std = rolstd.plot(color = 'black', label = 'Desviación estándar')
plt.legend(loc='best', fontsize = 16)
plt.title('Media móvil y desviación estándar', fontsize = 24)
plt.xlabel("")
plt.ylabel("Consumo absoluto (kWh)", fontsize = 24)
plt.show(block = False)

# Dickey-Fuller Test de la serie original
from statsmodels.tsa.stattools import adfuller

print("Resultados del  Dickey-Fuller Test:")
dftest = adfuller(df_total_per_hour_dic2014)
dfoutput = pd.Series(dftest[0:4], index=['Estadísticas', 'p-value', '#Lags Used', 'Número de observaciones'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)' %key] = value
    
print(dfoutput)




# Tomamos logaritmo
df_total_per_hour_dic2014_logScale = np.log(df_total_per_hour_dic2014)

movingAverage = df_total_per_hour_dic2014_logScale.rolling(24).mean()
movingSTD = df_total_per_hour_dic2014_logScale.rolling(24).std()

df_total_per_hour_dic2014_logScale.plot(label = "Logaritmo de la serie original")
movingAverage.plot(color = 'red', label = "Media móvil")
plt.xlabel("")
plt.title("Logaritmo de la serie original", fontsize = 24)
plt.legend(loc='best', fontsize = 20)
plt.show(block = False)

hourLogScaleMinusMovingAverage = df_total_per_hour_dic2014_logScale - movingAverage
hourLogScaleMinusMovingAverage.head(12)
hourLogScaleMinusMovingAverage.dropna(inplace = True)


from statsmodels.tsa.stattools import adfuller
def test_stationarity(df):
    
    # Rolling statistics
    movingAverage = df.rolling(window = 24).mean()
    movingSTD = df.rolling(window = 24).std()
 
    orig = df.plot(color = 'blue', label = 'Original')
    mean = movingAverage.plot(color = 'red', label = 'Media móvil')
    std = movingSTD.plot(color = 'black', label = 'Desviación estándar')
    plt.legend(loc = 'best', fontsize = 20)
    plt.title('Media móvil y desviación estándar', fontsize = 24)
    plt.show(block = False)
    
    # ADFT
    print('Results of Dickey-Fuller Test: ')
    dftest = adfuller(df, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Estadísticas', 'p-value', '#Lags Used', 'Número de observaciones'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' %key] = value
    print(dfoutput)


test_stationarity(hourLogScaleMinusMovingAverage)

# Shifting
hourLogDiffShifting = df_total_per_hour_dic2014_logScale - df_total_per_hour_dic2014_logScale.shift()
hourLogDiffShifting.dropna(inplace = True)
test_stationarity(hourLogDiffShifting)




# ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(hourLogDiffShifting, nlags = 200)
lag_pacf = pacf(hourLogDiffShifting, nlags = 200, method = 'ols')

# Plot ACF
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y = 0, linestyle = '--', color = 'gray')
plt.axhline(y = -1.96/np.sqrt(len(hourLogDiffShifting)), linestyle = '--', color = 'gray')
plt.axhline(y = 1.96/np.sqrt(len(hourLogDiffShifting)), linestyle = '--', color = 'gray')
plt.title('Autocorrelation function (ACF)', fontsize = 24)

# Plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y = 0, linestyle = '--', color = 'gray')
plt.axhline(y = -1.96/np.sqrt(len(hourLogDiffShifting)), linestyle = '--', color = 'gray')
plt.axhline(y = 1.96/np.sqrt(len(hourLogDiffShifting)), linestyle = '--', color = 'gray')
plt.title('PartialAutocorrelation function (PACF)', fontsize = 24)
plt.tight_layout()
plt.show()


from statsmodels.tsa.arima_model import ARIMA

#AR MODEL
#OJO TIEMPO DE EJECUCIÓN (~30min)
model = ARIMA(df_total_per_hour_dic2014_logScale, order=(24,1,2))
results_ARIMA = model.fit(disp = 1)
plt.plot(hourLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues, color = 'red')
plt.title('RSS: %.4f'%sum((results_ARIMA.fittedvalues - hourLogDiffShifting)**2), fontsize = 24)
plt.show()


#Predicciones para la primera semana de enero de 2015
results_ARIMA.plot_predict(700,900)
plt.title("Predicción diaria futura", fontsize = 24)
plt.legend(fontsize = 20)
plt.show()
x = results_ARIMA.forecast(steps=120)