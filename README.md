# Forecast
Forecast model using Prophet in Jupyter Notebooks



```python
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
```

```python
from fbprophet import Prophet
```

## The Data
     Transactions at advertised locations for the last 2 years, by Day &amp; Week. This data will be used to forecast out weekly Transactions to compare last year to.


```python
data = pd.read_csv('Sales_by_day.csv', parse_dates=['ds'])
txns = pd.DataFrame(data[['ds','# Txns']])
txns['y'] = np.log10(txns['# Txns'])
del txns['# Txns']
txns.head()
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-02-01</td>
      <td>4.721918</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-02-02</td>
      <td>4.709092</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-02-03</td>
      <td>4.673058</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-02-04</td>
      <td>4.684917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-02-05</td>
      <td>4.679173</td>
    </tr>
  </tbody>
</table>
</div>




```python
#forecasting tool used to fit model & plot data
m = Prophet()
m.fit(txns)
```






    <fbprophet.forecaster.Prophet at 0x7f0d14b3db00>




```python
#create future dataframe with number of periods to be forecasted
future = m.make_future_dataframe(periods=336)
future.describe()
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1061</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>1061</td>
    </tr>
    <tr>
      <th>top</th>
      <td>2015-07-19 00:00:00</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
    </tr>
    <tr>
      <th>first</th>
      <td>2015-02-01 00:00:00</td>
    </tr>
    <tr>
      <th>last</th>
      <td>2018-01-27 00:00:00</td>
    </tr>
  </tbody>
</table>
</div>



## Initial Forecast
    No additional information considered here i.e., Events or Holidays


```python
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>yhat</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1056</th>
      <td>2018-01-23</td>
      <td>4.622527</td>
      <td>4.564473</td>
      <td>4.683815</td>
    </tr>
    <tr>
      <th>1057</th>
      <td>2018-01-24</td>
      <td>4.660952</td>
      <td>4.602693</td>
      <td>4.723059</td>
    </tr>
    <tr>
      <th>1058</th>
      <td>2018-01-25</td>
      <td>4.663315</td>
      <td>4.605625</td>
      <td>4.720972</td>
    </tr>
    <tr>
      <th>1059</th>
      <td>2018-01-26</td>
      <td>4.723883</td>
      <td>4.663065</td>
      <td>4.788053</td>
    </tr>
    <tr>
      <th>1060</th>
      <td>2018-01-27</td>
      <td>4.730171</td>
      <td>4.669968</td>
      <td>4.788889</td>
    </tr>
  </tbody>
</table>
</div>




```python
%matplotlib inline
m.plot(forecast);
```


![png](https://github.com/SpencerBGuy/Forecast/blob/master/Plots/output_9_0.png)



```python
m.plot_components(forecast);
```


![png](https://github.com/SpencerBGuy/Forecast/blob/master/Plots/output_10_0.png)



```python
forecast.tail()
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>trend</th>
      <th>trend_lower</th>
      <th>trend_upper</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
      <th>seasonal</th>
      <th>seasonal_lower</th>
      <th>seasonal_upper</th>
      <th>seasonalities</th>
      <th>seasonalities_lower</th>
      <th>seasonalities_upper</th>
      <th>weekly</th>
      <th>weekly_lower</th>
      <th>weekly_upper</th>
      <th>yearly</th>
      <th>yearly_lower</th>
      <th>yearly_upper</th>
      <th>yhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1056</th>
      <td>2018-01-23</td>
      <td>4.707945</td>
      <td>4.689375</td>
      <td>4.727012</td>
      <td>4.564473</td>
      <td>4.683815</td>
      <td>-0.085418</td>
      <td>-0.085418</td>
      <td>-0.085418</td>
      <td>-0.085418</td>
      <td>-0.085418</td>
      <td>-0.085418</td>
      <td>-0.030828</td>
      <td>-0.030828</td>
      <td>-0.030828</td>
      <td>-0.054590</td>
      <td>-0.054590</td>
      <td>-0.054590</td>
      <td>4.622527</td>
    </tr>
    <tr>
      <th>1057</th>
      <td>2018-01-24</td>
      <td>4.707895</td>
      <td>4.689229</td>
      <td>4.727059</td>
      <td>4.602693</td>
      <td>4.723059</td>
      <td>-0.046943</td>
      <td>-0.046943</td>
      <td>-0.046943</td>
      <td>-0.046943</td>
      <td>-0.046943</td>
      <td>-0.046943</td>
      <td>0.003878</td>
      <td>0.003878</td>
      <td>0.003878</td>
      <td>-0.050821</td>
      <td>-0.050821</td>
      <td>-0.050821</td>
      <td>4.660952</td>
    </tr>
    <tr>
      <th>1058</th>
      <td>2018-01-25</td>
      <td>4.707845</td>
      <td>4.689044</td>
      <td>4.727106</td>
      <td>4.605625</td>
      <td>4.720972</td>
      <td>-0.044531</td>
      <td>-0.044531</td>
      <td>-0.044531</td>
      <td>-0.044531</td>
      <td>-0.044531</td>
      <td>-0.044531</td>
      <td>0.002379</td>
      <td>0.002379</td>
      <td>0.002379</td>
      <td>-0.046910</td>
      <td>-0.046910</td>
      <td>-0.046910</td>
      <td>4.663315</td>
    </tr>
    <tr>
      <th>1059</th>
      <td>2018-01-26</td>
      <td>4.707795</td>
      <td>4.688879</td>
      <td>4.727151</td>
      <td>4.663065</td>
      <td>4.788053</td>
      <td>0.016087</td>
      <td>0.016087</td>
      <td>0.016087</td>
      <td>0.016087</td>
      <td>0.016087</td>
      <td>0.016087</td>
      <td>0.059033</td>
      <td>0.059033</td>
      <td>0.059033</td>
      <td>-0.042945</td>
      <td>-0.042945</td>
      <td>-0.042945</td>
      <td>4.723883</td>
    </tr>
    <tr>
      <th>1060</th>
      <td>2018-01-27</td>
      <td>4.707746</td>
      <td>4.688721</td>
      <td>4.727194</td>
      <td>4.669968</td>
      <td>4.788889</td>
      <td>0.022425</td>
      <td>0.022425</td>
      <td>0.022425</td>
      <td>0.022425</td>
      <td>0.022425</td>
      <td>0.022425</td>
      <td>0.061438</td>
      <td>0.061438</td>
      <td>0.061438</td>
      <td>-0.039013</td>
      <td>-0.039013</td>
      <td>-0.039013</td>
      <td>4.730171</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Get R-squared to measure model variation
stats.corrcoef(x=txns['y'], y=forecast.loc[:724,'yhat'])
```




    array([[ 1.        ,  0.75357705],
           [ 0.75357705,  1.        ]])



## Additional information to Model
    Scratcher Event &amp; Holidays to consider


```python
#read in dates with Scratcher Event & Holidays listed
dates = pd.read_csv('holiday.csv',parse_dates=['Scratcher Holiday','Ad Changepoint','Year Changepoint','Holidays'])
#set up different event/holiday dataframes, their 'impact' windows and combine sets
scratch_hol = pd.DataFrame({
    'holiday':'holiday',
    'ds': pd.to_datetime(dates['Scratcher Holiday'].dropna()),
    'lower_window':0,
    'upper_window':7,
})
reg_hol = pd.DataFrame({
    'holiday':'holiday',
    'ds': pd.to_datetime(dates['Holidays'].dropna()),
    'lower_window':-2,
    'upper_window':2,
})
new_dates = pd.DataFrame({
    'ds':pd.to_datetime('2017-11-23'), 
    'holiday':'holiday', 
    'lower_window':-2, 
    'upper_window':2,},
    index=[21])
reg_hol = reg_hol.append(new_dates)

holidays = scratch_hol.append(reg_hol)
holidays.head()
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>holiday</th>
      <th>lower_window</th>
      <th>upper_window</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-12-14</td>
      <td>holiday</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-12-02</td>
      <td>holiday</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-11-09</td>
      <td>holiday</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-10-14</td>
      <td>holiday</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-08-12</td>
      <td>holiday</td>
      <td>0</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



## Forecast fitted with Events


```python
m1 = Prophet(holidays=holidays, holidays_prior_scale=.25)
forecast1 = m1.fit(txns).predict(future)
m1.plot(forecast1);
```




![png](https://github.com/SpencerBGuy/Forecast/blob/master/Plots/output_16_1.png)



```python
m1.plot_components(forecast1);
```


![png](https://github.com/SpencerBGuy/Forecast/blob/master/Plots/output_17_0.png)



```python
forecast1.to_csv('fcst_daily_txns.csv')
txns.to_csv('data_daily_txns.csv')
```


```python
#Get R-squared to measure model variation
stats.corrcoef(x=txns['y'], y=forecast1.loc[:724,'yhat'])
```




    array([[ 1.        ,  0.79451865],
           [ 0.79451865,  1.        ]])



## Data by Week


```python
week_data = pd.read_csv('week_data.csv', parse_dates=['ds'])
week_data.describe()
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th># Txns</th>
      <th>Sales $</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>112.000000</td>
      <td>1.120000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>387623.285714</td>
      <td>2.389304e+07</td>
    </tr>
    <tr>
      <th>std</th>
      <td>47703.433555</td>
      <td>7.113570e+06</td>
    </tr>
    <tr>
      <th>min</th>
      <td>316991.000000</td>
      <td>1.529568e+07</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>363127.750000</td>
      <td>2.034617e+07</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>375542.500000</td>
      <td>2.195671e+07</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>398826.000000</td>
      <td>2.475855e+07</td>
    </tr>
    <tr>
      <th>max</th>
      <td>596989.000000</td>
      <td>5.983885e+07</td>
    </tr>
  </tbody>
</table>
</div>




```python
week_bskt = week_data[['ds','# Txns','Sales $']]
week_bskt['y'] = week_bskt['Sales $']/week_bskt['# Txns']
del week_bskt['Sales $']
del week_bskt['# Txns']
week_bskt.describe()
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>112.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>60.751429</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.758831</td>
    </tr>
    <tr>
      <th>min</th>
      <td>48.252730</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>56.113656</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>58.333813</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>61.979078</td>
    </tr>
    <tr>
      <th>max</th>
      <td>119.595904</td>
    </tr>
  </tbody>
</table>
</div>




```python
week_txns = week_data[['ds','# Txns']]
week_txns['y'] = np.log10(week_txns['# Txns'])
del week_txns['# Txns']
week_txns.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 112 entries, 0 to 111
    Data columns (total 2 columns):
    ds    112 non-null datetime64[ns]
    y     112 non-null float64
    dtypes: datetime64[ns](1), float64(1)
    memory usage: 1.8 KB



```python
m_week = Prophet(holidays=holidays, changepoints=['2016-01-30','2017-02-04'],changepoint_prior_scale=.25)

m_week.fit(week_txns)

future_week = m_week.make_future_dataframe(periods=44, freq='W-SAT')

fcst_week = m_week.predict(future_week)
m_week.plot(fcst_week);
m_week.plot_components(fcst_week);
```




![png](https://github.com/SpencerBGuy/Forecast/blob/master/Plots/output_24_1.png)



![png](https://github.com/SpencerBGuy/Forecast/blob/master/Plots/output_24_2.png)



```python
fcst_week.tail()
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>trend</th>
      <th>trend_lower</th>
      <th>trend_upper</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
      <th>holiday</th>
      <th>holiday_lower</th>
      <th>holiday_upper</th>
      <th>holidays</th>
      <th>...</th>
      <th>seasonal</th>
      <th>seasonal_lower</th>
      <th>seasonal_upper</th>
      <th>seasonalities</th>
      <th>seasonalities_lower</th>
      <th>seasonalities_upper</th>
      <th>yearly</th>
      <th>yearly_lower</th>
      <th>yearly_upper</th>
      <th>yhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>151</th>
      <td>2017-12-30</td>
      <td>5.560744</td>
      <td>5.559143</td>
      <td>5.561825</td>
      <td>5.574087</td>
      <td>5.630299</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.042402</td>
      <td>0.042402</td>
      <td>0.042402</td>
      <td>0.042402</td>
      <td>0.042402</td>
      <td>0.042402</td>
      <td>0.042402</td>
      <td>0.042402</td>
      <td>0.042402</td>
      <td>5.603146</td>
    </tr>
    <tr>
      <th>152</th>
      <td>2018-01-06</td>
      <td>5.560542</td>
      <td>5.558877</td>
      <td>5.561655</td>
      <td>5.506108</td>
      <td>5.560721</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>-0.025646</td>
      <td>-0.025646</td>
      <td>-0.025646</td>
      <td>-0.025646</td>
      <td>-0.025646</td>
      <td>-0.025646</td>
      <td>-0.025646</td>
      <td>-0.025646</td>
      <td>-0.025646</td>
      <td>5.534896</td>
    </tr>
    <tr>
      <th>153</th>
      <td>2018-01-13</td>
      <td>5.560340</td>
      <td>5.558608</td>
      <td>5.561560</td>
      <td>5.468476</td>
      <td>5.523930</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>-0.063983</td>
      <td>-0.063983</td>
      <td>-0.063983</td>
      <td>-0.063983</td>
      <td>-0.063983</td>
      <td>-0.063983</td>
      <td>-0.063983</td>
      <td>-0.063983</td>
      <td>-0.063983</td>
      <td>5.496357</td>
    </tr>
    <tr>
      <th>154</th>
      <td>2018-01-20</td>
      <td>5.560138</td>
      <td>5.558269</td>
      <td>5.561413</td>
      <td>5.468794</td>
      <td>5.524086</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>-0.063961</td>
      <td>-0.063961</td>
      <td>-0.063961</td>
      <td>-0.063961</td>
      <td>-0.063961</td>
      <td>-0.063961</td>
      <td>-0.063961</td>
      <td>-0.063961</td>
      <td>-0.063961</td>
      <td>5.496177</td>
    </tr>
    <tr>
      <th>155</th>
      <td>2018-01-27</td>
      <td>5.559935</td>
      <td>5.557986</td>
      <td>5.561291</td>
      <td>5.491117</td>
      <td>5.547963</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>-0.041599</td>
      <td>-0.041599</td>
      <td>-0.041599</td>
      <td>-0.041599</td>
      <td>-0.041599</td>
      <td>-0.041599</td>
      <td>-0.041599</td>
      <td>-0.041599</td>
      <td>-0.041599</td>
      <td>5.518336</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
#Get R-squared to measure model variation
stats.corrcoef(x=week_txns['y'], y=fcst_week.loc[:111,'yhat'])
```




    array([[ 1.        ,  0.89732963],
           [ 0.89732963,  1.        ]])


