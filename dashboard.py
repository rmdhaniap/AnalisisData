import sys
!{sys.executable} -m pip install matplotlib

import pandas as pd 
import numpy as np  
#import matplotlib.pyplot as plt  
#import seaborn as sns  
#sns.set_style("whitegrid")
from datetime import datetime
#import scipy.stats as stats
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import streamlit as st
from PIL import Image

# Set style seaborn
#sns.set(style='dark')

day_df = pd.read_csv("day.csv")
day_df.head()

# Mengubah nama judul kolom
day_df.rename(columns={
    'dteday': 'dateday',
    'yr': 'year',
    'mnth': 'month',
    'weathersit': 'weather_cond',
    'cnt': 'count'
}, inplace=True)

# Mengubah angka menjadi keterangan
day_df['month'] = day_df['month'].map({
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
})
day_df['season'] = day_df['season'].map({
    1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'
})
day_df['weekday'] = day_df['weekday'].map({
    0: 'Sun', 1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat'
})
day_df['weather_cond'] = day_df['weather_cond'].map({
    1: 'Clear/Partly Cloudy',
    2: 'Misty/Cloudy',
    3: 'Light Snow/Rain',
    4: 'Severe Weather'
})

# Menyiapkan daily_rent_df
def create_daily_rent_df(df):
    daily_rent_df = df.groupby(by='dateday').agg({
        'count': 'sum'
    }).reset_index()
    return daily_rent_df

# Konversi kolom tanggal ke format datetime
day_df['dateday'] = pd.to_datetime(day_df['dateday'])

# Sidebar untuk memilih rentang tanggal
st.sidebar.header("Filter Tanggal")
min_date = day_df['dateday'].min()
max_date = day_df['dateday'].max()

start_date, end_date = st.sidebar.date_input(
    "Pilih Rentang Tanggal",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Menyiapkan daily_casual_rent_df
def create_daily_casual_rent_df(df):
    daily_casual_rent_df = df.groupby(by='dateday').agg({
        'casual': 'sum'
    }).reset_index()
    return daily_casual_rent_df

# Menyiapkan daily_registered_rent_df
def create_daily_registered_rent_df(df):
    daily_registered_rent_df = df.groupby(by='dateday').agg({
        'registered': 'sum'
    }).reset_index()
    return daily_registered_rent_df
    
# Menyiapkan season_rent_df
def create_season_rent_df(df):
    season_rent_df = df.groupby(by='season')[['registered', 'casual']].sum().reset_index()
    return season_rent_df

# Menyiapkan monthly_rent_df
def create_monthly_rent_df(df):
    monthly_rent_df = df.groupby(by='month').agg({
        'count': 'sum'
    })
    ordered_months = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ]
    monthly_rent_df = monthly_rent_df.reindex(ordered_months, fill_value=0)
    return monthly_rent_df

# Menyiapkan weekday_rent_df
def create_weekday_rent_df(df):
    weekday_rent_df = df.groupby(by='weekday').agg({
        'count': 'sum'
    }).reset_index()
    return weekday_rent_df

# Menyiapkan workingday_rent_df
def create_workingday_rent_df(df):
    workingday_rent_df = df.groupby(by='workingday').agg({
        'count': 'sum'
    }).reset_index()
    return workingday_rent_df

# Menyiapkan holiday_rent_df
def create_holiday_rent_df(df):
    holiday_rent_df = df.groupby(by='holiday').agg({
        'count': 'sum'
    }).reset_index()
    return holiday_rent_df

# Menyiapkan weather_rent_df
def create_weather_rent_df(df):
    weather_rent_df = df.groupby(by='weather_cond').agg({
        'count': 'sum'
    })
    return weather_rent_df


# Membuat komponen filter
min_date = pd.to_datetime(day_df['dateday']).dt.date.min()
max_date = pd.to_datetime(day_df['dateday']).dt.date.max()
 
with st.sidebar:
    st.image("Bike.png")
    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Rentang Waktu',
        min_value= min_date,
        max_value= max_date,
        value=[min_date, max_date]
    )

main_df = day_df[(day_df['dateday'] >= str(start_date)) & 
                (day_df['dateday'] <= str(end_date))]

# Menyiapkan berbagai dataframe
daily_rent_df = create_daily_rent_df(main_df)
daily_casual_rent_df = create_daily_casual_rent_df(main_df)
daily_registered_rent_df = create_daily_registered_rent_df(main_df)
season_rent_df = create_season_rent_df(main_df)
monthly_rent_df = create_monthly_rent_df(main_df)
weekday_rent_df = create_weekday_rent_df(main_df)
workingday_rent_df = create_workingday_rent_df(main_df)
holiday_rent_df = create_holiday_rent_df(main_df)
weather_rent_df = create_weather_rent_df(main_df)


#Tampilan Dashboard full

# Membuat jumlah penyewaan harian
st.subheader('Daily Rentals')
col1, col2, col3 = st.columns(3)

with col1:
    daily_rent_casual = daily_casual_rent_df['casual'].sum()
    st.metric('Casual User', value= daily_rent_casual)

with col2:
    daily_rent_registered = daily_registered_rent_df['registered'].sum()
    st.metric('Registered User', value= daily_rent_registered)
 
with col3:
    daily_rent_total = daily_rent_df['count'].sum()
    st.metric('Total User', value= daily_rent_total)

# Filter data berdasarkan rentang tanggal yang dipilih
filtered_df = day_df[(day_df['dateday'] >= pd.to_datetime(start_date)) & (day_df['dateday'] <= pd.to_datetime(end_date))]

# Visualisasi data penyewaan harian
st.subheader("Analisis Peminjaman Sepeda Harian")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x=filtered_df['dateday'], y=filtered_df['count'], marker="o", ax=ax, color='b')
ax.set_xlabel("Tanggal")
ax.set_ylabel("Jumlah Peminjaman")
ax.set_title("Tren Peminjaman Sepeda Harian")
plt.xticks(rotation=45)
st.pyplot(fig)

# Statistik deskriptif
st.subheader("Statistik Peminjaman Harian")
st.write(filtered_df[['dateday', 'count']].describe())

# Membuat jumlah penyewaan bulanan
st.subheader('Monthly Rentals')
fig, ax = plt.subplots(figsize=(24, 8))
ax.plot(
    monthly_rent_df.index,
    monthly_rent_df['count'],
    marker='o', 
    linewidth=2,
    color='tab:blue'
)

for index, row in enumerate(monthly_rent_df['count']):
    ax.text(index, row + 1, str(row), ha='center', va='bottom', fontsize=12)

ax.tick_params(axis='x', labelsize=25, rotation=45)
ax.tick_params(axis='y', labelsize=20)
st.pyplot(fig)

# Membuah jumlah penyewaan berdasarkan kondisi cuaca
st.subheader('Weatherly Rentals')

fig, ax = plt.subplots(figsize=(16, 8))

colors=["tab:blue", "tab:orange", "tab:green"]

sns.barplot(
    x=weather_rent_df.index,
    y=weather_rent_df['count'],
    palette=colors,
    ax=ax
)

for index, row in enumerate(weather_rent_df['count']):
    ax.text(index, row + 1, str(row), ha='center', va='bottom', fontsize=12)

ax.set_xlabel(None)
ax.set_ylabel(None)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=15)
st.pyplot(fig)

# Visualisasi perbedaan penggunaan sepeda antara hari kerja dan akhir pekan
st.subheader("Perbandingan Penggunaan Sepeda antara Hari Kerja dan Akhir Pekan")
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=['Weekend', 'Weekday'], y=[
    day_df[day_df['workingday'] == 0]['count'].mean(),  # Rata-rata jumlah penyewaan di akhir pekan
    day_df[day_df['workingday'] == 1]['count'].mean()   # Rata-rata jumlah penyewaan di hari kerja
], palette=["red", "blue"], ax=ax)
ax.set_xlabel("Hari")
ax.set_ylabel("Rata-rata Jumlah Penyewaan Sepeda")
ax.set_title("Perbandingan Penggunaan Sepeda antara Hari Kerja dan Akhir Pekan")
st.pyplot(fig)

# Visualisasi tren penggunaan sepeda dari tahun ke tahun
st.subheader("Tren Penggunaan Sepeda dari Tahun ke Tahun")
fig, ax = plt.subplots(figsize=(8, 5))
yearly_trend = day_df.groupby('year')['count'].sum().reset_index()
yearly_trend['year'] = yearly_trend['year'].map({0: 2011, 1: 2012})  # Mapping tahun
sns.lineplot(x=yearly_trend['year'], y=yearly_trend['count'], marker='o', color='green', ax=ax)
ax.set_xlabel("Tahun")
ax.set_ylabel("Total Penyewaan Sepeda")
ax.set_title("Tren Penggunaan Sepeda dari Tahun ke Tahun")
st.pyplot(fig)

# Visualisasi perbandingan pengguna casual dan registered
st.subheader("Perbandingan Jumlah Pengguna Casual dan Registered")
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=['Casual', 'Registered'], y=[day_df['casual'].sum(), day_df['registered'].sum()], palette=["purple", "cyan"], ax=ax)
ax.set_xlabel("Jenis Pengguna")
ax.set_ylabel("Total Penyewaan Sepeda")
ax.set_title("Perbandingan Jumlah Pengguna Casual dan Registered")
st.pyplot(fig)

# Analisis RFM (Recency, Frequency, Monetary) pada pengguna terdaftar
st.subheader("Analisis RFM")
rfm_df = day_df.groupby('dateday').agg({'registered': ['sum', 'count', 'mean']})
rfm_df.columns = ['Monetary', 'Frequency', 'Recency']
rfm_df.reset_index(inplace=True)
st.write(rfm_df.describe())

# Analisis Clustering dengan PCA untuk melihat pola penggunaan
st.subheader("Analisis PCA untuk Clustering")
features = ['temp', 'hum', 'windspeed', 'count']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(day_df[features])
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], alpha=0.5, color='brown', ax=ax)
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_title("Visualisasi PCA untuk Pola Penggunaan Sepeda")
st.pyplot(fig)

st.caption('Copyright © Ramadhani Ari Putra')
