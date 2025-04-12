import pandas as pd
import datetime
import os
import urllib
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

class VHIAnalysis:
    def __init__(self, base_folder='data_csv'):
        self.base_folder = base_folder
        self.lavender_malina_cmap = LinearSegmentedColormap.from_list(
            "lavender_malina", ["#E6E6FA", "#D5006D"], N=256)
        self.regions_true_id = {
            1: 'Вінницька',  2: 'Волинська',  3: 'Дніпропетровська',  4: 'Донецька',  5: 'Житомирська',
            6: 'Закарпатська',  7: 'Запорізька',  8: 'Івано-Франківська',  9: 'Київська',  10: 'Кіровоградська',
            11: 'Луганська',  12: 'Львівська',  13: 'Миколаївська',  14: 'Одеська',  15: 'Полтавська',
            16: 'Рівненська',  17: 'Сумська',  18: 'Тернопільська',  19: 'Харківська',  20: 'Херсонська',
            21: 'Хмельницька',  22: 'Черкаська',  23: 'Чернівецька',  24: 'Чернігівська',  25: 'Республіка Крим'
        }
        self.region_names_to_id = {v: k for k, v in self.regions_true_id.items()}
        self.columns = ['Year', 'Week', 'SMN', 'SMT', 'VCI', 'TCI', 'VHI', 'empty']
    
    def load_data(self, province_code):
        if not os.path.isdir(self.base_folder):
            os.makedirs(self.base_folder)

        current_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        file_name = f'vhi_id__{province_code}__{current_date}.csv'
        file_path = os.path.join(self.base_folder, file_name)

        url = f"https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/get_TS_admin.php?country=UKR&provinceID={province_code}&year1=1981&year2=2025&type=Mean"

        files_in_folder = [f for f in os.listdir(self.base_folder) if f.startswith(f'vhi_id__{province_code}__')]
        if files_in_folder:
            print(f"Data for province {province_code} already downloaded. Skipping download.")
            return

        try:
            response = urllib.request.urlopen(url)
            data = response.read()
            with open(file_path, 'wb') as file:
                file.write(data)
            print(f"VHI data successfully downloaded to file: {file_path}")
        except Exception as e:
            print(f"Failed to download data for province {province_code}: {e}")
    
    def data_frame(self):
        def process_csv(file_path):
            filename = os.path.basename(file_path)
            region_num = int(filename.split('_')[2])
            data = pd.read_csv(file_path, header=1, names=self.columns)
            data.at[0, 'Year'] = data.at[0, 'Year'][9:]
            data = data.drop(data.index[-1])
            data = data[data['VHI'] != -1]
            data = data.drop(columns=['empty'])
            data.insert(0, 'region_num', region_num, True)
            data['Week'] = data['Week'].astype(int)
            return data

        csv_files = glob.glob(f"{self.base_folder}/*.csv")
        data_frames = [process_csv(file) for file in csv_files]

        combined_data = pd.concat(data_frames).drop_duplicates().reset_index(drop=True)
        combined_data = combined_data[~combined_data.region_num.isin([12, 20])]

        region_translation = {
            1: 22, 2: 24, 3: 23, 4: 25, 5: 3, 6: 4, 7: 8, 8: 19, 9: 20, 10: 21,
            11: 9, 13: 10, 14: 11, 15: 12, 16: 13, 17: 14, 18: 15, 19: 16, 21: 17,
            22: 18, 23: 6, 24: 1, 25: 2, 26: 6, 27: 5
        }
        combined_data['region_num'] = combined_data['region_num'].replace(region_translation)

        return combined_data

    def filter_data(self, df, years_interval, weeks_interval, region, sort_option, parameter):
        region_num_selected = self.region_names_to_id.get(region)
        filtered_data = df[
            (df["Year"].between(years_interval[0], years_interval[1])) & 
            (df['Week'].between(weeks_interval[0], weeks_interval[1])) & 
            (df['region_num'] == region_num_selected)
        ]

        if sort_option == "За зростанням":
            filtered_data = filtered_data.sort_values(by=parameter, ascending=True)
        elif sort_option == "За спаданням":
            filtered_data = filtered_data.sort_values(by=parameter, ascending=False)

        return filtered_data

    def plot_line_chart(self, data, parameter):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x="Week", y=parameter, data=data, ax=ax)
        ax.set_title(f"{parameter} по тижнях")
        return fig

    def plot_comparison_chart(self, df, parameter, region, years_interval):
        region_num = self.region_names_to_id.get(region)
        subset = df[(df['region_num'] == region_num) & df['Year'].between(*years_interval)]
        grouped = subset.groupby('Year')[parameter].mean().reset_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='Year', y=parameter, data=grouped, ax=ax, palette="plasma")
        ax.set_title(f"Середній {parameter} по роках у {region}")
        plt.xticks(rotation=45)
        return fig

    def run_analysis(self):
        st.set_page_config(layout="wide")
        st.markdown("<h1 style='color:#800040'> Аналіз Vegetation Health Index (VHI)</h1>", unsafe_allow_html=True)

        df = self.data_frame()
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df = df.dropna(subset=['Year'])

        col1, col2 = st.columns([1, 3])

        with col1:
            st.subheader("Фільтри")
            view_option = st.radio("Вид відображення", ["Таблиця", "Лінійний графік", "Порівняння"])
            parameter = st.selectbox("Параметр для аналізу", ["VCI", "TCI", "VHI"])
            region = st.selectbox("Регіон України", list(self.regions_true_id.values()))
            years_interval = st.slider("Інтервал років", 1981, 2025, (1981, 2025))
            weeks_interval = st.slider("Інтервал тижнів", 1, 52, (1, 52))
            sort_option = st.radio("Сортування", ["Без сортування", "За зростанням", "За спаданням"])

            if st.button("Скинути фільтри"):
                st.experimental_rerun()

        with col2:
            filtered_data = self.filter_data(df, years_interval, weeks_interval, region, sort_option, parameter)

            if filtered_data.empty:
                st.warning("Дані за обраними фільтрами відсутні. Спробуйте змінити параметри.")
            else:
                if view_option == "Таблиця":
                    st.dataframe(filtered_data, use_container_width=True)
                elif view_option == "Лінійний графік":
                    fig = self.plot_line_chart(filtered_data, parameter)
                    st.pyplot(fig)
                elif view_option == "Порівняння":
                    fig = self.plot_comparison_chart(df, parameter, region, years_interval)
                    st.pyplot(fig)

# Запуск додатку
if __name__ == "__main__":
    analysis = VHIAnalysis()
    analysis.run_analysis()
