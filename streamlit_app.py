import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model.joblib')

tab1, tab2, tab3, tab4 = st.tabs(['Prediction', 'About', 'Dataset', 'Model Performance'])

with tab1:
    st.title("🚖 Taxi Fares Prediction")
    st.write("Aplikasi Prediksi Tarif Taksi untuk memberikan estimasi harga perjalanan yang transparan dan akurat, sehingga dapat memberikan perkiraan biaya berdasarkan jarak tempuh serta waktu keberangkatan yang dipilih.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        trip_distance = st.number_input("Trip Distance (miles)", min_value=0.55, max_value=180.0, value=25.0, step=0.1)
        pickup_location_id = st.number_input("Pickup Location ID", min_value=1, max_value=265, value=1)
        dropoff_location_id = st.number_input("Dropoff Location ID", min_value=1, max_value=265, value=1)


    with col2:
        passanger_count = st.slider("Passenger Count", min_value=1, max_value=4, value=1, step=1)
        day_of_week = st.selectbox("Day of the Week", options=['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu'])
        time_of_day = st.slider("Time of Day", min_value=0, max_value=24, value=8, step=1)

    if st.button("Predict Fare"):
        if pickup_location_id == dropoff_location_id:
            st.error("Tidak bisa dijemput dan diturunkan di lokasi yang sama.")
        else:
            day_of_week_mapping = {'Senin': 0, 'Selasa': 1, 'Rabu': 2, 'Kamis': 3, 'Jumat': 4, 'Sabtu': 5, 'Minggu': 6}
            
            if time_of_day <= 6:
                time_label = 'Malam_DiniHari'
            elif time_of_day <= 12:
                time_label = 'Pagi'
            elif time_of_day <= 18:
                time_label = 'Siang_Sore'
            else:
                time_label = 'Malam'

            input_df = pd.DataFrame([{
                'trip_distance': trip_distance,
                'pickup_location_id': pickup_location_id,
                'dropoff_location_id': dropoff_location_id,
                'day_of_week': day_of_week_mapping[day_of_week],
                'time_of_day': time_label
            }])
            
            predicted_fare = model.predict(input_df)[0]

            if passanger_count > 1:
                st.markdown(f"""
                <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); text-align: center;">
                    <h3 style="color: #555; margin-bottom: 0;">Estimasi Tarif Perjalanan</h3>
                    <h1 style="color: #2e7d32; margin: 10px 0;">${passanger_count * predicted_fare:.2f}</h1>
                    <h3 style="color: #2e7d32; margin: 0;">(${predicted_fare:.2f} untuk 1 penumpang)</h3>
                    <p style="color: #888; font-size: 16px; margin-top: 0;">*Harga di atas adalah estimasi dasar sebelum pajak dan biaya tol.</p>
                </div>
                """, unsafe_allow_html=True)
            else :
                st.markdown(f"""
                <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); text-align: center;">
                    <h3 style="color: #555; margin-bottom: 0;">Estimasi Tarif Perjalanan</h3>
                    <h1 style="color: #2e7d32; margin: 10px 0;">${predicted_fare:.2f} </h1>
                    <p style="color: #888; font-size: 12px; margin-top: 0;">*Harga di atas adalah estimasi dasar sebelum pajak dan biaya tol.</p>
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.title("ℹ️ About This App")
    st.subheader("Latar Belakang")
    st.write("Dalam industri ride-hailing dan taksi modern, transparansi harga merupakan fondasi utama dalam membangun kepercayaan pelanggan. Sering kali, calon penumpang merasa ragu untuk memesan perjalanan karena ketidakpastian biaya yang harus dibayarkan, terutama saat memasuki jam sibuk atau ketika menempuh rute jarak jauh. Ketidakpastian ini dapat menghambat keputusan pelanggan dan memengaruhi kepuasan mereka terhadap layanan.")
    
    st.subheader("Fitur Utama")
    st.write("Aplikasi ini memungkinkan pelanggan melihat estimasi tarif secara instan di layar smartphone mereka sebelum melakukan pemesanan. Dengan dukungan model Machine Learning berbasis Regresi, fitur ini mampu menghitung tarif secara cerdas berdasarkan:")
    st.write("- Waktu Penjemputan: Menyesuaikan tarif berdasarkan fluktuasi jam sibuk.")
    st.write("- Akurasi Lokasi: Menentukan koordinat penjemputan dan tujuan secara presisi.")
    st.write("- Prediksi Jarak: Mengkalkulasi estimasi harga berdasarkan rute perjalanan yang akan ditempuh.")

    st.subheader("Tujuan Aplikasi")
    st.write("Tujuan utama dari pengembangan fitur ini adalah untuk menghilangkan keraguan pelanggan dengan memberikan informasi biaya yang transparan sejak awal. Dengan adanya estimasi harga yang akurat di smartphone mereka, pelanggan dapat merencanakan perjalanan dengan lebih tenang dan efisien.")

with tab3:
    st.title("🗂️ Dataset")
    st.write("Ada 5 jenis data yang digunakan untuk training model, yaitu:")

    col1t2, col2t2 = st.columns(2)
    
    with col1t2:
        st.write("- Passenger Count")
        st.write("- Trip Distance (miles)")
        st.write("- Pickup Location ID")
    with col2t2:
        st.write("- Dropoff Location ID")
        st.write("- Day of the Week")
        st.write("- Time of Day")
    st.write("Variabel targetnya adalah jumlah tarif taxi.")

    df = pd.read_csv('data/taxi_fare.csv')
    st.subheader("Data Mentah (10 Baris Pertama)")
    st.dataframe(df.head(10))

    st.subheader("Distribusi Fare Amount")
    st.image("./assets/fare_amount_distribution.png", caption="Distribusi Jumlah Tarif Taxi")

    st.subheader("Perbandingan Trip Distance vs Fare Amount")
    st.image("./assets/trip_distance_vs_fare.png", caption="Perbandingan Jarak Perjalanan vs Tarif Taxi")

    st.subheader("Heatmap Korelasi")
    st.image("./assets/correlation_heatmap.png", caption="Heatmap Korelasi Antar Fitur")

with tab4:
    st.title("📈 Model Performance")
    st.write("Model ini dievaluasi menggunakan metrik Mean Absolute Error (MAE) and R-squared (R²).")

    with open('assets/hasil_evaluasi.txt', 'r') as f:
        isi_file = f.readlines()
        r2_manual = float(isi_file[0].strip())
        rmse_manual = float(isi_file[1].strip())

    col1, col2 = st.columns(2)
    col1.metric("R2 Score", f"{r2_manual:.4f}")
    col2.metric("RMSE", f"{rmse_manual:.2f}")

    st.image("./assets/actual_vs_predicted.png", caption="Actual vs. Predicted Fare Amount")

    st.write("Metrik ini menunjukkan bahwa model tersebut memiliki kinerja yang baik dalam memprediksi tarif taksi berdasarkan fitur-fitur yang diberikan. R² yang tinggi menunjukkan bahwa model mampu menjelaskan sebagian besar variabilitas dalam data, sementara RMSE yang rendah menunjukkan bahwa prediksi model cukup akurat dibandingkan dengan nilai aktual.")