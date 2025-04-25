# 🚲 Toronto Bike Share Prediction

This project explores how weather conditions influence hourly bike share usage in Toronto. Using a combination of public datasets from [Toronto Open Data Portal](https://open.toronto.ca/dataset/bike-share-toronto-ridership-data/) and the [OpenWeather API](https://openweathermap.org/), I build and compare several predictive models to forecast bike trip volume and identify the most influential weather features.

The project was completed as part of the **JSC370: Introduction to Data Science** course at the University of Toronto (Winter 2025).

🔗 **Project Website**  
👉 [https://christoffertan.github.io/toronto-bikeshare-analysis/](https://christoffertan.github.io/toronto-bikeshare-analysis/)

📄 **Download Report**  
👉 [Full PDF Report](docs/report.pdf)

🎬 **Walkthrough Video**  
👉 [Video](https://utoronto-my.sharepoint.com/:v:/g/personal/christoffer_tan_mail_utoronto_ca/EfCtQox2bz9FqhsjDTdDjqkBGFrFcEL2v4vgwq7oELDKEA?e=PkftRt)

---

## 🔍 Key Features

- **Weather-BikeShare Integration**: Merged hourly weather with bike trip volume for joint analysis.
- **Exploratory Visualizations**: Trends, distributions, and relationships between usage and weather conditions.
- **Predictive Modeling**:
  - Linear Regression
  - GLM (Negative Binomial)
  - GAM (Negative Binomial)
  - Random Forest (tuned and default)
  - XGBoost (tuned and default)
- **Evaluation Metrics**: R², RMSE, and MAE used to compare model performance.
- **Interactive Website**: Key figures and analysis hosted as a dynamic website.

---

## 🙏 Acknowledgements

Special thanks to **Professor Meredith Franklin** for her instruction and feedback in JSC370, and to the open data providers:

- [Bike Share Toronto Ridership Data](https://open.toronto.ca/dataset/bike-share-toronto-ridership-data/)
- [OpenWeather API](https://openweathermap.org/)

---

© 2025 Christoffer Tan • University of Toronto  
Contact: [christoffer.tan@mail.utoronto.ca](mailto:christoffer.tan@mail.utoronto.ca)
