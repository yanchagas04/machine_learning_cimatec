import pandas as pd
import data_profiling
from data_profiling import ProfileReport

df = pd.read_csv("data/raw/olist_orders_dataset.csv")

profile = ProfileReport(df, title="YData Profiling Report", explorative=True)

profile.to_file("reports/orders_report.html")
