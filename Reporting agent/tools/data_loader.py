import pandas as pd

def load_and_prepare_data(data_file:str):
    data = pd.read_csv(f"data\\{data_file}")
    data =  data.convert_dtypes()
    data = data.astype({"Sale_Date":"datetime64[ms]"})
    #data['Week_Start'] = data['Sale_Date'].dt.dayofweek
    # print(data.info())
    # print(data.head(5))
    report_componens = {
        "Total_Sales": data['Sales_Amount'].sum(),
        "Average_Sales": data['Sales_Amount'].mean(),
        "Total_Transactions": data.count(),
        "Top Region": dict(data.groupby('Region')['Sales_Amount'].sum().sort_values(ascending=False).head(1)),
        "Top Category": dict(data.groupby('Product_Category')['Sales_Amount'].sum().sort_values(ascending=False).head(1)),
        "Top Product": dict(data.groupby('Product_ID')['Sales_Amount'].sum().sort_values(ascending=False).head(1)),
        "Sales by Category": dict(data.groupby('Product_Category')['Sales_Amount'].sum().sort_values(ascending=False)),
        "Sales by Region": dict(data.groupby('Region')['Sales_Amount'].sum().sort_values(ascending=False))
    }
    stats = data.describe()
    print(stats)
    return data,stats,report_componens
    
load_and_prepare_data("sales_data.csv")
