import pandas as pd

def test_data_loading():
    df = pd.read_csv('data/insurance_data.txt', sep='|')
    assert not df.empty, "Dataframe is empty"
    assert 'TotalPremium' in df.columns, "TotalPremium column missing"