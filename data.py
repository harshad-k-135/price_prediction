nifty = yf.Ticker("^NSEI")
nifty = nifty.history(period='max')
del nifty['Dividends']
del nifty['Stock Splits']
nifty['Tomorrow'] = nifty['Close'].shift(-1)
nifty.dropna(inplace=True)
last_date = nifty.index[-1]
scaler = MinMaxScaler()
nifty_scaled = scaler.fit_transform(nifty[['Open', 'High', 'Low', 'Close', 'Volume', 'Tomorrow']].values)
X = nifty_scaled[:, :-1]
y = nifty_scaled[:, -1]