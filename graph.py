plt.figure(figsize=(12, 6))
plt.plot(nifty.index, nifty['Close'], label='Actual Prices')
plt.plot([nifty.index[-1], nifty.index[-1] + pd.DateOffset(days=1)], 
         [nifty['Close'].iloc[-1], predicted_price_next_day], 
         color='red', marker='o', linestyle='--', label='Predicted Price (Next Day)')

plt.title('Actual Prices and Predicted Price for Next Day')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()