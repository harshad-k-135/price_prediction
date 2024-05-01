last_date_features = X[-1].reshape(1, -1)
predicted_price_next_day = model.predict(last_date_features)
predicted_price_next_day = scaler.inverse_transform(np.concatenate((last_date_features, predicted_price_next_day), axis=1))[:, -1][0]
next_date = last_date + pd.DateOffset(days=1)
if next_date < nifty.index[-1]:
    next_date = nifty.index[-1]

print(f"Predicted price for the next day ({next_date}): {predicted_price_next_day:.4f}")