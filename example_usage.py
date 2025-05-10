from ensemble_hoax_news_detector import predict_hoax, explain_prediction

news = "Pemerintah mengumumkan vaksin baru yang diklaim 100% efektif."
result = predict_hoax(news)
print(result)

# Jalankan LIME
exp = explain_prediction(news)
exp.show_in_notebook()