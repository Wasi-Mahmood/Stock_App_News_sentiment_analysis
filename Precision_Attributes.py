import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler



def get_precision_attributes(n_days, forecasted_prices_df, true_prices_df):
    
    scaler = MinMaxScaler(feature_range=(0, 1))

    try:
        true_prices_df =true_prices_df.reset_index(drop=True)
    except:
        pass
    
    true_prices = true_prices_df[n_days+1:].iloc[:,0].values
    
    try:
        forecasted_prices = forecasted_prices_df.reset_index(drop =True).values
    except:
        pass
    
    try:
        forecasted_prices= scaler.inverse_transform(np.reshape(forecasted_prices, (-1, 1)))
        true_prices = scaler.inverse_transform(np.reshape(true_prices, (-1, 1)))
    except:
        pass
    
    # Define a deviation threshold (you can adjust this based on your specific requirements)
    deviation_threshold = 0.6

    # Calculate the absolute difference between forecasted and true prices
    price_difference = np.abs(forecasted_prices - true_prices)

    # Determine whether the predictions are within the deviation limit or not
    within_deviation_limit = price_difference <= deviation_threshold

    # Convert boolean values to binary labels (True: 1, False: 0)
    predicted_labels = within_deviation_limit.astype(int)
    true_labels = np.ones_like(predicted_labels)

    # Compute the confusion matrix
    matrix_confusion = confusion_matrix(true_labels, predicted_labels)

    # Compute F1 score
    f1 = f1_score(true_labels, predicted_labels)


    # Create a DataFrame to visualize the confusion matrix
    confusion_df = pd.DataFrame(matrix_confusion, index=['True', 'False'], columns=['Predicted True', 'Predicted False'])
    
    return confusion_df, f1
    