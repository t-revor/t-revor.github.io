import ast
import json
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from statsmodels.tsa.seasonal import seasonal_decompose
warnings.filterwarnings('ignore')
def create_key_list(key_column):
    """ This function creates a list of all the different
        elements present in a column of the dataset """
    key_list = list(key_column.unique())
    return key_list

def create_dict(series, key_list, key_column, values_column):
    """ This function creates a dictionary that has as keys
        the elements present in the passed list and as values
        the list of elements corresponding to the keys in
        another column of the dataset.
        It is necessary to pass the dataset, then the list of keys,
        then the column from which the keys were taken (in the form
        dataset_name.column_name) and finally the columns of the
        desired values (in the form 'column_name') """

    dictionary = {}
    for a_key in key_list:
        dictionary[a_key] = list(sorted(set(series.loc[key_column == a_key, values_column])))
    return dictionary

def clean_dict(a_dictionary, key, products_list=['all']):
    """ This function cleans a dictionary by removing items not
        needed, it can be the whole list of products or just
        some of them.
        It is necessary to pass a dictionary, the key of the
        object that has to be removed. If only some products
        need to be removed then it is necessary also to pass
        a list containing the product or products to eliminate """

    if products_list == ['all']:
        del a_dictionary[key]
    else:
        for a_product in products_list:
            a_dictionary[key].remove(a_product)

def clean_columns(series_column):
    """ This function replaces, in a series column, all the
        commas with dots and then casts the value type to float.
        It just needs the series_column to be passed in this format:
        dataset_name.loc[:,'column_name'] """

    series_column = series_column.map(lambda x: x.replace(",", "."))
    series_column = series_column.astype(float)
    return series_column

def print_key(key_list):
    """ This function prints all the segment available that are stored in a list """

    for a_key in key_list:
        print(a_key)

def print_products(products_list):
    """ This function prints all the products associated to a given segment and
        tells you which number to put in order to select that product"""

    for a_product in products_list:
        print(a_product + ' -> ' + str(products_list.index(a_product)))

def choose_product(key_list, products_dictionary):
    """ This function allows you to choose the product on which the
        analysis will be conducted.
        It takes as argument the dictionary of the products and
        the dictionary's keys list"""

    print_key(key_list)
    chosen_product_key = input("\nPlease, type the segment to be analysed: ")
    print('')
    print_products(products_dictionary[chosen_product_key])
    chosen_product_position = int(input("\nPlease, select the product to be analysed by typing the corresponding number: "))
    chosen_product = products_dictionary[chosen_product_key][chosen_product_position]
    return chosen_product

def single_product_series(orig_series,chosen_product,weeks_to_predict):
    """ This function creates a new time series of the weekly sales
        of a single product.
        It takes as arguments the full time series and the name of
        the chosen product which is stored in the variable
        chosen_product and the number of weeks to predict"""

    is_product = orig_series["Products"] == chosen_product
    product = orig_series.loc[is_product]
    product = product.reset_index(drop=True)
    product["Sales"] = product["Sales"].astype(float)
    product['Date'] = product['Date'].astype('datetime64[ns]')
    fill_date = pd.date_range(start="01/03/2017", end="30/03/2019")
    fill_date_dict = {"Date": fill_date, "Category": "Tamponi", "Segment": product.loc[0, "Segment"],
                      "Products": product.loc[0, "Products"],
                      }
    df_fill_date = pd.DataFrame(data=fill_date_dict)
    product = product.merge(df_fill_date, on=["Date"], how="right")
    product = product.sort_values(by="Date")
    product.loc[:, "Sales"] = product.loc[:, "Sales"].fillna(0)
    product.drop(["Category_x", "Segment_x", "Products_x", "Regione",
                  "Provincia", "Channel"], axis=1, inplace=True)
    product = pd.DataFrame([{'Date': k,
                             'Sales': v.Sales.sum()}
                            for k, v in product.groupby(['Date'])],
                           columns=['Date', 'Sales'])
    product = product.resample('W-Wed',
                               on='Date').sum().reset_index().sort_values(by='Date')
    future_product = pd.date_range(start=product.iloc[-1, 0], periods=weeks_to_predict, freq='W')
    product.set_index('Date', inplace=True)
    return product, future_product

def split_sequence(sequence, n_steps_in, n_steps_out):
    """ This function divides the sequence of values into multiple
        input/output patterns, where n_steps_in indicates the number
        of values in the input, which are the values that will be
        taken to make the predictions, while n_steps_out indicates
        how many points we want to forecast."""

    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def train_test(series, weeks_to_predict):
    """ This function splits the series into train and test set
        according to a percentage.
        It takes as arguments the series and number of weeks to predict."""

    train = series.iloc[:-weeks_to_predict]
    test = series.iloc[-weeks_to_predict:]
    return train, test

def fixed_train(train_series):
    """ This function takes as argument the train_set and casts it to a list"""

    product_to_lstm_train = train_series.tolist()
    return product_to_lstm_train


def define_model(train, product_to_lstm_train, n_steps_in, n_steps_out, n_features, idx, background_signal, forecast_test_list, series_name):
    """ This function defines the model and fits it and then
        it does the prediction. In the end it appends to the
        list forecast_test_list, because we will make run this
        function for n times to get a more robust prediction."""

    # split into samples
    X, y = split_sequence(product_to_lstm_train, n_steps_in, n_steps_out)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps_in, n_features)))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=30, verbose=0)
    x_input = np.asarray(product_to_lstm_train[-n_steps_in:])
    x_input = x_input.reshape((1, n_steps_in, n_features))
    #making the prediction
    forecast_on_test_set = model.predict(x_input, verbose=0)
    forecast_on_test_set_list = [item for item in forecast_on_test_set[0]]
    #taking back the values to the original ones
    background_signal_train = background_signal[0:len(train)]
    forecast_integrated_test = rollback(background_signal_train, forecast_on_test_set_list)
    forecast_series_test = pd.Series(forecast_integrated_test, index=idx, name=series_name)
    #setting to 0 all the negative predictions
    forecast_series_test[forecast_series_test < 0] = 0
    #appending the predictions to the prediction list
    forecast_test_list.append(forecast_series_test.tolist())

def median_list(n_steps_out, forecast_list,median_list):
    """ This function creates the forecast set by taking
        the median of the different forecasts made, which
        are stored in a list that will be passed as one
        of the arguments.
        It takes as argument the number of observation in
        each forecast, the list of all the forecasts, and
        the median_list which is the final forecast list."""

    for j in range(n_steps_out):
        same_timestamp_elements = []
        for i in range(len(forecast_list)):
            same_timestamp_elements.append(forecast_list[i][j])
        median_in_a_timestamp = np.median(same_timestamp_elements)
        median_list.append(median_in_a_timestamp)

def plot_single_product(product, chosen_product):
    """ This function plots the time series of a single product.
        It takes as arguments the product series and the chosen
        product name """

    product.plot()
    plt.title("Time series of " + chosen_product)
    plt.show()

def plot_forecast(product, weeks_to_predict, forecast_series_test, chosen_product):
    """ This function plots first the train set and the forecast
        on the test set, then it plots just the true data of the
        test set and the predicted ones, and finally it plots the
        residuals.
        It takes as arguments the product time series, the number
        of weeks to predict in order to know the division between
        train and test series, the forecast series for the test
        and the name of the product"""

    product[:-weeks_to_predict].plot()
    forecast_series_test.plot()
    plt.title("Train time series and forecasting on the test time series for " + chosen_product)
    plt.show()
    mse = metrics.mean_squared_error(list(product.iloc[-weeks_to_predict:, 0]), forecast_series_test.tolist())
    product[-weeks_to_predict:].plot()
    forecast_series_test.plot()
    plt.title("Comparison of the test series of " + chosen_product + ", with mse = " + str(int(mse)))
    plt.show()
    residuals = np.array(list(product.iloc[-weeks_to_predict:,0])) - np.array(forecast_series_test.tolist())
    residuals_series = pd.Series(residuals, index=forecast_series_test.index, name="Observed minus Predicted")
    residuals_series.plot()
    plt.title("Residuals for " + chosen_product)
    plt.show()

def plot_final_forecast(product, forecast_series, chosen_product):
    """ This function plots the forecast of unknown data.
        It takes as argument the product time series, the forecast
        series of the new data, and the name of the product"""

    product.plot()
    forecast_series.plot()
    plt.title("Time series and new forecasts of the " + chosen_product)
    plt.show()

def get_statistics(series):
    """ This function returns different statistics regarding the time series
        and we use it to access the residuals in the main program"""

    decompose = seasonal_decompose(series, model='additive')
    return decompose

def rollback(series_non_diff_list, forecast_list, interval=52):
    """ This function allows us to get the right values of the time series
        since the forecast is on the residuals.
        It takes as parameters the original series, the list with the forecast
        and the interval that is set to 52 because we suppose a lag of 1 year"""

    list_integrated = []
    for i in range(len(forecast_list)):
        element_integrated = forecast_list[i] + series_non_diff_list[i-interval]
        list_integrated.append(element_integrated)
    return list_integrated

def all_mse(series_tampons_global,segment_products_dict, weeks_to_predict, number_of_rerunnings):
    """ The return of this function is a dictionary having as keys
        the different Products and as values the associated mse. In
        order to build this dictionary the function takes as parameters
        the original time series with all the products so that by iterating
        over the products it is possible to create time series for each of
        them. Then it takes the dictionary (Segment, Products) that is
        used to iterate on all the products, it takes the weeks to predict
        and the number of rerunnings, which means how many times the model
        will be fitted in order to get a more robust prediction """

    mse_dict = {}
    for a_segment in segment_products_dict:
        for i in range(len(segment_products_dict[a_segment])):
            chosen_product = segment_products_dict[a_segment][i]
            product, future_product = single_product_series(series_tampons_global, chosen_product, weeks_to_predict)
            decompose = get_statistics(product)
            background_signal = decompose.seasonal.to_numpy() + decompose.trend.replace(np.nan, 0).to_numpy()
            background_signal = background_signal.tolist()
            decompose.resid[:] = list(np.asarray(product)[:, 0] - np.asarray(background_signal))
            train, test = train_test(decompose.resid, weeks_to_predict)
            product_to_lstm_train = fixed_train(train)
            forecast_test_list = []
            for j in range(number_of_rerunnings):
                define_model(train, product_to_lstm_train, 52, len(test), 1, test.index, background_signal,
                             forecast_test_list, "Sales_forecast_on_test")
            median_series_list = []
            median_list(len(test), forecast_test_list, median_series_list)
            mse = metrics.mean_squared_error(list(product.iloc[-weeks_to_predict:, 0]), median_series_list)
            mse_dict[str(segment_products_dict[a_segment][i])] = mse
    return mse_dict

def store_mse(a_dictionary):
    """ This function takes as paramenter a dictionary and
        saves it in a JSON file"""

    a_file = open("mse_lstm.json", "w")
    json.dump(a_dictionary, a_file)
    a_file.close()

def get_json(json_file):
    """ This function takes as parameter a JSON file, reads
        it, transforms what is inside in a dictionary and
        then it returns the dictionary"""

    a_file = open(json_file, "r")
    output = a_file.read()
    output_dictionary = ast.literal_eval(output)
    a_file.close()
    return output_dictionary

def prepare_data_for_plot(file, ordered_products):
    """ Since dictionaries are not ordered it is necessary
        to do this manually, so this functions takes in the
        JSON file and a tuple containing the products in the
        right order, uses the function 'get_json' to access
        the file and then it orders the dictionary and finally
        it returns the tuple y for the plot, which corresponds
        to the mse of each product"""

    products_mse = get_json(file)
    ordered_dict = {k: products_mse[k] for k in ordered_products}
    y = [ordered_dict[k] for k in ordered_dict.keys()]
    return y

def different_mse_plot(ordered_products, y_var, y_sar, y_lstm, label_var, label_sar, label_lstm):
    """ This function takes in the tuple of ordered products,
        the y vector of ordered mse for each model and the
        different labels for each model"""

    plt.plot(ordered_products, y_var, label = label_var)
    plt.plot(ordered_products, y_sar, label = label_sar)
    plt.plot(ordered_products, y_lstm, label = label_lstm)
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()
