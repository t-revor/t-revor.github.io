import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from sklearn import metrics
from datetime import datetime,date, timedelta
import json
import warnings
warnings.filterwarnings('ignore')#ignore warnings in jupyter

pd.set_option('display.max_rows', None)# allow you to see the complete dataframe in jupyter




def import_and_clean_historical_promo(Segmento_):
    """import and clean promo dataset for segment of my interest; promo from 2017 january to 2019 march"""
    df_promo = pd.read_csv("tampons_historical_promo_NEW.csv",sep=";")
    df_promo.head()
    promo_segment_ = df_promo[(df_promo.Segment == Segmento_) ]
    promo_segment_.rename(columns={'Inizio Promo':'Inizio_promo','Fine Promo':'Fine_promo'}, inplace=True)
    promo_segment_['Inizio_promo'] =promo_segment_["Inizio_promo"].astype('datetime64[ns]')
    promo_segment_['Fine_promo'] =promo_segment_["Fine_promo"].astype('datetime64[ns]')
    return promo_segment_

def import_and_clean_future_promo(Segmento_):
    """import and clean future promo dataset for segment of my interest; promo from 2019 march to 2019 september"""
    future_promo = pd.read_csv("tampons_promo_planning_NEW.csv",sep ="\t")
    future_promo_segment_ = future_promo[(future_promo["Segment"] == Segmento_)]
    future_promo_segment_.rename(columns={'Inizio Promo':'Inizio_promo','Fine Promo':'Fine_promo'}, inplace=True)
    future_promo_segment_['Inizio_promo'] = future_promo_segment_["Inizio_promo"].astype('datetime64[ns]')
    future_promo_segment_['Fine_promo'] = future_promo_segment_["Fine_promo"].astype('datetime64[ns]')
    return future_promo_segment_

def import_and_clean_historical_series_tampons():
    """import and clean sales of fater tampons from 3 jan. 2017 to 30 march 2019."""
    series_tampons_global = pd.read_csv("historical_series_tampons_NEW.csv",sep ="\t")
    series_tampons_global.rename(columns={'Data Rif':'Data','Standard Units':'Sales','Regione':'Regions','Provincia':'Provinces'}, inplace=True)
    series_tampons_global['Data'] =pd.to_datetime(series_tampons_global.Data,format ='%d/%m/%Y')
    series_tampons_global = series_tampons_global.sort_values(by='Data')
    series_tampons_global.loc[:,"Sales"] = clean_columns(series_tampons_global.loc[:,"Sales"])
    series_tampons_global = series_tampons_global.reset_index(drop=True)
    return series_tampons_global

def clean_columns(Sales):
    """Clean column 'Sales' of dataframe series_tampons_global """
    Sales = Sales.map(lambda x: x.replace(",","."))
    Sales = Sales.astype(float)
    return Sales

def historical_series_specific_product(series_tampons_global,Prodotto_):
    """subsample sales for a specific product 'Prodotto_'"""
    is_product = series_tampons_global["Products"] == Prodotto_
    product = series_tampons_global.loc[is_product]
    product = product.reset_index(drop=True)
    return product

def fill_with_zero_the_sales_of_missing_dates(product):
    """fill missing dates of dataframe product setting sales equal to 0"""
    product["Sales"] = product["Sales"].astype(float)
    product['Data'] = product['Data'].astype('datetime64[ns]')
    fill_date = pd.date_range(start = "01/03/2017", end = "30/03/2019")
    fill_date_dict = {"Data":fill_date,"Category":"Tamponi","Segment":product.loc[0,"Segment"],
                      "Products":product.loc[0,"Products"],
                      #"Sales":product.loc[0,"Sales"]
                     }
    df_fill_date = pd.DataFrame(data = fill_date_dict)
    product = product.merge(df_fill_date,on = ["Data"],how = "right")
    product = product.sort_values(by = "Data")
    product.loc[:,"Sales"] = product.loc[:,"Sales"].fillna(0)
    product.drop(["Category_x","Segment_x","Products_x","Regions",
                   "Provinces","Channel"],axis = 1,inplace = True)
    return product

def aggregate_customers_and_squeeze_series_on_weeks(product):
    """aggregate the sales on overall customers and squueze the series on weeks"""
    product = pd.DataFrame([{'Data': k,
                            'Sales': v.Sales.sum()}
                           for k,v in product.groupby(['Data'])],
                          columns=['Data', 'Sales'])
    weekly_data = product.resample('W-Wed', on='Data').sum().reset_index().sort_values(by='Data')
    return weekly_data

def count_true(mask):
    """count the number of True in 'mask'"""
    number_of_true = 0
    for bool in mask:
        if bool == True:
            number_of_true += 1
    return number_of_true

def generate_array_of_booleans_for_promotions(weekly_data,promo_segment_,number_of_customer):
    """generate a series of booleans where 1 means that a week before sell in the ten per cent
    of customers used promos on that segment,0 otherwise. It concernes until march 2019"""
    Is_promo = []
    for i in range(len(weekly_data)):
        data_verifica_promo = weekly_data["Data"][i:i+1]
        data_verifica_promo = data_verifica_promo -  timedelta(7)
        data_verifica_promo = list(map(str, data_verifica_promo)) #mappo in una stringa
        mask = (data_verifica_promo[0][0:10] > promo_segment_["Inizio_promo"]) & (data_verifica_promo[0][0:10] < promo_segment_["Fine_promo"])
        number_of_true = count_true(mask)
        if number_of_true >= 0.1*number_of_customer:
            Is_promo.append(1)
        else : Is_promo.append(0)
    Is_promo = pd.Series(Is_promo)
    return Is_promo

def append_array_of_booleans_for_future_promotions(Is_promo,future_weekly_dates,future_promo_segment_cust,number_of_customer):
    """generate a series of booleans where 1 means that a week before sell in the ten per cent
    of customers used promos on that segment,0 otherwise. It concernes from march 2019 to september 2019"""
    Is_promo_copy = Is_promo[:].tolist()
    for i in range(len(future_weekly_dates)):
        data_verifica_promo = future_weekly_dates[i:i+1]
        data_verifica_promo = data_verifica_promo -  timedelta(7)
        data_verifica_promo = list(map(str, data_verifica_promo))
        mask = (data_verifica_promo[0][0:10] > future_promo_segment_cust["Inizio_promo"]) & (data_verifica_promo[0][0:10] < future_promo_segment_cust["Fine_promo"])
        number_of_true = count_true(mask)
        if number_of_true >= 0.1*number_of_customer:
            Is_promo_copy.append(1)
        else : Is_promo_copy.append(0)
    Is_promo_copy = pd.Series(Is_promo_copy)
    return Is_promo_copy


def clean_sell_out(sell_out):
    """cleaning operations on dataframe of sells out"""
    sell_out["week"] = sell_out["week"].map(lambda x: x.lstrip("W "))
    sell_out["week"] = pd.to_datetime(sell_out.week)


def create_sell_out_specific_tampon(sell_out, tampone_str):
    """consider sell out of a specific tampon from 2017 to september 2019"""
    sell_out_ = sell_out[sell_out["Market Products"] == tampone_str]
    sell_out_.drop(columns=["week"], inplace=True)
    sell_out_.rename(columns={"Market Products": "Market_Prod"}, inplace=True)
    sell_out_.drop(columns=["Market_Prod"], inplace=True)
    return sell_out_


def create_sell_out_now(sell_out_specific_tampon, weekly_data):
    """consider sell out of a specific tampon from 2017 to march 2019"""
    sell_out_specific_tampon_now = sell_out_specific_tampon[:len(weekly_data)]
    sell_out_specific_tampon_now["Volumes"] = sell_out_specific_tampon_now["Volumes"].map(
        lambda x: x.replace(".", "").replace(",", "."))
    sell_out_specific_tampon_now.Volumes = pd.to_numeric(sell_out_specific_tampon_now.Volumes, errors="coerce")
    return sell_out_specific_tampon_now


def create_sell_out_future(sell_out_specific_tampon, n_period, weekly_data):
    """consider sell out of a specific tampon from march 2019 to september 2019"""
    sell_out_future = sell_out_specific_tampon[len(weekly_data):len(weekly_data) + n_period]
    sell_out_future["Volumes"] = sell_out_future["Volumes"].map(lambda x: x.replace(".", "").replace(",", "."))
    sell_out_future.Volumes = pd.to_numeric(sell_out_future.Volumes, errors="coerce")
    return sell_out_future


def create_sell_out_specific_tampon_list(lista_tamponi,sell_out):
    """create a list of dataframes of sells out from 2017 to september 2019; each element of the list represent a specific tampon sell out"""
    different_selling_out = []
    for indice_tamponi in range(len(lista_tamponi)):
        sell_out_of_a_specific_tampon = create_sell_out_specific_tampon(sell_out,lista_tamponi[indice_tamponi])
        different_selling_out.append(sell_out_of_a_specific_tampon)
    return different_selling_out



def create_sell_out_now_list(lista_tamponi,different_selling_out,weekly_data):
    """create list of sell out until march 2019, where each element represents a single brand"""
    different_selling_out_now = []
    for indice_tamponi in range(len(lista_tamponi)):
        different_selling_out_specific_tamp_now = create_sell_out_now(different_selling_out[indice_tamponi],weekly_data)
        different_selling_out_now.append(different_selling_out_specific_tamp_now)
    return different_selling_out_now

def create_sell_out_training(different_selling_out_now,percentuale_training):
    """create a list of sell out,where each element represents a single brand and has lenght of training set"""
    different_selling_training = []
    for sell_out_ in different_selling_out_now:
        training = sell_out_.iloc[:percentuale_training]
        different_selling_training.append(training)
    return different_selling_training

def create_sell_out_test(different_selling_out_now,percentuale_training):
    """create a list of sell out,where each element represents a single brand and has lenght of test set"""
    different_selling_test = []
    for sell_out_ in different_selling_out_now:
        test = sell_out_.iloc[percentuale_training:]
        different_selling_test.append(test)
    return different_selling_test


def create_sell_out_for_future_predictions(different_selling_out,n_period,weekly_data):
    """sell out from march to september 2019"""
    different_selling_out_future = []
    for sell_out_ in different_selling_out:
        sell_out_specific_tamp_future = create_sell_out_future(sell_out_,n_period,weekly_data)
        different_selling_out_future.append(sell_out_specific_tamp_future)
    return different_selling_out_future


def clean(advertising):
    """clean advertising datframe"""
    advertising.rename(columns = {"TAMPAX GRPs":"tampax_advertising"},inplace = True)
    advertising.loc[:,"Week"] = advertising.loc[:,"Week"].map(lambda x: x.replace("gen","01").replace("feb","02")
                                             .replace("mar","03")
                                             .replace("apr","04")
                                             .replace("mar","03")
                                             .replace("apr","04")
                                             .replace("mag","05")
                                             .replace("giu","06")
                                             .replace("lug","07")
                                             .replace("ago","08")
                                             .replace("set","09")
                                             .replace("ott","10")
                                             .replace("nov","11")
                                             .replace("dic","12")
                                             .replace("-","/"))



    advertising.loc[:,"Week"] = advertising.loc[:,"Week"].map(lambda x: x[0:6] + x[6:8].replace("17","2017")
                                                              .replace("18","2018").replace("19","2019"))

    advertising.loc[:,"Week"] = pd.to_datetime(advertising.loc[:,"Week"],format ='%d/%m/%Y')

    mask_of_na = advertising.loc[:,"tampax_advertising"].isna()
    advertising.loc[:,"tampax_advertising"][mask_of_na] = 0
    advertising.loc[:,"tampax_advertising"] = advertising.loc[:,"tampax_advertising"].astype(str)
    advertising.loc[:,"tampax_advertising"] = advertising.loc[:,"tampax_advertising"].map(lambda x: x.replace(",","."))
    advertising.loc[:,"tampax_advertising"] = advertising.loc[:,"tampax_advertising"].astype(float)

def convert_and_reshape_with_numpy(series):
    """convert a series to numpy and reshape to a column vector"""
    series_ = series[:]
    series_ = series_.to_numpy()
    series_ = series_.reshape(-1, 1)
    return series_


def convert_selling_out_to_numpy(different_selling_out,sell_out_list):
    """convert to numpy  sells out"""
    sell_out_list_ = sell_out_list[:]
    for indice in range(len(different_selling_out)):
        sell_out_list_[indice] = sell_out_list_[indice].to_numpy()
        sell_out_list_[indice] = sell_out_list_[indice].reshape(-1,1)
    return sell_out_list_


def plot_forecast(weekly_data, n_period, df_future_forecast, confint):
    """make plot of historical series 'weekly_data' and forecasted values"""
    index_of_future_forecast = np.arange(len(weekly_data), len(weekly_data) + n_period)
    # make series for plotting purpose
    df_future_forecast.set_index(index_of_future_forecast, inplace=True)
    weekly_data.set_index(np.arange(0, len(weekly_data)), inplace=True)
    lower_series = pd.Series(confint[:, 0], index=index_of_future_forecast)
    upper_series = pd.Series(confint[:, 1], index=index_of_future_forecast)
    # Plot
    plt.plot(weekly_data)
    plt.plot(df_future_forecast.future_prediction, color='darkgreen')
    plt.fill_between(lower_series.index,
                     lower_series,
                     upper_series,
                     color='k', alpha=.15)

    plt.title("Final Forecast of Sales Product ")
    plt.show()

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




def print_key(key_list):
    for a_key in key_list:
        print(a_key)


def print_products(products_list):
    for a_product in products_list:
        print(a_product + ' -> ' + str(products_list.index(a_product)))


def choose_product(key_list, products_dictionary):
    """allow the user to choose product among a list of products"""
    print_key(key_list)
    chosen_product_key = input("\nPlease, type the segment to be analysed: ")
    print('')
    print_products(products_dictionary[chosen_product_key])
    chosen_product_position = int(
        input("\nPlease, select the product to be analysed by typing the corresponding number: "))
    chosen_product = products_dictionary[chosen_product_key][chosen_product_position]
    return chosen_product_key,chosen_product

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


def remove_from_dict_the_products_with_few_values(segment_products_dict,segment_list):
    """Remove from segment_products_dict all the products with a small number of points"""
    clean_dict(segment_products_dict, 'Cotone')
    segment_list.remove('Cotone')
    clean_dict(segment_products_dict, 'Tampax&Go')
    segment_list.remove('Tampax&Go')
    clean_dict(segment_products_dict, 'Digital', ['Product 12', 'Product 14'])


def create_a_dict_with_mse_for_each_product(segment_products_dict,n_period):
    """create a dictionary with keys as products and values as respective mse.The result is a dictionary with all the product's mse"""
    mse_dict = {}
    for a_segment in segment_products_dict:
        for i in range(len(segment_products_dict[a_segment])):
            chosen_product = segment_products_dict[a_segment][i]
            series_tampons_global = import_and_clean_historical_series_tampons()

            Segmento_ =  a_segment
            Prodotto_ =  chosen_product
            product = historical_series_specific_product(series_tampons_global,Prodotto_)
            product = fill_with_zero_the_sales_of_missing_dates(product)
            weekly_data = aggregate_customers_and_squeeze_series_on_weeks(product)
            percentage_training = len(weekly_data) - n_period
            future_weekly_dates = pd.date_range(start= weekly_data.iloc[-1,0] , periods=n_period, freq='W')
            number_of_customer = series_tampons_global.loc[:,"Customers"].nunique()
            promo_segment_ = import_and_clean_historical_promo(Segmento_)
            future_promo_segment_ = import_and_clean_future_promo(Segmento_)
            Is_promo = generate_array_of_booleans_for_promotions(weekly_data,promo_segment_,number_of_customer)
            Is_promo_training = Is_promo.iloc[:percentage_training]
            Is_promo_test = Is_promo.iloc[percentage_training:]
            Is_promo_future = append_array_of_booleans_for_future_promotions(Is_promo,future_weekly_dates,future_promo_segment_,
                                                                     number_of_customer)

            lista_tamponi = ["TAMPONI Total","TAMPONI FATER","J&J","ZPL","ZZAO"]
            sell_out = pd.read_csv("volumes_sell_out_tampons.csv",sep =";")
            clean_sell_out(sell_out)
            different_selling_out = create_sell_out_specific_tampon_list(lista_tamponi,sell_out)
            different_selling_out_now = create_sell_out_now_list(lista_tamponi,different_selling_out,weekly_data)
            different_selling_training = create_sell_out_training(different_selling_out_now,percentage_training)
            different_selling_test = create_sell_out_test(different_selling_out_now,percentage_training)
            different_selling_out_future = create_sell_out_for_future_predictions(different_selling_out,n_period,weekly_data)
            advertising = pd.read_csv("tampax_GRPs.csv",sep =";")
            clean(advertising)
            advertising_now = advertising.loc[:(len(weekly_data)- 1),"tampax_advertising"]
            advertising_training = advertising_now.iloc[:percentage_training]
            advertising_test = advertising_now.iloc[percentage_training:]
            advertising_future = advertising.loc[len(weekly_data):len(weekly_data)+ (n_period-1),"tampax_advertising"]
            Is_promo = convert_and_reshape_with_numpy(Is_promo)
            advertising_now = convert_and_reshape_with_numpy(advertising_now)
            different_selling_out_now = convert_selling_out_to_numpy(different_selling_out,different_selling_out_now)
            variable_exog = np.concatenate((Is_promo,
                                   #different_selling_out_now[0],
                                   different_selling_out_now[1],#fater tampons
                                   different_selling_out_now[2],#j&j tampons(major competitor)
                                   #different_selling_out_now[3],
                                   #different_selling_out_now[4],
                                   advertising_now,
                                   ),axis = 1)

            smodel = pm.auto_arima(weekly_data["Sales"].to_numpy(),exogenous=variable_exog, start_p=0, start_q=0,
                             test = "adf",
                             max_p=1, max_q=1, m=52,
                             start_P=0, start_Q = 0,
                             max_P = 1,max_Q = 1, seasonal=True,
                             d=None, D=1, trace=True,
                             error_action='ignore',
                             suppress_warnings=True,
                             stepwise=True
                          )
            train = weekly_data.iloc[:percentage_training]
            test = weekly_data.iloc[percentage_training:]
            Is_promo_training = convert_and_reshape_with_numpy(Is_promo_training)
            advertising_training = convert_and_reshape_with_numpy(advertising_training)
            different_selling_training = convert_selling_out_to_numpy(different_selling_out, different_selling_training)
            variable_exog_training = np.concatenate((Is_promo_training,
                                   #different_selling_training[0],
                                   different_selling_training[1],#fater
                                   different_selling_training[2],#j&j
                                   #different_selling_training[3],
                                   #different_selling_training[4],
                                    advertising_training,
                                   #google_trend_training
                                    ),
                                   axis = 1)
            smodel.fit(train["Sales"].to_numpy(),exogenous=variable_exog_training)
            Is_promo_test = convert_and_reshape_with_numpy(Is_promo_test)
            advertising_test = convert_and_reshape_with_numpy(advertising_test)
            different_selling_test = convert_selling_out_to_numpy(different_selling_out,different_selling_test)
            variable_exog_test = np.concatenate((Is_promo_test,
                                   #different_selling_test[0],
                                   different_selling_test[1],#fater
                                   different_selling_test[2],#j&j
                                   #different_selling_test[3],
                                   #different_selling_test[4],
                                    advertising_test
                                   #google_trend_test
                                    ),axis = 1)
            forecast = smodel.predict(n_periods=len(weekly_data) - percentage_training,exogenous = variable_exog_test)
            date_forecast = test.index
            forecast_data = {"Data":date_forecast,"Prediction":forecast}
            df_forecast = pd.DataFrame(forecast_data)
            df_forecast.Prediction[df_forecast["Prediction"]<0] = 0


            mse = metrics.mean_squared_error(test.Sales.tolist(), df_forecast.loc[:,"Prediction"].tolist())
            mse_dict[str(segment_products_dict[a_segment][i])] = mse
    return mse_dict


def store_mse(a_dictionary):
    """ This function takes as paramenter a dictionary and
        saves it in a JSON file"""

    a_file = open("mse_sarimax.json", "w")
    json.dump(a_dictionary, a_file)
    a_file.close()


