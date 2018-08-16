# coding: utf-8
# preprocess module that reads order table
# cluster dealers based on training data 
# and generates training and testing data sets
import pandas as pd
import numpy as np
import datetime
import copy
import pyodbc
import sys
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans


def main():
    pass
if __name__ == "__main__":
    main()

##########################################################################################
######################## Read data #######################################################
##########################################################################################

def get_stable_dealers(csvfile='stable_STD2.csv'):
    stable = pd.read_csv(csvfile, index_col=0, dtype=str)
    stable_list = stable.values.ravel().tolist()
    print(len(stable_list), "stable dealers")
    return stable_list

def get_orders(stable_list, start_date='2014-01-01', end_date='2018-08-01'):
    
    """ reads part of ORDER table from remote database and return as pd.DataFrame """
    """ for orders with Requested Date predating Order Date, use Order Date as Requested Date """

    conn = pyodbc.connect("Driver={ODBC Driver 13 for SQL Server};Server=yetidb01.database.windows.net;database=YETISQLDW01;uid=ezeng;PWD=Ed1tH2EnG#")
    cursor = conn.cursor()
    
    query = """
    WITH ORDER_DIM AS (
        SELECT CustomerID, OrderDate, Requested_Date, OrderQty, UnitPrice, ProductID,
        CASE WHEN 
            Requested_Date < OrderDate THEN OrderDate ELSE Requested_Date
        END AS Req_Date
        FROM [EDW_DS].[ORDERFCT]
        WHERE CustomerID IN ({})
        AND RejectionReason IS NULL
        AND SalesOffice IN ('STD','HYB')
        AND Ordertype NOT IN ('ZARF','ZARM','ZEG2','ZERF','ZERM','ZRE')
        AND SoldtoParty NOT IN ('91860','0000107894','0000114883','0000108654')
        AND OrderQty > 0
        AND Requested_Date >= '{}' AND Requested_Date <= '{}'
    ), 
    PRODUCT AS (
        SELECT ProductID, ProductCategory, MasterSKU
        FROM [EDW_DS].[PRODUCT_DIM]
        WHERE ISCore = 'True'
    ),
    CUSTOMER AS (
        SELECT CustomerID, Zipcode, SalesOffice
        FROM [EDW_DS].[CUSTOMER_DIM]
    )
    SELECT ORDER_DIM.CustomerID, ORDER_DIM.OrderDate, ORDER_DIM.Req_Date AS Requested_Date, ORDER_DIM.OrderQty, ORDER_DIM.UnitPrice, PRODUCT.ProductCategory, PRODUCT.MasterSKU, Customer.Zipcode, Customer.SalesOffice
    FROM ORDER_DIM
    INNER JOIN PRODUCT ON PRODUCT.ProductID = ORDER_DIM.ProductID
    INNER JOIN CUSTOMER ON CUSTOMER.CustomerID = ORDER_DIM.CustomerID
    ORDER BY ORDER_DIM.OrderDate ASC;""".format(','.join(["'" + customer + "'" for customer in stable_list]), start_date, end_date)
    
    print("reading order table from ORDERFCT")
    order = pd.read_sql(query, conn)
    return order


def get_customer_dim(train, stable_list):
    
    """ returns a dataframe with customerID, zipcode and salesoffice """
    valid = (isinstance(stable_list, list) and isinstance(stable_list[0], str) and len(stable_list) > 0)
    if not valid:
        raise TypeError("call stable_list() first to generate a list of str for customerID")
    try:
        customer_dim = train.loc[(train['CustomerID']==stable_list[0]), ['CustomerID','Zipcode', 'SalesOffice']].head(1)
    except:
        AttributeError("Check split_datasets() parameters")
    for i in range(1,len(stable_list)):
        customer_id = stable_list[i]
        geo = train.loc[(train['CustomerID']==customer_id), ['CustomerID','Zipcode','SalesOffice']].head(1)
        customer_dim = pd.concat([customer_dim,geo], ignore_index=True, sort=False, copy=False)
    
    return customer_dim


##########################################################################################
######################## cleansing & split ###############################################
##########################################################################################

def combine_masterSKU(order):
    
    order['OrderDate'] = pd.to_datetime(order['OrderDate'])
    order['Requested_Date'] = pd.to_datetime(order['Requested_Date'])
    op = order.copy()
    
    # exclude selected masterSKUs
    exclusion = ['Tundra 50', 'Tundra 420', 'R12 Bottle', '2 Pack Lowball',
                 'R Wine Tumbler', '2 Pack Wine','R16 Pint', '2 Pack R16 Pint',
                 'Hopper 20', 'Hopper 40', 'Hopper2.0 20', 'Hopper2.0 40',
                 'Seamless R30 Tumbler','Seamless R20 Tumbler','Seamless R10 Lowball',
                 'Silo 6G','4 Pack Lowball', 'Tundra Haul',
                 'Panga 75 Duffel', 'Panga 100 Duffel', 
                 'Panga 50 Duffel','LoadOut 5G Bucket', 'R64 Bottle'
                 'BackFlip 24', 'R1/2G Jug', 'R1G Jug', 'R14 Mug',
                 'Camino 35 Carryall','Panga 28 Backpack', 'R Wine',
                 'BackFlip 24','R64 Bottle']
    op = op.loc[~(op['MasterSKU']).isin(exclusion)]
    
    # Fill histories of Flip 8 and Flip 18 with Flip 12
    flip12 = op.loc[(op['MasterSKU'] == 'Flip 12') 
                & (op['Requested_Date'] >= pd.Timestamp(2016,8,1)) 
                & (op['Requested_Date'] < pd.Timestamp(2017,7,1))].copy()
    flip12['MasterSKU'] = 'Flip 8'
    op = pd.concat([op, flip12])
    flip12['MasterSKU'] = 'Flip 18'
    op = pd.concat([op, flip12])
    
    # Fill R26 with R18
    r18 = op.loc[(op['MasterSKU'] == 'R18 Bottle')
                 & (op['Requested_Date'] >= pd.Timestamp(2016,3,1))
                 & (op['Requested_Date'] < pd.Timestamp(2017,5,1))].copy()
    r18['MasterSKU'] = 'R26 Bottle'
    op = pd.concat([op, r18])
    
    # combine large tundras
    large_tundra = ['Tundra 105', 'Tundra 110', 'Tundra 125', 'Tundra 160',
                   'Tundra 210', 'Tundra 250', 'Tundra 350']
    large_tundra_df = op.loc[op['MasterSKU'].isin(large_tundra), 'MasterSKU'] = 'Tundra 105-350'
    
    # combine tank 45 and tank 85
    tank = ['Tank 45', 'Tank 85']
    op.loc[op['MasterSKU'].isin(tank), 'MasterSKU'] = 'Tank 45-85'
    
    # combine Hopper 30 with Hopper 2.0 30
    op.loc[op['MasterSKU']=='Hopper 30', 'MasterSKU'] = 'Hopper2.0 30'
    
    op['Requested_Date'] = op['Requested_Date'].apply(pd.to_datetime, errors='raise')
    
    print(len(op),"records\n")
    print("Forecasting {} stable dealers and {} masterSKUs:".format(op['CustomerID'].nunique(), op['MasterSKU'].nunique()))
    print(op['MasterSKU'].unique())
    print("Training and testing data date range: {} to {}\n".format(op['Requested_Date'].min(), op['Requested_Date'].max()))
    
    return op



def split_datasets(table, predict=False, train_start='2014-01-01', train_end='2017-05-31', test_start='2017-06-01', test_end='2018-05-31'):
    
    """ split data into training vs testing based on time cut-offs """
    """ date format: str 'yyyy-mm-dd' """
    """ split date are inclusive on both ends """
    """ if predict=True, the entire dataset becomes training (no testing) data """

    try:
        train_start = pd.Timestamp(train_start)
        train_end = pd.Timestamp(train_end)
        test_start = pd.Timestamp(test_start)
        test_end = pd.Timestamp(test_end)
    except:
        print("Check date format: str 'YYYY-MM-DD'")
    
    # format training data
    train = table.loc[(table['Requested_Date'] >= train_start) & (table['Requested_Date'] <= train_end)]
    train = train.reset_index()
    train.drop(columns=['index'],inplace=True)
    print("training data date range: {} to {}".format(train.Requested_Date.min(), train.Requested_Date.max()))
    print("training: {0:.2%}".format(len(train)/len(table)))
    
    if predict:
        op = train
    else:    # format testing data
        test = table.loc[(table['Requested_Date'] >= test_start) & (table['Requested_Date'] <= test_end)]
        print("testing data date range: {} to {}".format(test.Requested_Date.min(), test.Requested_Date.max()))
        test = test.reset_index()
        test.drop(columns=['index'], inplace=True)
        op = train, test
        
    return op

##########################################################################################
######################### create dealer cluster feature ##################################
##########################################################################################

def cluster_dealers(train, customer_dim, stable_list, n=5, plot_elbow=False, to_csv=False):

    """ returns a dataframe with customer features and customer group labels """
    """ optional csv file in the same path """
    
    valid = (isinstance(customer_dim, pd.DataFrame) and isinstance(stable_list, list)
             and len(customer_dim) > 0 and len(stable_list) > 0)
    if not valid:
        raise TypeError("Call stable_list() to generate stable_list, then get_customer_dim() to generate customer_dim")
    
    data = train.reset_index()
    data.drop(columns=['index'], inplace=True)
    
    features = create_customer_feature_df(train, customer_dim, stable_list)
    customer_features = label_customer_clusters(n, features)
    
    if plot_elbow:
        kmeans_elbow(features)
    if to_csv:
        customer_features.to_csv("customer_features.csv")
        
    return customer_features



def kmeans_elbow(customer_feature_df):
    """ visual heuristic to find the optimal number of cluster for KMeans """
    """ plot multiple results to decide """
    X = StandardScaler().fit_transform(customer_feature_df)
    score_list = [KMeans(n_clusters=k,init='random').fit(X).score(X) for k in range(1,10)]
    plt.plot([i for i in range(1,10)], score_list)
    plt.grid()
    plt.show()

    
def label_customer_clusters(n, customer_features_df):
    """ adds group label to the end of customer_feature dataframe """
    op = customer_features_df.copy()
    km = KMeans(n_clusters=n).fit(customer_features_df.values)
    op['cluster'] = km.labels_
    return op

def create_customer_feature_df(train, customer_dim, stable_list, end_date='2018-5-31'):
    
    """ call this function before performing KMeans clustering """
    """ 1 row per stable customer """
    """ columns = purchase stats based on this customer's training data """
    
    i = 0
    dealer_list = []
    end_date = pd.Timestamp(end_date)
    print("job started:", datetime.datetime.now())
    
    for customer_id in stable_list:
        
        history = train.loc[train['CustomerID'] == customer_id]
        row = customer_dim.loc[customer_dim['CustomerID'] == customer_id]
        
        dealer = Customer(customer_id, history, row, end_date)
        dealer_list.append(dealer)
        
        i += 1
        if i % 100 == 0:
            sys.stdout.write('\r'+"{0:.3%}".format(i/len(stable_list)))
            sys.stdout.flush()

    op = pd.DataFrame.from_records([dealer.to_dict() for dealer in dealer_list])
    print("\njob finish time:", datetime.datetime.now())
    return op



class Customer:
    
    def __init__(self, customer_id, order_history, customer_dim_row, end_date):
        
        self.id = customer_id
        self.history = order_history.sort_values(by='Requested_Date').reset_index()
        self.zipcode = int(customer_dim_row['Zipcode'].str.split('-',expand=False).values[0][0])
        self.sales_office = 1 if customer_dim_row['SalesOffice'].values[0] == 'HYB' else 0
        self.end_date = end_date
        
        self.__calc_first_requested_date()
        self.__calc_overall_order_quantity()
        self.__calc_average_order_quantity()
        self.__calc_diversity()
        self.__calc_variance()
        self.__calc_average_unit_price()
        self.__calc_patient_rate()
        self.__calc_average_reorder_intervals()
        self.__calc_new_product_rate()
        
    def __calc_first_requested_date(self):
        self.first_date = self.history['Requested_Date'].min()
        
    def __calc_overall_order_quantity(self):
        self.overall_quantity = self.history['OrderQty'].sum()
        
    def __calc_average_order_quantity(self):
        if self.first_date is None:
            self.__calc_first_requested_date()
        history_length = (self.end_date - self.first_date).days
        self.average_quantity = self.overall_quantity / history_length
        
    def __calc_diversity(self):
        self.product_diversity = self.history['MasterSKU'].nunique() * self.history['ProductCategory'].nunique()
    
    def __calc_variance(self):
        self.quantity_variance = self.history['OrderQty'].var()
        
    def __calc_average_unit_price(self):
        self.average_unit_price = np.mean(self.history['UnitPrice'].values)
        
    def __calc_patient_rate(self):
        """ patient order = (requested date - order date) > 0, otherwise impatient """
        """ inaccurate as some orders have Requested_Date predating Order_Date """
        """ and their Requested_Dates have been changed to Order_Dates """
        wait_days = (self.history['Requested_Date'] - self.history['OrderDate']).dt.days
        patient = wait_days.loc[wait_days > 0]
        self.patient_rate = len(patient)/len(wait_days)

    def __calc_average_reorder_intervals(self):
        """ number of days between orders for each dealer, drop NA (first record has no prior order) """
        intervals = self.history['Requested_Date'].diff().dt.days.dropna(axis=0).values
        self.average_reorder_intervals = np.mean(intervals)
        
    def __is_new_product(self, masterSKU, requested_date):
        new = False
        if requested_date >= datetime.date(2014,1,1) and requested_date <= datetime.date(2014,3,1):
            if masterSKU in ['Tundra 35','Tundra 250','Tundra 45','Tank 85', 'Roadie 20','Tundra 160',
                             'Tundra 125','Tundra 110','Tundra 65','Tundra 250']:
                new = True
        elif requested_date >= datetime.date(2014,3,1) and requested_date <= datetime.date(2014,4,1):
            if masterSKU in ['Tundra 75', 'Tundra 105']:
                new = True
        elif requested_date >= datetime.date(2014,4,1) and requested_date <= datetime.date(2014,5,1):
            if masterSKU in ['R20 Tumbler', 'R30 Tumbler']:
                new = True
        elif requested_date >= datetime.date(2014,9,1) and requested_date <= datetime.date(2014,10,1):
            if masterSKU == 'Hopper 30':                                                         # not found --no space
                new = True
        elif requested_date >= datetime.date(2015,2,1) and requested_date <= datetime.date(2015,4,1): 
            if masterSKU in ['R Colsters', 'Hopper 20', 'Tank 45']:
                new = True
        elif requested_date >= datetime.date(2015,9,1) and requested_date <= datetime.date(2015,10,15):
            if masterSKU == 'R10 Lowball':
                new = True
        elif requested_date >= datetime.date(2016,1,1) and requested_date <= datetime.date(2016,3,1):
            if masterSKU == 'Tundra 350':
                new = True
        elif requested_date >= datetime.date(2016,2,1) and requested_date <= datetime.date(2016,4,1):
            if masterSKU == 'Hopper 40':
                new = True
        elif requested_date >= datetime.date(2016,3,1) and requested_date <= datetime.date(2016,4,1):
            if masterSKU in ['R36 Bottle', 'R18 Bottle', 'R64 Bottle']:
                new = True
        elif requested_date >= datetime.date(2016,8,1) and requested_date <= datetime.date(2016,9,1):
            if masterSKU == 'Flip 12':
                new = True
        elif requested_date >= datetime.date(2016,9,1) and requested_date <= datetime.date(2016,10,1):
            if masterSKU == 'Tundra 210':
                new = True 
        elif requested_date >= datetime.date(2017,2,1) and requested_date <= datetime.date(2017,4,1):
            if masterSKU in ['Hopper2.0 40','Hopper2.0 30', 'Hopper2.0 20']:      # not found in ZVA05 table --no space
                new = True
        elif requested_date >= datetime.date(2017,5,1) and requested_date <= datetime.date(2017,6,1):
            if masterSKU in ['R26 Bottle','R1G Jug','R1/2G Jug']:    # 'R 1G Jug' and 'R 1/2G Jug' not found --no space
                new = True
        elif requested_date >= datetime.date(2017,7,1) and requested_date <= datetime.date(2017,8,1):
            if masterSKU in ['Flip 8', 'Flip 18']:
                new = True
        elif requested_date >= datetime.date(2017,9,1) and requested_date <= datetime.date(2017,10,1):
            if masterSKU == 'R14 Mug':
                new = True
        elif requested_date >= datetime.date(2018,3,1) and requested_date <= datetime.date(2018,4,1):
            if masterSKU == 'BackFlip 24':                                       # not found --capital letter F in ZVA05
                new = True
        return new
    
    def __label_new_products(self, df):
        """ label orders of new product """
        """ returns a dataframe with one additional column """
        """ 'is_new_product' = 1 if purchase is within 60 days of launch date """
        op = df.copy()
        op['new_product'] = op.apply(lambda _: '', axis=1)
        cols = op.columns
        for i, row in df.iterrows():                # indices may encouter problem if database table or SQL query change
            sku = row[cols.get_loc('MasterSKU')]
            date = row[cols.get_loc('Requested_Date')].to_pydatetime().date()
            op.iat[i, cols.get_loc('new_product')] = 1 if is_new_product(sku,date) else 0
        return op
    
    def __calc_new_product_rate(self):
        """ the proportion of order quantity that is new product """
        """ new product = new product master SKU ordered within 60 days of launch date """
        df = self.__label_new_products(self.history)
        new_product_purchase = df.loc[df['new_product'] == 1]
        rate = len(new_product_purchase) / len(self.history)
        self.new_product_rate = rate
        
    def to_dict(self):
        return {
            'new_product_rate': self.new_product_rate,
            'avg_reorder_interval': self.average_reorder_intervals,
            'patient_rate': self.patient_rate,
            'avg_unit_price': self.average_unit_price, 
            'quantity_variance': self.quantity_variance, 
            'product_diversity': self.product_diversity, 
            'overall_quantity': self.overall_quantity,
            'avg_quantity': self.average_quantity, 
            'zipcode': self.zipcode,
            'sales_office': self.sales_office, 
            'CustomerID': self.id,
        }


##########################################################################################
####################### update features in training/testing data #########################
##########################################################################################
    

def aggregate_data(df, customer_feature_df):
    
    """ return date-sorted and dummy-encoded dataframe for modeling """
    
    valid = (isinstance(df, pd.DataFrame) and isinstance(customer_feature_df, pd.DataFrame)
             and len(df) > 0 and len(customer_feature_df) > 0)
    if not valid:
        raise TypeError("Check training/testing and/or customer_feature_df.")
    
    op = df.copy()
    
    print("adding month and year columns")
    op = add_year_month_cols(op)
    
    print("adding new product label")
    op = label_new_product(op)
    
    print("adding price change info")
    op = add_price_change(op)
    
    print("adding customer cluster")
    op = add_customer_clusters(op, customer_feature_df)
    
    print("converting categorical values to numerical")
    op = encode_categorical_columns(op)
    
    print("calculating monthly sum order quantity per customer cluster (response vector Y)")
    op = calculate_monthly_sum(op)

    op.sort_values(by=['Requested_Date'], ascending=True, inplace=True)
    
    return op



def add_year_month_cols(df):
    op = df.copy()
    op['year'] = op.apply(lambda _: '', axis=1)
    op['month'] = op.apply(lambda _: '', axis=1)
    op = extract_year_month(op)
    return op
def extract_year_month(df):
    op = df.copy()
    op['Requested_Date'] = pd.to_datetime(op['Requested_Date'])
    op['year'], op['month'] = op['Requested_Date'].dt.year, op['Requested_Date'].dt.month
    return op



def label_new_product(df):
    
    """ label orders of new product """
    """ returns a dataframe with one additional column """
    """ 'is_new_product' = 1 if purchase is within 60 days of launch date """
    
    op = df.copy()
    op['new_product'] = op.apply(lambda _: '', axis=1)
    
    # indices may encouter problem if database table or SQL query change
    cols = op.columns
    for i, row in df.iterrows():
        sku = row[cols.get_loc('MasterSKU')]
        date = row[cols.get_loc('Requested_Date')].to_pydatetime().date()
        op.iat[i, cols.get_loc('new_product')] = 1 if is_new_product(sku,date) else 0
    return op




def is_new_product(masterSKU, requested_date):
    new = False
    if requested_date >= datetime.date(2014,1,1) and requested_date <= datetime.date(2014,3,1):
        if masterSKU in ['Tundra 35','Tundra 250','Tundra 45','Tank 85', 'Roadie 20','Tundra 160',
                         'Tundra 125','Tundra 110','Tundra 65','Tundra 250']:
            new = True
    elif requested_date >= datetime.date(2014,3,1) and requested_date <= datetime.date(2014,4,1):
        if masterSKU in ['Tundra 75', 'Tundra 105']:
            new = True
    elif requested_date >= datetime.date(2014,4,1) and requested_date <= datetime.date(2014,5,1):
        if masterSKU in ['R20 Tumbler', 'R30 Tumbler']:
            new = True
    elif requested_date >= datetime.date(2014,9,1) and requested_date <= datetime.date(2014,10,1):
        if masterSKU == 'Hopper 30':                                       # not found --no space
            new = True
    elif requested_date >= datetime.date(2015,2,1) and requested_date <= datetime.date(2015,4,1): 
        if masterSKU in ['R Colsters', 'Hopper 20', 'Tank 45']:
            new = True
    elif requested_date >= datetime.date(2015,9,1) and requested_date <= datetime.date(2015,10,15):
        if masterSKU == 'R10 Lowball':
            new = True
    elif requested_date >= datetime.date(2016,1,1) and requested_date <= datetime.date(2016,3,1):
        if masterSKU == 'Tundra 350':
            new = True
    elif requested_date >= datetime.date(2016,2,1) and requested_date <= datetime.date(2016,4,1):
        if masterSKU == 'Hopper 40':
            new = True
    elif requested_date >= datetime.date(2016,3,1) and requested_date <= datetime.date(2016,4,1):
        if masterSKU in ['R36 Bottle', 'R18 Bottle', 'R64 Bottle']:
            new = True
    elif requested_date >= datetime.date(2016,8,1) and requested_date <= datetime.date(2016,9,1):
        if masterSKU == 'Flip 12':
            new = True
    elif requested_date >= datetime.date(2016,9,1) and requested_date <= datetime.date(2016,10,1):
        if masterSKU == 'Tundra 210':
            new = True 
    elif requested_date >= datetime.date(2017,2,1) and requested_date <= datetime.date(2017,4,1):
        if masterSKU in ['Hopper2.0 40','Hopper2.0 30', 'Hopper2.0 20']:    # not found in ZVA05 table --no space
            new = True
    elif requested_date >= datetime.date(2017,5,1) and requested_date <= datetime.date(2017,6,1):
        if masterSKU in ['R26 Bottle','R1G Jug','R1/2G Jug']:  # 'R 1G Jug' and 'R 1/2G Jug' not found --no space
            new = True
    elif requested_date >= datetime.date(2017,7,1) and requested_date <= datetime.date(2017,8,1):
        if masterSKU in ['Flip 8', 'Flip 18']:
            new = True
    elif requested_date >= datetime.date(2017,9,1) and requested_date <= datetime.date(2017,10,1):
        if masterSKU == 'R14 Mug':
            new = True
    elif requested_date >= datetime.date(2018,3,1) and requested_date <= datetime.date(2018,4,1):
        if masterSKU == 'BackFlip 24':                                    # not found --capital letter F in ZVA05
            new = True   
    return new




def add_price_change(df):
    
    """ estimates the impact of promotion (% off) on sale quantities """
    """ price_change == 0 if no change """
    """ price_change > 0 if discounted """
    """ price_change < 0 if price increased """
    op = df.copy()
    op['price_change'] = op.apply(lambda _: '', axis=1)
    
    # indices may encouter problem if database table or SQL query change
    cols = op.columns
    for i, row in df.iterrows():
        master_SKU = row[cols.get_loc('MasterSKU')]
        requested_date = row[cols.get_loc('Requested_Date')].to_pydatetime().date()
        op.iat[i, cols.get_loc('price_change')] = calc_promo_rate(master_SKU, requested_date)

    return op



def calc_promo_rate(masterSKU, requested_date):
    
    """ use 2 months within price change as direct influences """
    """ return a rate of change in relation to the original price: (old-new)/old """
    
    rate = 0    # default no discount
    
    if masterSKU == 'R10 Lowball':
        if requested_date >= datetime.date(2017,10,1) and requested_date < datetime.date(2017,12,1):
            rate = (24.99-19.99)/24.99
    if masterSKU == 'R18 Bottle':
        if requested_date >= datetime.date(2017,10,1) and requested_date < datetime.date(2017,12,1):
            # combine gross and discount STD promo (no color information on MasterSKU level)
            gross = (39.99-29.99)/39.99
            discount = (29.99-22.49)/29.99
            rate = (gross + discount) / 2            
    if masterSKU == 'R36 Bottle':
        if requested_date >= datetime.date(2017,10,1) and requested_date < datetime.date(2017,11,15):
            rate = (49.99-37.49)/49.99
    if masterSKU in ['Hopper2.0 30','Hopper 30']:     # hoppers are combined
        if requested_date >= datetime.date(2017,2,1) and requested_date < datetime.date(2017,4,1):
            rate = (349.99-249.99)/349.99
    if masterSKU == 'Roadie 20':
        if requested_date >= datetime.date(2017,5,1) and requested_date < datetime.date(2017,7,1):
            rate = (249.99-199.99)/249.99
    if masterSKU == 'Flip 12':
        if requested_date >= datetime.date(2017,5,1) and requested_date < datetime.date(2017,7,1):
            rate = (299.99-249.99)/299.99
    if masterSKU == 'R Colsters':
        if requested_date >= datetime.date(2017,5,1) and requested_date < datetime.date(2017,7,1):
            rate = (29.99-24.99)/29.99
    if masterSKU == 'R18 Bottle':
        if requested_date >= datetime.date(2017,10,15) and requested_date < datetime.date(2017,11,15):
            rate = (29.99-22.49)/29.99
    if masterSKU == 'R20 Tumbler':
        if requested_date >= datetime.date(2017,5,1) and requested_date < datetime.date(2017,7,1):
            rate = (29.99-24.99)/29.99
    if masterSKU == 'R26 Bottle':
        if requested_date >= datetime.date(2017,10,15) and requested_date < datetime.date(2017,11,15):
            rate = (39.99-29.99)/39.99
    if masterSKU == 'R30 Tumbler':
        if requested_date >= datetime.date(2017,5,1) and requested_date < datetime.date(2017,7,1):
            rate = (39.99-34.99)/39.99
    if masterSKU == 'R36 Bottle':     
        if requested_date >= datetime.date(2017,10,15) and requested_date < datetime.date(2017,11,15):
            rate = (49.99-37.49)/49.99
    if masterSKU == 'Tundra 35': 
        if requested_date >= datetime.date(2018,1,1) and requested_date < datetime.date(2017,3,1):
            rate = (299.99-249.99)/299.99
    if masterSKU == 'Tundra 45':
        if requested_date >= datetime.date(2018,1,1) and requested_date < datetime.date(2017,3,1):
            rate = (349.99-299.99)/349.99
    if masterSKU == 'Tundra 65':
        if requested_date >= datetime.date(2018,1,1) and requested_date < datetime.date(2017,3,1):
            rate = (399.99-349.99)/399.99
    return rate



def add_customer_clusters(df, customer_feature_df):
    """ add a single column of cluster number """
    op = df.copy()
    clusters = customer_feature_df[['CustomerID', 'cluster']]
    op = pd.merge(op, clusters, how='left', on='CustomerID')
    return op



def calculate_monthly_sum(df):
    
    """ calculate monthly sum order quantity per master SKU for each dealer cluster (response vector) """
    
    op = df.copy()
    op['monthly_sum_order_qty'] = np.nan
    hi = len(df)
    print("job start time:", datetime.datetime.now())

    cols = op.columns
    
    # indices may encouter problem if database table or SQL query change
    for i, row in df.iterrows():

        subset_sum = df.loc[(df['year'] == row[cols.get_loc('year')]) 
                            & (df['month'] == row[cols.get_loc('month')]) 
                            & (df['ProductCategory'] == row[cols.get_loc('ProductCategory')])
                            & (df['MasterSKU'] == row[cols.get_loc('MasterSKU')])
                            & (df['cluster'] == row[cols.get_loc('cluster')]),'OrderQty'].sum(min_count=1)
        
        op.loc[( (np.isnan(op['monthly_sum_order_qty']))
                        & (op['year'] == row[cols.get_loc('year')]) 
                        & (op['month'] == row[cols.get_loc('month')])
                        & (df['ProductCategory'] == row[cols.get_loc('ProductCategory')])
                        & (df['MasterSKU'] == row[cols.get_loc('MasterSKU')])
                        & (df['cluster'] == row[cols.get_loc('cluster')])
                        ), 'monthly_sum_order_qty'] = subset_sum

        if (i % 1000) == 0 and i <= hi:
            sys.stdout.write('\r'+"{0:.3%}".format(i/hi))
            sys.stdout.flush()

    print("\njob finish time:", datetime.datetime.now()) 
    return op



def encode_categorical_columns(df):
    """ dummy encode product category & masterSKU """
    """ LabelEncoder assumes natural ordering. Similar products are numerically close """
    op = df.copy()
    le = LabelEncoder()
    for c in ['ProductCategory', 'MasterSKU']:
        op[c] = le.fit_transform(op[c].values)
    return op