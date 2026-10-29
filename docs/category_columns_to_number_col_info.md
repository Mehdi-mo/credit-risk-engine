first i want to check cardinality 

i use this code : 
not_number_cols = ['purpose', 'home_ownership', 'verification_status', 'application_type','initial_list_status', 'disbursement_method', 'addr_state', 'zip_code']

df[not_number_cols].nunique().sort_values()

and i get these values : 
application_type         2
initial_list_status      2
disbursement_method      2
verification_status      3
home_ownership           4
purpose                 13
addr_state              50
zip_code               845
dtype: int64

i conclude that application_type , initial_list_status, disbursement_method they mean sth binary or a yes or no 

so i make : 
binary_cat_cols = ['application_type', 'initial_list_status', 'disbursement_method']

their values with value_count(): 
--- application_type ---
application_type
Individual    25884
Joint App       396
Name: count, dtype: int64
--- initial_list_status ---
initial_list_status
w    16043
f    10237
Name: count, dtype: int64
--- disbursement_method ---
disbursement_method
Cash         26213
DirectPay       67
Name: count, dtype: int64

one note about disbursment_method and application_type : it has 2 different values and one has way more values than others i shoild maybe delte it because it might not be useful to the model

disbursement_method
Cash         26213
DirectPay       67

application_type
Individual    25884
Joint App       396


groups that have cardinality is 1: zip_code because it has more than 100 unique values so it should be treated with esp care 

cardinality_cat_col = ['zip_code']


address_state has about 50 different values and it is not good idea to use one-hot encoding method because it makes more columns than there are rows so maybe we should drop it ot treat it different


might be good to use = 
purpose (13 different values but not one-hot), 
home_ownership = 3 different values and deleting the ones with any value,
verification_status = it is very good , 3 different values but maybe source verified and verified mean the same and so we can have 2 , 
initial_list_status = good , two values , we can turn it to 1 and 0 ,


i checked and they dont have missing values = 
purpose                0.0
home_ownership         0.0
verification_status    0.0
application_type       0.0
initial_list_status    0.0
disbursement_method    0.0
addr_state             0.0
zip_code               0.0
dtype: float64

