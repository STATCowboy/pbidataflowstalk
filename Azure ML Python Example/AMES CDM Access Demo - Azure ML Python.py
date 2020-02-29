#
# Author: Jamey Johnston
# Title: Power BI dataflows for Data Scientist Talk: Access CDM Folder in Azure ML Python and do Magic!
# Date: 2020/02/27
# Blog: http://www.STATCowboy.com
# Twitter: @STATCowboy
# Git Repo: https://github.com/STATCowboy/pbidataflowstalk
#


# CdmModel Helper - https://github.com/Azure-Samples/cdm-azure-data-services-integration/tree/master/AzureMachineLearning

# Utilizes the Azure DataLake service client library for Python with ADLS gen2 Support including hierarchical namespaces
# Located here - https://pypi.org/project/azure-storage-file-datalake/


# Import packages to read ADLS gen2 Hierarchal Blobs
from azure.storage.filedatalake import DataLakeServiceClient
from azure.core.exceptions import ResourceExistsError

# Import Credentials for ADLS connectivity
import Credentials

# Import CDM Folder helper  
import CdmModel

# Import packages to read and do magic on data
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
import urllib
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score

# Helper function to convert datatypes for CDM
def type_conveter(input_type):
    switcher = {
        'boolean': 'bool',
        'int64': 'int64'
    }
    return switcher.get(input_type, 'str')

# Helper function to retrieve files from ADLS gen2
def getADLSfile(fileSysClient, fileName):
    file_client = fileSysClient.get_file_client(fileName)
    return file_client.read_file()


# Set URL and Entity(table) in PBI dataflows
account_url = "https://{}.dfs.core.windows.net/".format(Credentials.accountName)
powerbi_url = "https://{}.dfs.core.windows.net/{}".format(Credentials.accountName, Credentials.dataflowContainer)
entity_name = "AmesHousingData"

# Make connection to Data Lake
datalake_service = DataLakeServiceClient(
            account_url=account_url, credential=Credentials.credential
        )

# Make a client to Power BI dataflows blob
filesystem_client = datalake_service.get_file_system_client(Credentials.dataflowContainer)

# Read CDM Model definition from model.json file for CDM folder
# Location is '<Workspace>/<Dataflow Name>/model.json'
cdm_model_file = 'Ames Housing/Housing Data/model.json'
cdm_model_json = getADLSfile(filesystem_client, cdm_model_file).decode('utf-8')

cdm_model = CdmModel.Model.fromJson(cdm_model_json)

# Set name of Entity (table) you want to read
ames_housing_entitiy = cdm_model.entities[entity_name]

# Get path to CSV file for Entity(table)
csv_path = ames_housing_entitiy.partitions[0].location
csv_path = urllib.parse.unquote(csv_path).replace(powerbi_url, '')
csv_bytes = getADLSfile(filesystem_client, csv_path)


# Schema to read file from CDM model
schema = "cdm"

# Read to pandas dataframe with defined schema from cdm_model.json
names = [attribute.name for attribute in ames_housing_entitiy.attributes]
types = dict([(attribute.name, type_conveter(attribute.dataType.value)) for attribute in ames_housing_entitiy.attributes]) if schema is "cdm" else dict([(attribute.name, 'str') for attribute in ames_housing_entitiy.attributes])

# Generate the data frame forcing the column names and types to be those from the model.json schema
buff = BytesIO(csv_bytes)
housingDF = pd.read_csv(buff, names=names, dtype=types, na_filter = False)
buff.close()
housingDF[['Gr Liv Area', 'SalePrice']]

# Let's do Data Science

# split the values into two series instead a list of tuples
X = housingDF['Gr Liv Area'].values.reshape(-1,1)
y = housingDF['SalePrice'].values.reshape(-1,1)

# Do some EDA plots on X and Y variables
plt.figure(figsize=(15,10))
plt.tight_layout()
plt.title("Gr Liv Area")
seabornInstance.distplot(X)
plt.show()

plt.figure(figsize=(15,10))
plt.tight_layout()
plt.title("Sale Price")
seabornInstance.distplot(y)
plt.show()

# Log Transform the X (Gr Liv Area) as it is right skewed (see plot)
X = np.log(X)

plt.figure(figsize=(15,10))
plt.tight_layout()
plt.title("Gr Liv Area [Log Transformed]")
seabornInstance.distplot(X)
plt.show()

# Log Transform the Y (SalePrice) as it is right skewed (see plot)
y = np.log(y)

plt.figure(figsize=(15,10))
plt.tight_layout()
plt.title("Sale Price [Log Transformed]")
seabornInstance.distplot(X)
plt.show()

# Get a Linear Regression
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
regr = LinearRegression()

# Split Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Linear regression with the train data to obtain a model
regr.fit(X_train, y_train)

# Check that the coeffients 
m = np.exp(regr.coef_[0])
b = regr.intercept_
print(' y = {0} + x * {1}'.format(b, m))

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# Plot outputs to see how we did
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()