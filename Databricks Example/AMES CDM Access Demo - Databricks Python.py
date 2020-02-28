#
# Author: Jamey Johnston
# Title: Power BI dataflows for Data Scientist Talk: Access CDM Folder in Databricks
# Date: 2020/02/27
# Blog: http://www.STATCowboy.com
# Twitter: @STATCowboy
# Git Repo: https://github.com/STATCowboy/pbidataflowstalk
#


# Need to install JAR file in Databricks environment for support
# JAR file and Instructions are here - https://github.com/temalo/spark-cdm

# Get Keys for Access from Key Vault
appId = dbutils.secrets.get(scope = "pbisecretscope", key = "appId")
appKey = dbutils.secrets.get(scope = "pbisecretscope", key = "appKey")
tenantId = dbutils.secrets.get(scope = "pbisecretscope", key = "tenantId")

print(appId)

# COMMAND ----------

housingDataDf = (spark.read.format("com.microsoft.cdm")
                          .option("cdmModel", "https://<STORAGEACCOUNTNAME4PBIDATAFLOWS>.dfs.core.windows.net/powerbi/Ames Housing/Housing Data/model.json")
                          .option("entity", "AmesHousingData")
                          .option("appId", appId)
                          .option("appKey", appKey)
                          .option("tenantId", tenantId)
                          .load())


display(housingDataDf)
