from azure.ai.ml import MLClient 
from azure.identity import DefaultAzureCredential
       
if __name__ == "__main__":
    subscription_id = "your_subscription_id"
    resource_group = "your_resource_group"
    workspace = "your_workspace_name"
    ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)
    ml_client.compute.get("cpu-cluster")
