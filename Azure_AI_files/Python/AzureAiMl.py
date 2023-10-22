import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

def create_ml_client(subscription_id, resource_group, workspace):
    """
    Create an Azure Machine Learning client using the provided credentials.

    Args:
        subscription_id (str): Azure subscription ID.
        resource_group (str): Azure resource group name.
        workspace (str): Azure Machine Learning workspace name.

    Returns:
        MLClient: Azure Machine Learning client instance.
    """
    try:
        # Create MLClient with Azure Identity DefaultAzureCredential
        ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)
        return ml_client
    except Exception as e:
        raise Exception("Failed to create MLClient: " + str(e))

if __name__ == "__main__":
    subscription_id = "your_subscription_id"
    resource_group = "your_resource_group"
    workspace = "your_workspace_name"

    try:
        ml_client = create_ml_client(subscription_id, resource_group, workspace)
        # Use ml_client for further operations
        print("Azure Machine Learning client created successfully.")
    except Exception as e:
        print(f"Error: {e}")
