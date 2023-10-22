from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

class AzureMLClient:
    def __init__(self, subscription_id, resource_group, workspace):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace = workspace
        self.ml_client = self._create_ml_client()

    def _create_ml_client(self):
        try:
            ml_client = MLClient(DefaultAzureCredential(), self.subscription_id, self.resource_group, self.workspace)
            return ml_client
        except Exception as e:
            raise Exception("Failed to create MLClient: " + str(e))

if __name__ == "__main__":
    subscription_id = "your_subscription_id"
    resource_group = "your_resource_group"
    workspace = "your_workspace_name"

    try:
        azure_ml = AzureMLClient(subscription_id, resource_group, workspace)
        ml_client = azure_ml.ml_client
        # Use ml_client for further operations
        print("Azure Machine Learning client created successfully.")
    except Exception as e:
        print(f"Error: {e}")
