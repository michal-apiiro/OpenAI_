using Azure.AI.OpenAI;

// Initialize the Azure OpenAI client
OpenAIClient client = new OpenAIClient(new Uri(oaiEndpoint), new AzureKeyCredential(oaiKey));

// Build completion options object
ChatCompletionsOptions chatCompletionsOptions = new ChatCompletionsOptions()
{
    Messages =
    {
       new ChatMessage(ChatRole.System, "You are a helpful assistant. Summarize the following text in 60 words or less."),
       new ChatMessage(ChatRole.User, text),
    },
    MaxTokens = 120,
    Temperature = 0.7f,
};

// Send request to Azure OpenAI model
ChatCompletions response = client.GetChatCompletions(
    deploymentOrModelName: oaiModelName, 
    chatCompletionsOptions);
string completion = response.Choices[0].Message.Content;

Console.WriteLine("Summary: " + completion + "\n");