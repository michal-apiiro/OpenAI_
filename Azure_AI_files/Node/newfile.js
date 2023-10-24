import { HfAgent } from "@huggingface/agents";
const { OpenAIClient } = require("@azure/openai");
const { DefaultAzureCredential } = require("@azure/identity");

const client = new OpenAIClient("<endpoint>", new DefaultAzureCredential());
const openai = client
const HF_ACCESS_TOKEN = "hf_..."; // get your token at https://huggingface.co/settings/tokens
const agent = new HfAgent(HF_ACCESS_TOKEN);
