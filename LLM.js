// Import dotenv config for environment variables
import 'dotenv/config';

// Import necessary components from LangChain
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { StringOutputParser } from "@langchain/core/output_parsers";  // Import the output parser

async function run() {
    // Create a new model using GPT-4
    const model = new ChatOpenAI({ model: "gpt-4" });

    // Create an instance of the StringOutputParser
    const parser = new StringOutputParser();

    // Chain the model and the parser
    const chain = model.pipe(parser);

    // Prepare the messages for the model
    const messages = [
        new SystemMessage("Translate the following from English into Italian"),
        new HumanMessage("hi!"),
    ];

    // Call the chain's invoke method to process the messages and get the parsed output
    const parsedResult = await chain.invoke(messages);

    // Log the parsed result (which should be the translated string)
    console.log(parsedResult);
}

run();
