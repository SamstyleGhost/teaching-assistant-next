/* eslint-disable @typescript-eslint/no-explicit-any */
import { Attachment, ChatCompletionMessage, CompletionRequest, Maxim, MaximLogger } from "@maximai/maxim-js";
import { OpenAI } from "openai";
import { ChatCompletionMessageParam } from "openai/resources/chat/completions";

// Declare global maxim logger type
declare global {
  var maxim: MaximLogger | undefined;
}

// Initialize Maxim logger with error handling
const initializeMaximLogger = async (): Promise<MaximLogger> => {
  try {
    if (!process.env.MAXIM_API_KEY || !process.env.MAXIM_LOG_REPO_ID) {
      throw new Error('MAXIM_API_KEY and MAXIM_LOG_REPO_ID must be set');
    }

    // Check if already initialized
    if (global.maxim) {
      return global.maxim;
    }

    const maxim = new Maxim({
      apiKey: process.env.MAXIM_API_KEY,
    });

    const logger = await maxim.logger({
      id: process.env.MAXIM_LOG_REPO_ID,
    });

    if (!logger) {
      throw new Error('Failed to initialize Maxim logger');
    }

    global.maxim = logger;
    console.log('Maxim logger initialized successfully');
    return logger;
  } catch (error) {
    console.error('Failed to initialize Maxim logger:', error);
    throw error;
  }
};

export async function getMaximLogger(): Promise<MaximLogger> {
  if (global.maxim) {
    return global.maxim;
  }

  return await initializeMaximLogger();
}

// Converter function for OpenAI choices to Maxim format
export function convertOpenAIChoicesToMaxim(openaiChoices: OpenAI.Chat.Completions.ChatCompletion.Choice[]): Array<{
  index: number;
  message: ChatCompletionMessage;
  logprobs: null;
  finish_reason: string;
}> {
  return openaiChoices.map((choice) => ({
    index: choice.index,
    message: {
      role: "assistant" as const,
      content: choice.message.content,
      tool_calls: choice.message.tool_calls as any, // Type compatibility handled in maximUtils
      function_call: choice.message.function_call as any,
    },
    logprobs: null,
    finish_reason: choice.finish_reason || "stop",
  }));
}

export function parseOpenAIMessages(messages: ChatCompletionMessageParam[]): {
  messages: Array<CompletionRequest | ChatCompletionMessage>;
	attachments: Attachment[];
} {
  const maximMessages: Array<CompletionRequest | ChatCompletionMessage> = [];
  const attachments: Attachment[] = [];

  for (const message of messages) {
    // Handle role mapping
    let role: "user" | "system" | "assistant" | "tool" | "function";
    if (message.role === "developer") {
      role = "system";
    } else if (message.role === "tool") {
      role = "tool";
    } else {
      role = message.role as "user" | "assistant" | "system" | "tool" | "function";
    }

    // Handle content conversion
    const { content, extractedAttachments } = convertContent(message.content);
    attachments.push(...extractedAttachments);

    // Create the message based on role
    if (role === "assistant") {
      const assistantMessage: ChatCompletionMessage = {
        role: "assistant",
        content: typeof content === "string" ? content : null,
      };

      // Handle tool calls if present
      if ("tool_calls" in message && message.tool_calls) {
        assistantMessage.tool_calls = message.tool_calls as any;
      } else if ("function_call" in message && message.function_call) {
        // Legacy function call format
        assistantMessage.function_call = message.function_call as any;
      }

      maximMessages.push(assistantMessage);
    } else {
      // Handle user, system, tool, function messages
      const completionMessage: CompletionRequest = {
        role,
        content,
      };

      // Add tool_call_id for tool messages
      if ("tool_call_id" in message && message.tool_call_id) {
        completionMessage.tool_call_id = message.tool_call_id;
      }

      // Add name for function messages (legacy)
      if ("name" in message && message.name) {
        // Note: Maxim's CompletionRequest doesn't have a name field,
        // but we can include it in content or handle as needed
      }

      maximMessages.push(completionMessage);
    }
  }

  return { messages: maximMessages, attachments };
}

// Helper function to convert OpenAI content format to Maxim format
function convertContent(content: OpenAI.Chat.ChatCompletionMessageParam["content"]): {
  content: string | Array<any>;
  extractedAttachments: Attachment[];
} {
  const extractedAttachments: Attachment[] = [];

  if (typeof content === "string") {
    return { content, extractedAttachments };
  }

  if (!content) {
    return { content: "", extractedAttachments };
  }

  if (Array.isArray(content)) {
    const maximContent: Array<any> = [];
    let textContent = "";

    for (const part of content) {
      if (part.type === "text") {
        textContent += part.text;
        maximContent.push({
          type: "text",
          text: part.text,
        });
      } else if (part.type === "image_url") {
        // Extract image as attachment
        const attachmentId = `image_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        extractedAttachments.push({
          id: attachmentId,
          type: "url",
          url: part.image_url.url,
          mimeType: "image/*", // Could be more specific based on URL
        });

        maximContent.push({
          type: "image_url",
          image_url: {
            url: part.image_url.url,
            detail: part.image_url.detail || "auto",
          },
        });
      }
      // Note: Other content types like input_audio could be handled here
    }

    // If there's only text content, return as string for simplicity
    if (maximContent.length === 1 && maximContent[0].type === "text") {
      return { content: textContent, extractedAttachments };
    }

    return { content: maximContent, extractedAttachments };
  }

  return { content: "", extractedAttachments };
}