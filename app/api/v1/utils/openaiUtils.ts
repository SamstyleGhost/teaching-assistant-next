import OpenAI from 'openai';
import dotenv from 'dotenv';
import path from 'path';
import { readPdfContent } from './fileUtils';
import { convertOpenAIChoicesToMaxim, getMaximLogger, parseOpenAIMessages } from './maximUtils';
import { v4 as uuidv4 } from 'uuid';

dotenv.config();


const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface TeacherParams {
  nativeLanguage: string;
  targetLanguage: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  messages: ChatMessage[];
}

let klingonDictionaryCache: string | null = null;

export const getTeacherResponse = async ({
  nativeLanguage,
  targetLanguage,
  difficulty,
  messages,
}: TeacherParams): Promise<string | null> => {
  let additionalContext = '';

  const maximLogger = await getMaximLogger();

  const trace = maximLogger.trace({
    id: uuidv4(),
  })

  if (targetLanguage.toLowerCase() === 'klingon') {
    if (!klingonDictionaryCache) {
      try {
        const filePath = path.join(process.cwd(), 'data', 'Franchise - The Klingon Dictionary.pdf');
        console.log(`Loading Klingon dictionary from: ${filePath}`);
        klingonDictionaryCache = await readPdfContent(filePath);
      } catch (error) {
        console.error('Failed to load Klingon dictionary:', error);
      }
    }

    if (klingonDictionaryCache) {
      additionalContext = klingonDictionaryCache;
      const retrieval = trace.retrieval({
        id: uuidv4(),
        name: "klingon-dictionary",
      });
      retrieval.input(targetLanguage);
      retrieval.output(additionalContext);
      retrieval.end();
    }
  }

  const systemPrompt = createSystemPrompt(nativeLanguage, targetLanguage, difficulty, additionalContext);

  const conversation = [
    { role: 'system', content: systemPrompt },
    ...messages,
  ] as OpenAI.Chat.ChatCompletionMessageParam[];

  try {

    const modelParameters = {
      temperature: 0.7,
    }
    
    const generation = trace.generation({
      id: uuidv4(),
      model: 'gpt-4o-mini',
      messages: parseOpenAIMessages(conversation).messages,
      modelParameters: modelParameters,
      provider: 'openai',
    });

    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: conversation,
      ...modelParameters,
    }, {
      headers: { "maxim-trace-id": trace.id }
    });
    
    generation.result({
      id: uuidv4(),
      object: "chat.completion",
      created: Date.now(),
      model: 'gpt-4o-mini',
      choices: convertOpenAIChoicesToMaxim(response.choices),
      usage: response.usage ? {
        prompt_tokens: response.usage.prompt_tokens,
        completion_tokens: response.usage.completion_tokens,
        total_tokens: response.usage.total_tokens,
      } : {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
      },
    })


    trace.end();
    return response.choices[0].message.content;
  } catch (error) {
    console.error('Error calling OpenAI:', error);
    throw new Error('Failed to get response from OpenAI');
  }
};

const createSystemPrompt = (
  nativeLanguage: string,
  targetLanguage: string,
  difficulty: string,
  additionalContext: string = ''
): string => {
  let difficultyInstruction = '';

  switch (difficulty) {
    case 'beginner':
      difficultyInstruction = `The user is a beginner in ${targetLanguage}. Use simple vocabulary and basic grammar structures. Explain concepts clearly in ${nativeLanguage} when necessary, but encourage usage of ${targetLanguage}. Focus on basic greetings, numbers, and common phrases.`;
      break;
    case 'intermediate':
      difficultyInstruction = `The user is at an intermediate level in ${targetLanguage}. You can use more complex sentences and grammar. Explanations in ${nativeLanguage} should be minimal, primarily for complex nuances. Focus on conversational flow and expanding vocabulary.`;
      break;
    case 'advanced':
      difficultyInstruction = `The user is an advanced learner of ${targetLanguage}. Converse almost exclusively in ${targetLanguage}. Use sophisticated vocabulary, idioms, and complex grammatical structures. Only use ${nativeLanguage} for very subtle linguistic distinctions or if explicitly asked.`;
      break;
    default:
      difficultyInstruction = `Adjust your teaching to the user's level.`;
  }

  let prompt = `You are an expert language teacher. Your goal is to teach the user ${targetLanguage}. The user is fluent in ${nativeLanguage}.
  
  ${difficultyInstruction}
  
  Be patient, encouraging, and helpful. Correct mistakes gently by providing the correct form and briefly explaining why, if appropriate for their level.
  Engage in a conversation to help them practice. Ask questions to prompt them to use the language.
  
  If the user asks a question in ${nativeLanguage}, answer it but try to bridge it back to ${targetLanguage}.
  `;

  if (additionalContext) {
    prompt += `\n\nHere is the official reference material for ${targetLanguage}. Use this strictly for vocabulary and grammar rules:\n\n${additionalContext}`;
  }

  return prompt;
};
