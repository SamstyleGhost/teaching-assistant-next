import fs from 'fs';
import { PDFParse } from 'pdf-parse'

export const readPdfContent = async (filePath: string): Promise<string> => {
  try {
    const dataBuffer = fs.readFileSync(filePath);
    const parser = new PDFParse(dataBuffer);
    const data = await parser.getText();

    return data.text;
  } catch (error) {
    console.error('Error reading PDF file:', error);
    throw new Error('Failed to read PDF content');
  }
};