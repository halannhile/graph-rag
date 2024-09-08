import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import logging
logger = logging.getLogger(__name__)

class LLMInterface:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=api_key)

    def extract_elements(self, text):
        prompt = f"""
        Extract entities and relationships from the following text. 
        Format the output as a JSON array of objects. Each object should have a 'type' field ('entity' or 'relationship') and other relevant fields.
        For entities, include 'id' and 'name' fields.
        For relationships, include 'source', 'target', and 'type' fields.

        Text: {text}

        Output the JSON array only, with no additional text.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts structured information from text."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.choices[0].message.content.strip()
            logger.info(f"LLM response: {content}")
            elements = json.loads(content)
            
            if not isinstance(elements, list):
                logger.error(f"Expected a JSON array from LLM response, got: {type(elements)}")
                return []
            
            return elements
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from LLM response: {e}")
            logger.error(f"LLM response content: {content}")
            return []
        except Exception as e:
            logger.error(f"Error in extract_elements: {e}", exc_info=True)
            return []

    def generate_element_summary(self, element_data):
        prompt = f"Summarize the following element: {json.dumps(element_data)}"
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes information."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in generate_element_summary: {e}")
            return ""

    def generate_community_summary(self, elements):
        prompt = f"Summarize the following group of related elements: {json.dumps(elements)}"
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes groups of related information."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in generate_community_summary: {e}")
            return ""

    def process_query(self, query, context):
        prompt = f"Context: {context}\n\nQuery: {query}"
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in process_query: {e}")
            return ""