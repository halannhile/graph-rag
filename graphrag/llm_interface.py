import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

class LLMInterface:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def extract_entities_and_relations(self, text: str):
        if not self.client.api_key:
            raise ValueError("OpenAI API key is not set. Please check your .env file.")

        prompt = f"""
        Extract entities and relations from the following text. 
        Format the output exactly as follows:
        
        Entities:
        - Entity1
        - Entity2
        ...

        Relations:
        - Entity1, relation, Entity2
        - Entity3, relation, Entity4
        ...

        Text: {text}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts entities and relations from text."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            output = response.choices[0].message.content.strip()
            
            # Split the output into entities and relations sections
            sections = output.split("\n\n")
            
            entities = []
            relations = []
            
            for section in sections:
                if section.startswith("Entities:"):
                    entities = [e.strip('- ').strip() for e in section.split('\n')[1:] if e.strip()]
                elif section.startswith("Relations:"):
                    for line in section.split('\n')[1:]:
                        if line.strip():
                            parts = [p.strip() for p in line.strip('- ').split(',')]
                            if len(parts) == 3:
                                relations.append(tuple(parts))
                            else:
                                print(f"Warning: Skipping invalid relation: {line}")
            
            return entities, relations
        except Exception as e:
            raise ValueError(f"Error calling OpenAI API: {str(e)}")

    def generate_summary(self, text: str):
        if not self.client.api_key:
            raise ValueError("OpenAI API key is not set. Please check your .env file.")

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                    {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
                ]
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise ValueError(f"Error calling OpenAI API: {str(e)}")

    def process_query(self, query: str, context: str):
        if not self.client.api_key:
            raise ValueError("OpenAI API key is not set. Please check your .env file.")

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
                ]
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise ValueError(f"Error calling OpenAI API: {str(e)}")