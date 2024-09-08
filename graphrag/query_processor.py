import logging

logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self, graph_creator):
        self.graph_creator = graph_creator
        self.llm_interface = graph_creator.llm_interface

    def process_query(self, query: str):
        logger.info(f"Processing query: {query}")
        
        # Get relevant community summaries
        relevant_summaries = self.get_relevant_community_summaries(query)
        
        # Generate community answers
        community_answers = self.generate_community_answers(query, relevant_summaries)
        
        # Generate global answer
        global_answer = self.generate_global_answer(query, community_answers)
        
        return global_answer

    def get_relevant_community_summaries(self, query):
        # Implement logic to find relevant community summaries based on the query
        # This could involve semantic similarity or keyword matching
        # For simplicity, we'll return all community summaries for now
        return list(self.graph_creator.community_summaries.values())

    def generate_community_answers(self, query, relevant_summaries):
        answers = []
        for summary in relevant_summaries:
            answer = self.llm_interface.process_query(query, summary)
            answers.append(answer)
        return answers

    def generate_global_answer(self, query, community_answers):
        context = "\n".join(community_answers)
        return self.llm_interface.process_query(query, context)