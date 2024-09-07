import networkx as nx
import logging

logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self, graph_creator):
        self.graph_creator = graph_creator
        self.llm_interface = graph_creator.llm_interface

    def process_query(self, query: str):
        logger.info(f"Processing query: {query}")
        
        # Extract key entities from the query
        query_entities = self.extract_entities_from_query(query)
        logger.info(f"Extracted entities from query: {query_entities}")
        
        # Find relevant subgraph
        relevant_subgraph = self.get_relevant_subgraph(query_entities)
        logger.info(f"Relevant subgraph contains {len(relevant_subgraph.nodes)} nodes and {len(relevant_subgraph.edges)} edges")
        
        # Generate context from relevant subgraph
        context = self.generate_context_from_subgraph(relevant_subgraph)
        logger.info("Generated context from subgraph")
        
        # Process query with context
        final_answer = self.llm_interface.process_query(query, context)
        logger.info("Generated final answer")
        
        return final_answer

    def extract_entities_from_query(self, query: str):
        # Use the LLM to extract entities from the query
        entities, _ = self.llm_interface.extract_entities_and_relations(query)
        return entities

    def get_relevant_subgraph(self, query_entities):
        G = self.graph_creator.G
        relevant_nodes = set()
        
        for entity in query_entities:
            if entity in G:
                relevant_nodes.add(entity)
                relevant_nodes.update(G.neighbors(entity))
        
        return G.subgraph(relevant_nodes)

    def generate_context_from_subgraph(self, subgraph):
        context_parts = []
        
        for node in subgraph.nodes:
            node_info = f"Entity: {node}\n"
            node_info += f"Connections:\n"
            
            for neighbor in subgraph.neighbors(node):
                edge_data = subgraph[node][neighbor]
                relation = edge_data.get('type', 'related to')
                node_info += f"- {relation} {neighbor}\n"
            
            context_parts.append(node_info)
        
        full_context = "\n".join(context_parts)
        return self.llm_interface.generate_summary(full_context)

    def get_community_context(self, nodes):
        # For simplicity, we're just joining node names. In a real implementation,
        # you'd want to fetch actual content associated with these nodes.
        return " ".join(nodes)