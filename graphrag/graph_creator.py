import networkx as nx
from community import community_louvain
import logging

logger = logging.getLogger(__name__)

class GraphCreator:
    def __init__(self, llm_interface):
        self.G = nx.Graph()
        self.llm_interface = llm_interface

    def create_graph(self, chunks):
        logger.info(f"Starting graph creation with {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            try:
                self.process_chunk_for_graph(chunk)
                logger.info(f"Processed chunk {i+1}/{len(chunks)}")
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                continue
        
        if self.G.number_of_nodes() > 0:
            logger.info("Performing community detection")
            partition = community_louvain.best_partition(self.G)
            nx.set_node_attributes(self.G, partition, "community")
        else:
            logger.warning("No nodes in graph, skipping community detection")
        
        logger.info(f"Graph creation complete. Nodes: {self.G.number_of_nodes()}, Edges: {self.G.number_of_edges()}")
        return self.G

    def process_chunk_for_graph(self, chunk: str):
        entities, relations = self.llm_interface.extract_entities_and_relations(chunk)
        
        logger.info(f"Extracted {len(entities)} entities and {len(relations)} relations")
        
        # Add entities as nodes
        for entity in entities:
            if entity not in self.G:
                self.G.add_node(entity)
        
        # Add relations as edges
        for relation in relations:
            if len(relation) != 3:
                logger.warning(f"Skipping invalid relation: {relation}")
                continue
            source, relation_type, target = relation
            if self.G.has_edge(source, target):
                # If edge exists, update weight
                self.G[source][target]['weight'] += 1
            else:
                # If edge doesn't exist, create it with weight 1
                self.G.add_edge(source, target, weight=1, type=relation_type)

    def get_graph_data(self):
        data = nx.node_link_data(self.G)
        # Ensure nodes have labels
        for node in data['nodes']:
            node['label'] = node['id']
        # Ensure edges have labels
        for edge in data['links']:
            edge['label'] = self.G[edge['source']][edge['target']].get('type', '')
        return data

    def get_community_summaries(self):
        communities = {}
        for node, data in self.G.nodes(data=True):
            community = data.get('community', 0)  # Default to community 0 if not set
            if community not in communities:
                communities[community] = []
            communities[community].append(node)
        
        return communities