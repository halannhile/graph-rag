import networkx as nx
from community import community_louvain
import logging
logger = logging.getLogger(__name__)

class GraphCreator:
    def __init__(self, llm_interface):
        self.G = nx.Graph()
        self.llm_interface = llm_interface
        self.communities = {}
        self.community_summaries = {}
        self.element_summaries = {}

    def add_document_to_graph(self, chunks):
        logger.info(f"Processing document with {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            try:
                self.process_chunk(chunk)
                logger.info(f"Processed chunk {i+1}/{len(chunks)}")
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")

    def process_chunk(self, chunk):
        try:
            elements = self.llm_interface.extract_elements(chunk)
            logger.info(f"Extracted elements: {elements}")
            if not elements:
                logger.warning("No elements extracted from chunk")
                return
            for element in elements:
                self.add_element_to_graph(element)
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}", exc_info=True)

    def add_element_to_graph(self, element):
        try:
            if element['type'] == 'entity':
                self.G.add_node(element['id'], **element)
                logger.info(f"Added node: {element['id']}")
            elif element['type'] == 'relationship':
                self.G.add_edge(element['source'], element['target'], **element)
                logger.info(f"Added edge: {element['source']} -> {element['target']}")
            else:
                logger.warning(f"Unknown element type: {element['type']}")
        except KeyError as e:
            logger.error(f"Missing key in element: {str(e)}")
        except Exception as e:
            logger.error(f"Error adding element to graph: {str(e)}", exc_info=True)

    def create_element_summaries(self):
        logger.info("Creating element summaries")
        for node in self.G.nodes(data=True):
            node_id, node_data = node
            summary = self.llm_interface.generate_element_summary(node_data)
            self.element_summaries[node_id] = summary

        for edge in self.G.edges(data=True):
            source, target, edge_data = edge
            summary = self.llm_interface.generate_element_summary(edge_data)
            self.element_summaries[f"{source}-{target}"] = summary

    def create_communities(self):
        logger.info("Creating communities")
        self.communities = community_louvain.best_partition(self.G)

    def create_community_summaries(self):
        logger.info("Creating community summaries")
        community_elements = {}
        for node, community in self.communities.items():
            if community not in community_elements:
                community_elements[community] = []
            community_elements[community].append(self.element_summaries[node])

        for community, elements in community_elements.items():
            summary = self.llm_interface.generate_community_summary(elements)
            self.community_summaries[community] = summary

    def get_graph_data(self):
        data = {"nodes": [], "edges": []}
        for node, attrs in self.G.nodes(data=True):
            data["nodes"].append({"id": node, "label": attrs.get("name", node)})
        for source, target, attrs in self.G.edges(data=True):
            data["edges"].append({
                "from": source,
                "to": target,
                "label": attrs.get("type", "")
            })
        logger.info(f"Generated graph data with {len(data['nodes'])} nodes and {len(data['edges'])} edges")
        logger.debug(f"Graph data: {data}")
        return data