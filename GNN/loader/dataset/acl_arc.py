import os
import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.transforms as T

os.environ["TRANSFORMERS_SAFE_LOAD"] = "1"
# Define the intent mapping at the top level, as it's a constant
INTENT_TO_INT_ACL = {
    "Background": 0,
    "Uses": 1,
    "Future": 2,
    "CompareOrContrast": 3,
    "Motivation": 4,
    "Extends": 5,
    # Add special types for author and venue relations
    "author": 6,
    "venue": 7
}

class ACLCitationDataset(InMemoryDataset):
    """
    A PyTorch Geometric dataset for the ACL-ARC citation network.

    This dataset represents a heterogeneous graph of papers, authors, and venues.
    It is designed for a relation prediction task, where the goal is to predict
    the intent of a citation between two papers.

    Args:
        root (str): Root directory where the dataset should be saved.
        lang_model_name (str): The name of the SentenceTransformer model to use
            for generating node and edge features.
        include_authors (bool): If True, adds author nodes and 'author_of' edges.
        include_venues (bool): If True, adds venue nodes and 'published_in' edges.
        transform (callable, optional): A function/transform that takes in an
            `torch_geometric.data.Data` object and returns a transformed version.
            The data object will be transformed before every access. (default: None)
        pre_transform (callable, optional): A function/transform that takes in
            an `torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before being saved to disk.
            (default: None)
    """
    def __init__(self, root, lang_model_name='/home/CitationIntent/.cache/huggingface/hub/models--allenai--specter2_base/snapshots/3447645e1def9117997203454fa4495937bfbd83', 
                include_authors=True, include_venues=True,
                transform=None, pre_transform=None):
        
        self.lang_model_name = lang_model_name
        self.include_authors = include_authors
        self.include_venues = include_venues
        
        # A unique name for the processed file based on options
        self.processed_name = f'data_auth_{include_authors}_venue_{include_venues}.pt'

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)

    @property 
    def raw_dir(self):
        return os.path.join(self.root, 'raw')

    @property
    def revised_dir(self):
        # The user should place the 'acl-arc' folder in the root directory
        return os.path.join(self.root, 'revised')

    @property
    def processed_dir(self):
        # We can have a subdirectory for processed files
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        # These are the files the loader expects to find in raw_dir
        return ['papers_full.jsonl', 'train.jsonl', 'dev.jsonl', 'test.jsonl']

    @property
    def revised_file_names(self):
        # These are the files the loader expects to find in revised_dir
        return ['papers_full.jsonl', 'train.jsonl', 'dev.jsonl', 'test.jsonl']

    @property
    def processed_file_names(self):
        # This is the file that will be saved in processed_dir
        return [self.processed_name]

    def download(self):
        # This dataset is not automatically downloadable.
        # You should place the 'acl-arc' directory containing the .jsonl files
        # into a 'raw' folder within your specified root directory.
        # e.g., <root>/raw/papers_full.jsonl
        raise IOError(
            f"Dataset not found. Please place the acl-arc files in {self.revised_dir}"
            )

    def process(self):
        # Check if raw files exist
        if not all(os.path.exists(os.path.join(self.revised_dir, f)) for f in self.revised_file_names):
            self.download()
            
        # 1. Load paper metadata first
        papers_meta = {}
        with open(os.path.join(self.revised_dir, 'papers_full.jsonl'), "r", encoding="utf-8") as f:
            for line in f:
                paper = json.loads(line.strip())
                if 'acl_id' in paper:
                    papers_meta[paper["acl_id"]] = {
                        'title': paper.get('title', ''),
                        'abstract': paper.get('abstract', ''),
                        # 'authors': paper.get('authors', []),
                        # 'venue': paper.get('venue', '')
                    }
        print(f"Loaded metadata for {len(papers_meta)} papers.")

        # --- Data Structures to build the graph ---
        node_map = {}  # Maps original ID (paper_id, author_id, venue_name) to a new integer node ID
        node_texts = [] # Text for each node (title for papers, empty for others)

        edge_list = [] # List of [src, dst] pairs
        edge_texts = [] # Text for each edge (citation context)
        edge_labels = [] # Integer label for each edge (citation intent)
        edge_type = [] # Integer label for each edge (citation intent)
        # --- Helper function to add nodes ---
        def add_node_if_not_exists(node_id, text=""):
            if node_id not in node_map:
                node_map[node_id] = len(node_texts)
                node_texts.append(text)
            return node_map[node_id]

        # 2. Process train, dev, and test splits to build one large graph
        splits = {'train': [], 'dev': [], 'test': []}
        for split_name in splits.keys():
            path = os.path.join(self.revised_dir, f'{split_name}.jsonl')
            
            print(f"Processing {path}...")
            with open(path, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc=f"Reading {split_name}"):
                    citation = json.loads(line.strip())
                    
                    # --- Add Paper Nodes ---
                    citing_id = citation["citing_id"]
                    cited_id = citation["cited_id"]

                    citing_title = papers_meta.get(citing_id, {}).get('title', citation["citing_title"])
                    citing_abstract = papers_meta.get(citing_id, {}).get('abstract', "")
                    cited_title = papers_meta.get(cited_id, {}).get('title', citation["cited_title"])
                    cited_abstract = papers_meta.get(cited_id, {}).get('abstract', "")

                    src_node_idx = add_node_if_not_exists(citing_id, citing_title + ' ' + citing_abstract)
                    dst_node_idx = add_node_if_not_exists(cited_id, cited_title + ' ' + cited_abstract)

                    # --- Add the primary citation edge ---
                    current_edge_index = len(edge_list)
                    splits[split_name].append(current_edge_index) # Track this edge for the mask
                    
                    edge_list.append([src_node_idx, dst_node_idx])
                    edge_texts.append(citation["text"])
                    edge_labels.append(INTENT_TO_INT_ACL[citation["intent"]])
                    edge_type.append(0)
                    # --- Optionally add Author and Venue nodes/edges ---
                    for paper_id in [citing_id, cited_id]:
                        paper_node_idx = node_map[paper_id]
                        meta = papers_meta.get(paper_id)
                        if not meta:
                            continue

                        # Add author nodes and edges
                        if self.include_authors:
                            for author in meta.get('authors', []):
                                if 'authorId' in author and author['authorId'] is not None:
                                    author_map_id = 'author_' + author['authorId']
                                    author_node_idx = add_node_if_not_exists(author_map_id, text=author['name'])
                                    # Edge: Author -> Paper
                                    edge_list.append([author_node_idx, paper_node_idx])
                                    edge_texts.append("") # No text for this relation type
                                    edge_labels.append(INTENT_TO_INT_ACL["author"])
                                    edge_type.append(1)
                        # Add venue nodes and edges
                        if self.include_venues and meta.get('venue'):
                            venue_name = meta['venue']
                            venue_node_idx = add_node_if_not_exists(venue_name, text=venue_name)
                            # Edge: Venue -> Paper
                            edge_list.append([venue_node_idx, paper_node_idx])
                            edge_texts.append("") # No text for this relation type
                            edge_labels.append(INTENT_TO_INT_ACL["venue"])
                            edge_type.append(2)
        # 3. Create edge masks for the relation prediction task
        num_edges = len(edge_list)
        train_mask = torch.zeros(num_edges, dtype=torch.bool)
        val_mask = torch.zeros(num_edges, dtype=torch.bool)
        test_mask = torch.zeros(num_edges, dtype=torch.bool)

        train_mask[torch.tensor(splits['train'])] = True
        val_mask[torch.tensor(splits['dev'])] = True
        test_mask[torch.tensor(splits['test'])] = True

        train_edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        val_edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        test_edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        train_edge_index = train_edge_index[:,train_mask]
        val_edge_index = val_edge_index[:,val_mask]
        test_edge_index = test_edge_index[:,test_mask]

        train_edge_label = torch.tensor(edge_labels, dtype=torch.long)[train_mask]
        val_edge_label = torch.tensor(edge_labels, dtype=torch.long)[val_mask]
        test_edge_label = torch.tensor(edge_labels, dtype=torch.long)[test_mask]
        
        train_edge_label = torch.where(
            (train_edge_label == INTENT_TO_INT_ACL["author"]) | (train_edge_label == INTENT_TO_INT_ACL["venue"]),
            0,
            train_edge_label
        )
        val_edge_label = torch.where(
            (val_edge_label == INTENT_TO_INT_ACL["author"]) | (val_edge_label == INTENT_TO_INT_ACL["venue"]),
            0,
            val_edge_label
        )
        test_edge_label = torch.where(
            (test_edge_label == INTENT_TO_INT_ACL["author"]) | (test_edge_label == INTENT_TO_INT_ACL["venue"]),
            0,
            test_edge_label
        )

        edge_type = torch.tensor(edge_type, dtype=torch.long)
        print(f"Total nodes: {len(node_texts)}")
        print(f"Total edges: {len(edge_list)}")
        print(f"Train/Val/Test citation edges: {len(splits['train'])}/{len(splits['dev'])}/{len(splits['test'])}")
        
        # 4. Generate embeddings (this is the expensive pre-processing part)
        print(f"Loading sentence transformer: '{self.lang_model_name}'...")
        lang_model = SentenceTransformer(self.lang_model_name, device='cuda' if torch.cuda.is_available() else 'cpu')

        print("Encoding node features (titles)...")
        x = lang_model.encode(node_texts, show_progress_bar=True, convert_to_tensor=True)
        
        print("Encoding edge features (citation contexts)...")
        edge_attr = lang_model.encode(edge_texts, show_progress_bar=True, convert_to_tensor=True)

        # 5. Create the PyG Data object
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        y = torch.tensor(edge_labels, dtype=torch.long)

        data = Data(
            x=x.cpu(),  # Node features
            edge_index=edge_index.cpu(),
            edge_attr=edge_attr.cpu(), # Edge features
            y=y.cpu(), # Edge labels
            edge_type=edge_type.cpu(),
            # train_mask=train_mask.cpu(),
            # val_mask=val_mask.cpu(),
            # test_mask=test_mask.cpu(),
            train_edge_index=train_edge_index.cpu(),
            val_edge_index=val_edge_index.cpu(),
            test_edge_index=test_edge_index.cpu(),
            train_edge_label=train_edge_label.cpu(),
            val_edge_label=val_edge_label.cpu(),
            test_edge_label=test_edge_label.cpu()
        )

        # Apply the pre-transform if it exists (e.g., T.ToSparseTensor())
        if self.pre_transform is not None:
            print("Applying pre-transform...")
            data = self.pre_transform(data)

        print("Saving processed data object...")
        torch.save(self.collate([data]), self.processed_paths[0])
        print("Done.")
        
if __name__ == '__main__':
    # Make sure you have a directory structure like this:
    # ./my_dataset/
    # └── raw/
    #     ├── papers_full.jsonl
    #     ├── train.jsonl
    #     ├── dev.jsonl
    #     └── test.jsonl
    # --- Option 1: Load with authors and venues ---
    # The first time this is run, it will execute the `process` method.
    # Subsequent runs will load the saved file instantly.
    root_path = "./datasets/acl-arc" # Directory to store processed data
    print(os.path.abspath(root_path))
    print("--- Loading dataset with authors and venues ---")
    dataset_full = ACLCitationDataset(
        root=root_path,
        include_authors=True,
        include_venues=True,
        pre_transform=T.ToSparseTensor(),
    )
    
    # Get the single graph object
    graph_full = dataset_full[0]
    print("\nFull Graph Object:")
    print(graph_full)
    print(f"Number of training edges: {graph_full.train_mask.sum()}")
    print(f"Number of validation edges: {graph_full.val_mask.sum()}")
    print(f"Number of testing edges: {graph_full.test_mask.sum()}")
    
    # --- Option 2: Load without authors and venues ---
    # This will trigger a new processing run because the options are different,
    # resulting in a different processed file name.
    print("\n\n--- Loading dataset with only papers ---")
    dataset_papers_only = ACLCitationDataset(
        root=root_path,
        include_authors=False,
        include_venues=False
    )
    
    graph_papers_only = dataset_papers_only[0]
    print("\nPapers-Only Graph Object:")
    print(graph_papers_only)
    
    # Example of how you would use this in a training loop with GraphGym or PyG
    # The masks select the edges for training, validation, or testing.
    # For example, to get the training labels:
    train_edge_labels = graph_full.y[graph_full.train_mask]
    print(f"\nLabels for the {len(train_edge_labels)} training edges.")