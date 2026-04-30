"""
Query decomposition using LLM.
Extracts positive and negative intents from text.
"""

import os
import json
import hashlib
import yaml
from tqdm import tqdm


class SubQueryExtractor:
    """
    Decomposes queries into positive/negative intents using LLM.
    Implements granular caching to avoid redundant LLM calls.
    """
    def __init__(self, llm_client, model_name, prompt_version="v1", dataset_name="NegationCLIP"):
        """
        Args:
            llm_client: LocalQwenClient or compatible LLM interface
            model_name: Name of LLM model
            prompt_version: Prompt template version (e.g., 'v3')
            dataset_name: For cache organization
        """
        self.llm_client = llm_client
        self.llm_model_name = model_name
        self.prompt_version = prompt_version
        self.dataset_name = dataset_name
        
        # Load prompt template
        prompt_path = f"./prompt/{prompt_version}.yaml"
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r') as f:
                self.prompt_config = yaml.safe_load(f)
        else:
            # Fallback prompt
            self.prompt_config = {
                'system': 'You are a helpful assistant. Extract positive and negative intents from queries.'
            }
            
        self.cache_dir = f"./llm_cache/{dataset_name}/{prompt_version}"
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, query):
        """Generate unique cache path for query"""
        unique_id = f"{query}_{self.llm_model_name}"
        query_hash = hashlib.md5(unique_id.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{query_hash}.json")

    def save_to_cache(self, query, parsed_json):
        """Save decomposition result to cache"""
        cache_path = self._get_cache_path(query)
        result = {
            "query": query,
            "positives": parsed_json.get("positives", []),
            "negatives": parsed_json.get("negatives", []),
            "metadata": {
                "llm_model": self.llm_model_name,
                "prompt_version": self.prompt_version,
                "dataset": self.dataset_name
            }
        }
        with open(cache_path, 'w') as f:
            json.dump(result, f, indent=4)
        return result

    def get_decomposition_batch(self, queries):
        """
        Process batch of queries, using cache for hits.
        
        Args:
            queries: List of query strings
        
        Returns:
            List of decomposition results with same order as input
        """
        needed_queries = []
        results = {}
        
        # Check cache
        for q in queries:
            path = self._get_cache_path(q)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    results[q] = json.load(f)
            else:
                needed_queries.append(q)
        
        # Process uncached queries with LLM
        if needed_queries:
            system_msg = self.prompt_config['system']
            batch_responses = self.llm_client.generate_batch(system_msg, needed_queries)
            for q, resp in zip(needed_queries, batch_responses):
                results[q] = self.save_to_cache(q, resp)
                
        return [results[q] for q in queries]

    def extract_all_queries(self, dataloader, batch_size=16):
        """
        Extract and cache decompositions for all queries in dataloader.
        
        Args:
            dataloader: DataLoader with 'pos_text'/'neg_text' or 'text' keys
            batch_size: Batch size for LLM inference
        
        Returns:
            dict: Statistics about extraction
        """
        print(f"Starting decomposition for {len(dataloader.dataset)} samples...")
        
        # Collect queries
        all_queries = []
        dataset = dataloader.dataset
        
        if hasattr(dataset, '__getitem__'):
            sample = dataset[0]
            if "pos_text" in sample and "neg_text" in sample:
                for i in range(len(dataset)):
                    item = dataset[i]
                    all_queries.append(item['pos_text'])
                    all_queries.append(item['neg_text'])
            elif "text" in sample:
                for i in range(len(dataset)):
                    all_queries.append(dataset[i]['text'])
        
        # Find uncached queries
        unique_queries = list(set(all_queries))
        queries_to_process = [q for q in unique_queries if not os.path.exists(self._get_cache_path(q))]
        
        cached_count = len(unique_queries) - len(queries_to_process)
        print(f"Total Unique: {len(unique_queries)} | Cached: {cached_count} | To Process: {len(queries_to_process)}")

        if not queries_to_process:
            print("All queries already cached.")
            return {"new_calls": 0, "cached": cached_count, "errors": 0}

        # Process in batches
        stats = {"new_calls": 0, "errors": 0}
        
        for i in tqdm(range(0, len(queries_to_process), batch_size), desc="LLM Processing"):
            chunk = queries_to_process[i : i + batch_size]
            
            try:
                self.get_decomposition_batch(chunk)
                stats["new_calls"] += len(chunk)
            except Exception as e:
                print(f"\nError at batch {i}: {e}")
                stats["errors"] += len(chunk)

        print(f"\n--- Extraction Summary ---")
        print(f"Total Unique: {len(unique_queries)}")
        print(f"Cached: {cached_count}")
        print(f"Newly Extracted: {stats['new_calls']}")
        print(f"Errors: {stats['errors']}")
        
        return {**stats, "cached": cached_count, "total": len(unique_queries)}
