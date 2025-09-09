"""
Data loading utilities for HuggingFace datasets and local data.
"""

from typing import Optional, Dict, Any, Union
from datasets import Dataset, load_dataset, load_from_disk
import pandas as pd
import torch
from torch.utils.data import DataLoader


class DatasetLoader:
    """Utility class for loading datasets from various sources."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dataset loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def load_hf_dataset(
        self, 
        dataset_name: str, 
        split: Optional[str] = None,
        **kwargs
    ) -> Dataset:
        """
        Load dataset from HuggingFace Hub.
        
        Args:
            dataset_name: Name of the dataset on HuggingFace Hub
            split: Dataset split to load (train, test, validation)
            **kwargs: Additional arguments for load_dataset
            
        Returns:
            Loaded dataset
        """
        dataset = load_dataset(dataset_name, split=split, **kwargs)
        return dataset
    
    def load_local_dataset(self, data_path: str) -> Dataset:
        """
        Load dataset from local disk.
        
        Args:
            data_path: Path to the local dataset
            
        Returns:
            Loaded dataset
        """
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            dataset = Dataset.from_pandas(df)
        elif data_path.endswith('.json') or data_path.endswith('.jsonl'):
            dataset = Dataset.from_json(data_path)
        else:
            # Assume it's a saved HuggingFace dataset
            dataset = load_from_disk(data_path)
        
        return dataset
    
    def create_dataloader(
        self, 
        dataset: Dataset, 
        batch_size: int = 32,
        shuffle: bool = True,
        **kwargs
    ) -> DataLoader:
        """
        Create PyTorch DataLoader from dataset.
        
        Args:
            dataset: Dataset to create DataLoader from
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            **kwargs: Additional arguments for DataLoader
            
        Returns:
            PyTorch DataLoader
        """
        def collate_fn(batch):
            # Simple collate function - customize as needed
            return {key: torch.stack([item[key] for item in batch]) 
                   for key in batch[0].keys()}
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=collate_fn,
            **kwargs
        )
        
        return dataloader