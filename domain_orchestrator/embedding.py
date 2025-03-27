from transformers import CLIPModel, CLIPProcessor
from abc import abstractmethod
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image


#abstract class
class EmbeddingModel:
    """Abstract class for embedding models."""
    @abstractmethod
    def embed_image(self, image_path):
        """Embed a single image."""
        pass


class ClipEmbeddingModel(EmbeddingModel):
    """Handles image and dataset embedding operations."""

    def __init__(self):
        self.embedding_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to("cuda")
        self.embedding_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

    def embed_image(self, image_path) -> npt.NDArray:
        """Embed a single image."""
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file '{image_path}' not found.")
            raise
        except Exception as e:
            print(f"Error opening image '{image_path}': {e}")
            raise

        if self.embedding_processor is None or self.embedding_model is None:
            print("Error: CLIP model or processor is not initialized.")
            raise
        
        inputs = self.embedding_processor(images=image, return_tensors="pt").to("cuda")

        # Generate image embeddings
        with torch.no_grad():
            image_embeddings = (
                self.embedding_model.get_image_features(**inputs).detach().cpu().numpy()
            )
        return image_embeddings


class EmbeddingManager:
    """Handles image and dataset embedding operations."""
    
    def __init__(self, embedding_model: EmbeddingModel = ClipEmbeddingModel()):
        self.embedding_model = embedding_model
    
    def embed_image(self, image_path) -> npt.NDArray:
        """Embed a single image."""
        return self.embedding_model.embed_image(image_path)
        
    def embed_dataset(self, dataset_path, debug=False) -> npt.NDArray:
        """Embed all images in a dataset."""
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path '{dataset_path}' not found.")

        print(f"Embedding dataset from '{dataset_path}' ...")
        dataset_embeddings = []

        # IDD has both png and jpg images in train set
        image_files = list(dataset_path.rglob("*.png")) + list(dataset_path.rglob("*.jpg"))
    
        if not image_files:
            print(f"Warning: No images found in dataset path '{dataset_path}'.")
            return []

        for img in image_files:
            embedding = self.embed_image(img)
            if embedding is not None:
                dataset_embeddings.append(embedding)
            else:
                raise ValueError(f"Error embedding image '{img}'.")

        print("Finished embedding dataset.")
        return dataset_embeddings
        
    def calculate_statistics(self, domain_name, domain_path, train_path):

        """
        Calculate or load domain statistics.
        Args:
            domain_name (str): The name of the domain.
            domain_path (Path): The path to the domain database where the statistics will be saved.
            train_path (Path): The path to the train set.
        Returns:
            dict: A dictionary containing the statistics.
        """
        suffix = "_statistics.npz"
        statistics_path = domain_path / f"{domain_name}{suffix}"
        stats_dict = {}

        print(f"Statistics file: {statistics_path}")
        if statistics_path.exists():  # Load the data if it exists
            try:
                print(f"Loading statistics from {domain_name}{suffix} ...")
                stats = np.load(statistics_path)
                stats_dict.update({
                    "train_average_embedding": stats["train_average_embedding"],
                })
                print(f"Statistics loaded from {domain_name}{suffix}")
                return stats_dict
            except Exception as e:
                print(f"Error loading statistics file '{statistics_path}': {e}")
                return None

        print(f"Statistics file {statistics_path} does not exist, calculating statistics for domain '{domain_name}' ...")
        train_dataset_embeddings = self.embed_dataset(train_path)

        if not train_dataset_embeddings:
            raise ValueError("No embeddings were generated for dataset.")

        try:
            train_average_embedding = np.mean(train_dataset_embeddings, axis=0)
        except Exception as e:
            print(f"Error computing mean embedding: {e}")
            raise

        stats_dict.update({
            "train_average_embedding": train_average_embedding
        })

        try:
            np.savez(
                statistics_path,
                train_average_embedding=train_average_embedding,
            )
            print(f"Statistics saved to {domain_name}{suffix}")
        except Exception as e:
            print(f"Error saving statistics file '{statistics_path}': {e}")
            raise
        return stats_dict
