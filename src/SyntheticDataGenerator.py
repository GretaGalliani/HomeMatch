import os
import sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Optional
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI
import chromadb

# Set up source path
source = os.popen("git rev-parse --show-toplevel").read().strip("\n")
sys.path.insert(0, source)

from src.Configurator import Configurator

class SyntheticDataGenerator:
    """
    A class to generate synthetic property listings using a Large Language Model (LLM).

    This class handles the end-to-end process of generating synthetic property listings,
    including data loading, LLM-based text generation, embedding computation, and storage
    in both CSV and vector database formats.

    Attributes:
        config (Configurator): Configuration object containing API keys, model parameters,
            file paths, and other settings.
        n_listings (int): Number of synthetic listings to generate.
        verbose (int): Verbosity level for logging (0: minimal, 1: detailed).
        house_df (pd.DataFrame): Complete dataset of property information.
        df_house_sampled (pd.DataFrame): Randomly sampled subset of property data.
        df_listings (pd.DataFrame): Generated property listings.

    Example:
        >>> config = Configurator("config.yaml")
        >>> generator = SyntheticDataGenerator(config, n_listings=100, verbose=1)
        >>> generator.load_house_data()
        >>> generator.generate_listings()
        >>> generator.save_listings()
        >>> generator.save_embeddings_chromadb()
    """

    def __init__(self, config: Configurator, n_listings: int, verbose: int = 0):
        """
        Initialize the SyntheticDataGenerator.

        Args:
            config (Configurator): Configuration object for the generator.
            n_listings (int): Number of synthetic listings to generate.
            verbose (int, optional): Verbosity level. Defaults to 0.
        """
        self.config = config
        self.n_listings = n_listings
        self.verbose = verbose
        self.house_df: Optional[pd.DataFrame] = None
        self.df_house_sampled: Optional[pd.DataFrame] = None
        self.df_listings: Optional[pd.DataFrame] = None

    # Data Loading and Processing Methods
    def load_house_data(self) -> None:
        """
        Load property data from the configured CSV file.

        The file path is retrieved from the config using the 'house_data' key.
        """
        self.house_df = pd.read_csv(self.config.get("house_data"))

        if self.verbose:
            print(f'House data loaded from {self.config.get("house_data")}')
            print(f"Total properties loaded: {len(self.house_df)}")
            print("Dataset sample:")
            display(self.house_df.head())

    # Core Generation Methods
    def generate_listings(self) -> pd.DataFrame:
        """
        Generate synthetic property listings using the configured LLM.

        This method:
        1. Samples the specified number of properties from the dataset
        2. Loads and configures the prompt template
        3. Generates listings using the LLM for each property
        4. Compiles the results into a DataFrame

        Returns:
            pd.DataFrame: DataFrame containing generated property listings with columns:
                - id: property identifier
                - listing: generated listing text

        Raises:
            ValueError: If house data hasn't been loaded yet.
        """
        if self.house_df is None:
            raise ValueError("House data must be loaded first. Call load_house_data()")

        random.seed(1807)
        self.df_house_sampled = self.house_df.sample(self.n_listings)

        # Load and define prompt template
        prompt_template_txt = self._load_listing_prompt_template()
        property_prompt = self._define_langchain_prompt_template(prompt_template_txt)

        # Initialize LLM and listing DataFrame
        llm = ChatOpenAI(**self.config.get("open_ai_parameters"))
        df_listings = pd.DataFrame()

        # Generate listings
        for _, row in tqdm(
            self.df_house_sampled.iterrows(),
            total=self.n_listings,
            desc="Generating listings",
        ):
            query = property_prompt.format(**row)
            llm_result = llm.invoke(query)
            df_act = self._listing_to_dataframe(id=row["id"], listing=llm_result)
            df_listings = pd.concat([df_listings, df_act], ignore_index=True)

        self.df_listings = df_listings

        if self.verbose:
            print(f"Generated {self.n_listings} listings")
            print("\nExample:")
            print("Input property data:")
            print(self.df_house_sampled.iloc[0])
            print("\nGenerated listing:")
            print(self.df_listings.listing.values[0])

        return self.df_listings

    # Storage Methods
    def save_listings(self) -> None:
        """
        Save generated listings to a CSV file.

        The file path is retrieved from the config using the 'listings' key.

        Raises:
            ValueError: If listings haven't been generated yet.
        """
        if self.df_listings is None:
            raise ValueError("No listings to save. Call generate_listings() first")

        self.df_listings.to_csv(self.config.get("listings"), index=False)

        if self.verbose:
            print(f'Listings saved to {self.config.get("listings")}')

    def save_embeddings_chromadb(self) -> None:
        """
        Compute embeddings for listings and store them in ChromaDB.

        This method:
        1. Initializes a ChromaDB client
        2. Computes embeddings for all listings
        3. Creates or recreates a ChromaDB collection
        4. Stores the listings with their embeddings and metadata

        Raises:
            ValueError: If listings haven't been generated yet.
        """
        if self.df_listings is None:
            raise ValueError("No listings to embed. Call generate_listings() first")

        # Initialize ChromaDB client and compute embeddings
        client = chromadb.PersistentClient(
            path=self.config.get("chromadb_persist_directory")
        )
        embeddings = self._get_embeddings(self.df_listings.listing.values)

        # Create/recreate collection
        collection = self._create_chromadb_collection(
            client=client,
            collection_name=self.config.get("property_listings_collection"),
        )

        # Add documents and embeddings to ChromaDB
        for i, row in tqdm(
            self.df_listings.iterrows(),
            total=len(self.df_listings),
            desc="Adding to ChromaDB",
        ):
            collection.add(
                documents=row["listing"],
                embeddings=embeddings[i],
                metadatas={"id": row["id"]},
                ids=str(row["id"]),
            )

        if self.verbose:
            print(
                f"Embeddings saved in ChromaDB {self.config.get('property_listings_collection')} collection"
            )

    # Helper Methods
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for a list of texts using OpenAI's embedding model.

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            np.ndarray: Array of embeddings, shape (n_texts, embedding_dim).
        """
        # Normalize text by replacing newlines with spaces
        texts = [text.replace("\n", " ") for text in texts]

        client = OpenAI(
            base_url=os.environ["OPENAI_API_BASE"], api_key=os.environ["OPENAI_API_KEY"]
        )

        response = client.embeddings.create(
            input=texts, model=self.config.get("embedding_model")
        )

        return np.array([item.embedding for item in response.data])

    def _create_chromadb_collection(
        self, client: chromadb.Client, collection_name: str
    ) -> chromadb.Collection:
        """
        Create or recreate a ChromaDB collection.

        Args:
            client (chromadb.Client): Initialized ChromaDB client.
            collection_name (str): Name for the collection.

        Returns:
            chromadb.Collection: The created collection.
        """
        # Check if the collection exists
        existing_collections = [col.name for col in client.list_collections()]

        if collection_name in existing_collections:
            # Delete if it exists
            client.delete_collection(collection_name)
            if self.verbose:
                print(f"Collection '{collection_name}' deleted")

        # Create the collection
        collection = client.create_collection(name=collection_name)

        if self.verbose:
            print(f"Collection '{collection_name}' created")

        return collection

    def _load_listing_prompt_template(self) -> str:
        """
        Load the prompt template for generating house listings.

        Returns:
            str: Content of the prompt template file.
        """
        with open(self.config.get("generate_property_listing_template"), "r") as f:
            return f.read()

    def _define_langchain_prompt_template(
        self, prompt_template_txt: str
    ) -> PromptTemplate:
        """
        Create a LangChain PromptTemplate for property listings.

        Args:
            prompt_template_txt (str): Raw template text.

        Returns:
            PromptTemplate: Configured template ready for use.
        """
        return PromptTemplate(
            input_variables=self.config.get("generate_property_listing_input_prompt_variables"),
            template=prompt_template_txt,
        )

    def _listing_to_dataframe(self, id: int, listing: str) -> pd.DataFrame:
        """
        Convert a generated listing into a DataFrame row.

        Args:
            id (int): Property identifier.
            listing (str): Generated listing text.

        Returns:
            pd.DataFrame: Single-row DataFrame with the listing data.
        """
        return pd.DataFrame({"id": [id], "listing": [listing.content]})
