def calc_dist(a: list[float], b: list[float], dist: str = "cosine") -> float:
    """Calculate the distance between two vectors.

    Args:
        a (list[float]): The first vector.
        b (list[float]): The second vector.
        dist (str): Distance function. Can be "l2sqr", "l2" or "cosine". (default: "cosine", for RAG)
    """
    ...

class RagVecDB:
    """A vector database for RAG using HNSW index."""

    def __init__(
        self,
        dim: int,
        dist: str = "cosine",
        ef_construction: int = 200,
        M: int = 16,
        max_elements: int = 0,
        seed: int | None = None,
    ) -> None:
        """Create a new HNSW index.

        Args:
            dim (int): Dimension of the vectors.
            dist (str): Distance function. Can be "l2sqr", "l2" or "cosine". (default: "cosine", for RAG)
            ef_construction (int): Number of elements to consider during construction. (default: 200)
            M (int): Number of neighbors to consider during search. (default: 16)
            max_elements (int): The initial capacity of the index. (default: 0, auto-grow)
            seed (int | None): Random seed for the index. (default: None, random)

        Random seed will never be saved. Never call `add` on a loaded index if you want to have deterministic index construction.
        """
        ...

    @staticmethod
    def load(path: str) -> "RagVecDB":
        """Load an existing HNSW index from disk."""
        ...

    def save(self, path: str) -> None:
        """Save the HNSW index to disk. The random seed is not saved."""
        ...

    def add(self, vec: list[float], metadata: dict[str, str]) -> int:
        """Add a vector to the index. Use `batch_add` for better performance.

        Returns:
            ID of the added vector.
        """
        ...

    def batch_add(
        self, vec_list: list[list[float]], metadata_list: list[dict[str, str]]
    ) -> list[int]:
        """Add multiple vectors to the index.

        Returns:
            List of IDs of the added

        Args:
            vec_list (list[list[float]]): List of vectors.
                - If the vec_list is too large, it will be split into smaller chunks.
                - If the vec_list is too small or the index is too small, it will be the same as calling `add` multiple times.
            metadata_list (list[dict[str, str]]): List of metadata.
        """
        ...

    def __len__(self) -> int:
        """Return the number of vectors in the index."""
        ...

    def get_vec(self, id: int) -> list[float]:
        """Get the vector by id."""
        ...

    def get_metadata(self, id: int) -> dict[str, str]:
        """Get the metadata by id."""
        ...

    def set_metadata(self, id: int, metadata: dict[str, str]) -> None:
        """Set the metadata by id."""
        ...

    def search_as_pair(
        self,
        query: list[float],
        k: int,
        ef: int | None = None,
        max_distance: float | None = None,
    ) -> list[tuple[int, float]]:
        """Search for the nearest neighbors of a vector.

        Returns:
            A list of (id, distance) pairs.

        Args:
            query (list[float]): The query vector.
            k (int): The number of neighbors to search for.
            ef (int | None): Search radius. (default: None, auto)
            max_distance (float | None): Elements with a distance greater than this will be ignored. (default: None, no limit)
        """
        ...

    def search(
        self,
        query: list[float],
        k: int,
        ef: int | None = None,
        max_distance: float | None = None,
    ) -> list[dict[str, str]]:
        """Search for the nearest neighbors of a vector.

        Returns:
            A list of metadata.

        Args:
            query (list[float]): The query vector.
            k (int): The number of neighbors to search for.
            ef (int | None): Search radius. (default: None, auto)
            max_distance (float | None): Elements with a distance greater than this will be ignored. (default: None, no limit)
        """
        ...

class RagMultiVecDB:
    """A group of vector databases for automatic searching and merging KNN results."""

    def __init__(self, multi_vec_db: list[RagVecDB]) -> None:
        """Create a new multi-vector database."""
        ...

    def get_vec(self, db_id: int, vec_id: int) -> list[float]:
        """Get a vector by (db_id, vec_id)."""
        ...

    def get_metadata(self, db_id: int, vec_id: int) -> dict[str, str]:
        """Get the metadata by (db_id, vec_id)."""
        ...

    def search_as_pair(
        self,
        query: list[float],
        k: int,
        ef: int | None = None,
        max_distance: float | None = None,
    ) -> list[tuple[int, int, float]]:
        """Search for the nearest neighbors of a vector.

        Returns:
            A list of (db_id, vec_id, distance) tuples.

        Args:
            query (list[float]): The query vector.
            k (int): The number of neighbors to search for.
            ef (int | None): Search radius. (default: None, auto)
            max_distance (float | None): Elements with a distance greater than this will be ignored. (default: None, no limit)
        """
        ...

    def search(
        self,
        query: list[float],
        k: int,
        ef: int | None = None,
        max_distance: float | None = None,
    ) -> list[dict[str, str]]:
        """Search for the nearest neighbors of a vector.

        Returns:
            A list of metadata.

        Args:
            query (list[float]): The query vector.
            k (int): The number of neighbors to search for.
            ef (int | None): Search radius. (default: None, auto)
            max_distance (float | None): Elements with a distance greater than this will be ignored. (default: None, no limit)
        """
        ...
