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
            dim: Dimension of the vectors.
            dist: Distance function. Can be "l2sqr", "l2" or "cosine". (default: "cosine", for RAG)
            ef_construction: Number of elements to consider during construction. (default: 200)
            M: Number of neighbors to consider during search. (default: 16)
            max_elements: The initial capacity of the index. (default: 0, auto-grow)
            seed: Random seed for the index. (default: None, random)

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
        """Add a vector to the index.

        Returns the ID of the added vector.

        Use `batch_add` for better performance.
        """
        ...

    def batch_add(
        self, vec_list: list[list[float]], metadata_list: list[dict[str, str]]
    ) -> list[int]:
        """Add multiple vectors to the index.

        Returns the id list of the added vectors.

        If the vec_list is too large, it will be split into smaller chunks.
        If the vec_list is too small or the index is too small, it will be the same as calling `add` multiple times.
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

    def search_as_id(
        self, query: list[float], k: int, ef: int | None = None
    ) -> list[int]:
        """Search for the nearest neighbors of a vector, and return the ids."""
        ...

    def search(
        self, query: list[float], k: int, ef: int | None = None
    ) -> list[dict[str, str]]:
        """Search for the nearest neighbors of a vector.

        Returns a list of metadata.
        """
        ...
