def calc_dist(a: list[float], b: list[float], dist: str = "cosine") -> float:
    """
    Calculate the distance between two vectors.

    `dist` can be "l2sqr" or "cosine" (default: "cosine", for RAG).

    - l2sqr: squared Euclidean distance
    - cosine: cosine distance (1 - cosine_similarity) [0.0, 2.0]

    Raises:
        ValueError: If the distance function is invalid.
    """
    ...

class BareVecTable:
    """
    Bare Vector Database Table.

    Prefer using VecDB to manage multiple tables.
    """
    def __init__(self, dim: int, dist: str = "cosine") -> None:
        """
        Create a new Table. (Using HNSW internally)

        Args:
            dim (int): Dimension of the vectors.
            dist (str): Distance function. See `calc_dist` for details.

        Raises:
            ValueError: If the distance function is invalid.
        """
        ...

    def dim(self) -> int:
        """Get the dimension of the vectors."""
        ...

    def dist(self) -> str:
        """Get the distance algorithm name."""
        ...

    def __len__(self) -> int:
        """Get the number of vectors in the index."""
        ...

    @staticmethod
    def load(path: str) -> "BareVecTable":
        """Load an existing index from disk."""
        ...

    def save(self, path: str) -> None:
        """Save the index to disk."""
        ...

    def add(self, vec: list[float], metadata: dict[str, str]) -> None:
        """Add a vector to the index.
        Use `batch_add` for better performance."""
        ...

    def batch_add(
        self, vec_list: list[list[float]], metadata_list: list[dict[str, str]]
    ) -> None:
        """Add multiple vectors to the index."""
        ...

    def search(
        self,
        query: list[float],
        k: int,
        ef: int | None = None,
        upper_bound: float | None = None,
    ) -> list[tuple[dict[str, str], float]]:
        """Search for the nearest neighbors of a vector.
        Returns a list of (metadata, distance) pairs."""
        ...

    def build_hnsw_index(self, ef_construction: int | None = None) -> None:
        """Build HNSW index for the table."""
        ...

    def clear_hnsw_index(self) -> None:
        """Clear HNSW index for the table."""
        ...

    def has_hnsw_index(self) -> bool:
        """Check if the table has HNSW index."""
        ...

    def delete(self, pattern: dict[str, str]) -> None:
        """Delete vectors with metadata that match the pattern."""
        ...

class VecDB:
    """
    Vector Database. Prefer using this to manage multiple tables.

    Ensures:
    - Auto-save. The database will be saved to disk when necessary.
    - Parallelism. `allow_threads` is used to allow multi-threading.
    - Thread-safe. Read and write operations are atomic.
    - Unique. Only one manager for each database.
    """
    def __init__(self, dir: str) -> None:
        """
        Create a new VecDB, it will create a new directory if it does not exist.
        """
        ...

    def create_table_if_not_exists(
        self, key: str, dim: int, dist: str = "cosine"
    ) -> bool:
        """Create a new table if it does not exist.

        Args:
            key (str): The table name.
            dim (int): Dimension of the vectors.
            dist (str): Distance function. See `calc_dist` for details.

        Raises:
            ValueError: If the distance function is invalid.
        """
        ...

    def get_table_info(self, key: str) -> tuple[int, int, str]:
        """Get table info.

        Returns:
            (dim, len, dist)
        """
        ...

    def delete_table(self, key: str) -> bool:
        """
        Delete a table and waits for all operations to finish.
        Returns False if the table does not exist.
        """
        ...

    def get_all_keys(self) -> list[str]:
        """Get all table names."""
        ...

    def get_cached_tables(self) -> list[str]:
        """Returns a list of table keys that are cached."""
        ...

    def remove_cached_table(self, key: str) -> None:
        """Remove a table from the cache and wait for all operations to finish.
        Does nothing if the table is not cached."""
        ...

    def add(self, key: str, vec: list[float], metadata: dict[str, str]) -> None:
        """Add a vector to the table.
        Use `batch_add` for better performance."""
        ...

    def batch_add(
        self, key: str, vec_list: list[list[float]], metadata_list: list[dict[str, str]]
    ) -> None:
        """Add multiple vectors to the table."""
        ...

    def delete(self, key: str, pattern: dict[str, str]) -> None:
        """Delete vectors with metadata that match the pattern."""
        ...

    def search(
        self,
        key: str,
        query: list[float],
        k: int,
        ef: int | None = None,
        upper_bound: float | None = None,
    ) -> list[tuple[dict[str, str], float]]:
        """Search for the nearest neighbors of a vector.
        Returns a list of (metadata, distance) pairs."""
        ...

    def build_hnsw_index(self, key: str, ef_construction: int | None = None) -> None:
        """Build HNSW index for the table."""
        ...

    def clear_hnsw_index(self, key: str) -> None:
        """Clear HNSW index for the table."""
        ...

    def has_hnsw_index(self, key: str) -> bool:
        """Check if the table has HNSW index."""
        ...
