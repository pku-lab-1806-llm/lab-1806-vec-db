from typing import Literal

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

    def get_len(self, key: str) -> int:
        """Get the number of vectors in the table."""
        ...

    def get_dim(self, key: str) -> int:
        """Get the dimension of the vectors in the table."""
        ...

    def get_dist(self, key: str) -> str:
        """Get the distance function of the table."""
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

    def contains_key(self, key: str) -> bool:
        """Check if a table exists."""
        ...

    def get_cached_tables(self) -> list[str]:
        """Returns a list of table keys that are cached."""
        ...

    def contains_cached(self, key: str) -> bool:
        """Check if a table is cached."""
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

    def extract_data(self, key: str) -> list[tuple[list[float], dict[str, str]]]:
        """Extract all vectors and metadata from the table."""
        ...

    def build_hnsw_index(self, key: str, ef_construction: int | None = None) -> None:
        """Build HNSW index for the table. Skip when already built."""
        ...

    def clear_hnsw_index(self, key: str) -> None:
        """Clear HNSW index for the table."""
        ...

    def has_hnsw_index(self, key: str) -> bool:
        """Check if the table has HNSW index."""
        ...

    def build_pq_table(
        self,
        key: str,
        train_proportion: float | None = None,
        n_bits: Literal[4, 8] | None = None,
        m: int | None = None,
    ):
        """Build PQ table for the table. Skip when already built.

        Args:
            train_proportion: The proportion of vectors used for training. Range: (0.0, 1.0), default is 0.1.
            n_bits: The number of bits per sub-vector, can be 4 or 8, default is 4.
            m: The number of sub-vectors, Range: 1..=dim, default is ceil(dim / 3).
        """
        ...

    def clear_pq_table(self, key: str) -> None:
        """Clear PQ table for the table."""
        ...

    def has_pq_table(self, key: str) -> bool:
        """Check if the table has PQ table."""
        ...
