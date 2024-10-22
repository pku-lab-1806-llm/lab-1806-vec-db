def calc_dist(a: list[float], b: list[float], dist: str = "cosine") -> float:
    """
    Calculate the distance between two vectors.

    `dist` can be "l2sqr", "l2" or "cosine". (default: "cosine", for RAG)

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
            dist (str): Distance function. Can be "l2sqr", "l2" or "cosine". (default: "cosine", for RAG)

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
        """
        Load an existing index from disk.

        Raises:
            RuntimeError: If the file is not found or the index is corrupted.
        """
        ...

    def save(self, path: str) -> None:
        """
        Save the index to disk.

        Raises:
            RuntimeError: If the file cannot be written.
        """
        ...

    def add(self, vec: list[float], metadata: dict[str, str]) -> int:
        """
        Add a vector to the index.

        Returns the ID of the added vector.

        Use `batch_add` for better performance.
        """
        ...

    def batch_add(
        self, vec_list: list[list[float]], metadata_list: list[dict[str, str]]
    ) -> list[int]:
        """
        Add multiple vectors to the index.
        Returns the id list of the added vectors.

        If the vec_list is too large, it will be split into smaller chunks.
        If the vec_list is too small or the index is too small, it will be the same as calling `add` multiple times.
        """
        ...

    def search(
        self,
        query: list[float],
        k: int,
        ef: int | None = None,
        upper_bound: float | None = None,
    ) -> list[tuple[dict[str, str], float]]:
        """
        Search for the nearest neighbors of a vector.

        Returns a list of (metadata, distance) pairs.
        """
        ...

class VecDB:
    """
    Vector Database.

    Prefer using this to manage multiple tables.
    """
    def __init__(self, dir: str) -> None:
        """
        Create a new VecDB, it will create a new directory if it does not exist.

        Automatically save the database to disk when dropped. Cache the tables when accessing their contents.
        """
        ...

    def create_table_if_not_exists(
        self, name: str, dim: int, dist: str = "cosine"
    ) -> bool:
        """
        Create a new table if it does not exist.

        Raises:
            RuntimeError: If the file is corrupted.
        """
        ...

    def get_table_info(self, key: str) -> tuple[int, int, str]:
        """
        Get table info.

        Returns:
            (dim, len, dist)

        Raises:
            RuntimeError: If the table is not found.
        """
        ...

    def delete_table(self, key: str) -> None:
        """
        Delete a table.

        The file is not deleted immediately. When a new table with the same name is created, the old file will be overwritten.
        """
        ...

    def get_all_keys(self) -> list[str]:
        """Get all table names."""
        ...

    def add(self, key: str, vec: list[float], metadata: dict[str, str]) -> int:
        """Add a vector to the table."""
        ...

    def batch_add(
        self, key: str, vec_list: list[list[float]], metadata_list: list[dict[str, str]]
    ) -> list[int]:
        """Add multiple vectors to the table."""
        ...

    def search(
        self,
        key: str,
        query: list[float],
        k: int,
        ef: int | None = None,
        upper_bound: float | None = None,
    ) -> list[tuple[dict[str, str], float]]:
        """
        Search for the nearest neighbors of a vector.

        Returns a list of (metadata, distance) pairs.
        """
        ...

    def join_search(
        self,
        key_list: set[str],
        query: list[float],
        k: int,
        ef: int | None = None,
        upper_bound: float | None = None,
    ) -> list[tuple[str, dict[str, str], float]]:
        """
        Search for the nearest neighbors of a vector in multiple tables.

        Returns a list of (table_name, metadata, distance) pairs.
        """
        ...
