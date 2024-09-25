use serde::{Deserialize, Serialize};

/// The reusable data for the visited list.
///
/// Remember to call `setup` before each time you use it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReusableVisited {
    /// The turn of the visited list.
    turn: usize,
    /// The visited list. `true` if `visited[i] == turn`.
    visited: Vec<usize>,
}
impl ReusableVisited {
    pub fn new() -> Self {
        Self {
            turn: 0,
            visited: Vec::new(),
        }
    }
    pub fn is_visited(&self, index: usize) -> bool {
        self.visited[index] == self.turn
    }
    pub fn visit(&mut self, index: usize) {
        self.visited[index] = self.turn;
    }
    /// Setup the visited list for the new search.
    ///
    /// Exactly O(1) when `n` is the same as the last time.
    pub fn setup(&mut self, n: usize) {
        self.turn += 1;
        self.visited.resize(n, 0);
    }
}
