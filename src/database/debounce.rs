use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    thread, time,
};
pub struct DebounceWaitList {
    list: BTreeSet<usize>,
    count: usize,
}
impl DebounceWaitList {
    pub fn new() -> Self {
        Self {
            list: BTreeSet::new(),
            count: 0,
        }
    }
    pub fn new_task(&mut self) -> usize {
        self.count += 1;
        self.list.insert(self.count);
        self.count
    }
    pub fn is_last(&self, task_id: usize) -> bool {
        self.list.last() == Some(&task_id)
    }
    pub fn contains(&self, task_id: usize) -> bool {
        self.list.contains(&task_id)
    }
    pub fn clear(&mut self) {
        self.list.clear();
    }
}

pub trait ThreadSave: Send + Sync {
    fn save_to(&self, path: impl AsRef<Path>);
}

pub struct ThreadSavingManager<T> {
    pub(crate) target: PathBuf,
    delay: time::Duration,
    obj: Arc<T>,
    max_wait: Option<time::Duration>,
    wait_list: Arc<Mutex<DebounceWaitList>>,
}

impl<T: ThreadSave + 'static> ThreadSavingManager<T> {
    pub fn new(
        target: PathBuf,
        delay: time::Duration,
        obj: Arc<T>,
        max_wait: Option<time::Duration>,
    ) -> Self {
        Self {
            target,
            delay,
            obj,
            max_wait,
            wait_list: Arc::new(Mutex::new(DebounceWaitList::new())),
        }
    }
    pub fn force_save(&self) {
        self.obj.save_to(&self.target);
    }
    pub fn save(&self) {
        // target: /path/to/file.db
        let target = self.target.clone();
        // tmp: /path/to/file.tmp
        let tmp = target.with_extension("tmp").to_str().unwrap().to_string();
        let obj = self.obj.clone();
        let wait_list = self.wait_list.clone();
        let delay = self.delay.clone();
        let max_wait = self.max_wait.clone();
        thread::spawn(move || {
            let task_id = wait_list.lock().unwrap().new_task();
            thread::sleep(delay);
            {
                let mut wait_list = wait_list.lock().unwrap();
                if wait_list.is_last(task_id) {
                    obj.save_to(&tmp);
                    std::fs::rename(&tmp, &target).unwrap();
                    wait_list.clear();
                    return;
                }
            }
            if let Some(max_wait) = max_wait {
                if max_wait > delay {
                    thread::sleep(max_wait - delay);
                }
                let mut wait_list = wait_list.lock().unwrap();
                if wait_list.contains(task_id) {
                    obj.save_to(&tmp);
                    std::fs::rename(&tmp, &target).unwrap();
                    wait_list.clear();
                }
            }
        });
    }
}
