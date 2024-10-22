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
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
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
    /// Save to a tmp file and rename it to the target file.
    fn atomic_save_to(&self, path: impl AsRef<Path>) {
        let tmp_dir = tempfile::tempdir().unwrap();
        let mut tmp = tmp_dir.path().join("tmp");
        if let Some(ext) = path.as_ref().extension() {
            tmp.set_extension(ext);
        }
        self.save_to(&tmp);
        if tmp.is_file() {
            std::fs::rename(&tmp, path).unwrap();
        }
    }
}

pub struct ThreadSavingManager<T: ThreadSave + 'static> {
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
    pub fn sync_save(&self) {
        let mut wait_list = self.wait_list.lock().unwrap();
        if !wait_list.is_empty() {
            self.obj.atomic_save_to(&self.target);
            wait_list.clear();
        }
    }
    pub fn signal(&self) {
        // target: /path/to/file.db
        let target = self.target.clone();
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
                    obj.atomic_save_to(&target);
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
                    obj.atomic_save_to(&target);
                    wait_list.clear();
                }
            }
        });
    }
}
impl<T: ThreadSave + 'static> Drop for ThreadSavingManager<T> {
    fn drop(&mut self) {
        self.sync_save();
    }
}
