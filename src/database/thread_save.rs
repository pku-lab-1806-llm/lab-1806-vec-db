use std::{
    path::{Path, PathBuf},
    sync::{Arc, Condvar, Mutex},
    thread, time,
};

/// Trait for objects that can be saved to a file in a background thread.
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

/// Auto-save the object to the file with a interval in a separate thread.
/// Sync the object to the file when it is dropped. (Or you can call `sync_save` manually.)
///
/// Order of mutex lock: mark -> obj -> stop_cond. (To avoid deadlock)
pub struct ThreadSavingManager<T: ThreadSave + 'static> {
    obj: Arc<T>,
    pub(crate) target: PathBuf,
    mark: Arc<Mutex<bool>>,
    stop_cond: (Arc<Mutex<bool>>, Arc<Condvar>),
}

impl<T: ThreadSave + 'static> ThreadSavingManager<T> {
    /// Create a new auto-saving manager.
    pub fn new(obj: Arc<T>, target: PathBuf, interval: time::Duration) -> Self {
        let mark = Arc::new(Mutex::new(false));
        let stop_cond = (Arc::new(Mutex::new(false)), Arc::new(Condvar::new()));
        {
            let obj = obj.clone();
            let target = target.clone();
            let mark = mark.clone();
            let stop_cond = stop_cond.clone();
            thread::spawn(move || loop {
                // Wait for the stop signal or the interval.
                let (stopped, _) = stop_cond
                    .1
                    .wait_timeout_while(stop_cond.0.lock().unwrap(), interval, |stopped| !*stopped)
                    .unwrap();
                if *stopped {
                    break;
                }
                drop(stopped);
                // Release the stop_cond lock before we lock the mark.
                // Equivalent to two separate tasks with correct lock order:
                // 1. _ -> _ -> stop_cond
                // 2. mark -> obj -> _
                let mut mark = mark.lock().unwrap();
                if *mark {
                    obj.atomic_save_to(&target);
                    *mark = false;
                }
            });
        }
        Self {
            obj,
            target,
            mark,
            stop_cond,
        }
    }
    /// Sync the object to the file if it is modified.
    /// If `stop_thread` is true, stop the auto-saving thread after syncing.
    pub fn sync_save(&self, stop_thread: bool) {
        // mark -> obj -> stop_cond
        let mut mark = self.mark.lock().unwrap();
        if *mark {
            self.obj.atomic_save_to(&self.target);
            *mark = false;
        }
        if stop_thread {
            let (lock, cond) = &self.stop_cond;
            let mut stopped = lock.lock().unwrap();
            *stopped = true;
            cond.notify_one();
        }
    }
    /// Mark the object as modified.
    pub fn mark_modified(&self) {
        let mut mark = self.mark.lock().unwrap();
        *mark = true;
    }
}
impl<T: ThreadSave + 'static> Drop for ThreadSavingManager<T> {
    fn drop(&mut self) {
        self.sync_save(true);
    }
}
