use std::{
    path::{Path, PathBuf},
    sync::{Arc, Condvar, Mutex, RwLock},
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
            std::fs::copy(&tmp, path).unwrap();
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
    pub fn new(obj: T, target: impl AsRef<Path>, interval: time::Duration, mark: bool) -> Self {
        let mark = Arc::new(Mutex::new(mark));
        let stop_cond = (Arc::new(Mutex::new(false)), Arc::new(Condvar::new()));
        let obj = Arc::new(obj);
        let target = target.as_ref().to_path_buf();
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
}
impl<T: ThreadSave + 'static> Drop for ThreadSavingManager<T> {
    fn drop(&mut self) {
        self.sync_save(true);
    }
}
impl<T> ThreadSavingManager<RwLock<T>>
where
    RwLock<T>: ThreadSave + 'static,
{
    /// Create a new auto-saving manager for a RwLock.
    pub fn new_rw(obj: T, target: impl AsRef<Path>, interval: time::Duration, mark: bool) -> Self {
        Self::new(RwLock::new(obj), target, interval, mark)
    }
    pub fn read(&self) -> std::sync::RwLockReadGuard<'_, T> {
        self.obj.read().unwrap()
    }
    pub fn write(&self) -> std::sync::RwLockWriteGuard<'_, T> {
        let mut mark = self.mark.lock().unwrap();
        let guard = self.obj.write().unwrap();
        *mark = true;
        guard
    }
}
impl<T> ThreadSavingManager<Mutex<T>>
where
    Mutex<T>: ThreadSave + 'static,
{
    /// Create a new auto-saving manager for a Mutex.
    pub fn new_mutex(
        obj: T,
        target: impl AsRef<Path>,
        interval: time::Duration,
        mark: bool,
    ) -> Self {
        Self::new(Mutex::new(obj), target, interval, mark)
    }
    pub fn lock(&self) -> std::sync::MutexGuard<'_, T> {
        let mut mark = self.mark.lock().unwrap();
        let guard = self.obj.lock().unwrap();
        *mark = true;
        guard
    }
}
