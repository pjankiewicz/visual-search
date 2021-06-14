use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};

#[derive(Clone)]
pub struct WorkQueue<T: Send + Clone + PartialEq> {
    pub inner: Arc<Mutex<VecDeque<T>>>,
}

impl<T: Send + Clone + PartialEq> PartialEq for WorkQueue<T> {
    fn eq(&self, other: &Self) -> bool {
        let this_inner = self.inner.lock().unwrap();
        let other_inner = other.inner.lock().unwrap();
        *this_inner == *other_inner
    }
}

impl<T: Send + PartialEq + Clone> WorkQueue<T> {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    fn aquire(&self) -> MutexGuard<'_, VecDeque<T>> {
        if let Ok(q) = self.inner.lock() {
            q
        } else {
            panic!("WorkQueue::get_work() tried to lock a poisoned mutex")
        }
    }

    pub fn get_work(&self) -> Option<T> {
        self.aquire().pop_front()
    }

    pub fn add_work(&self, work: T) -> usize {
        let mut q = self.aquire();
        if !q.contains(&work) {
            q.push_back(work);
        }
        q.len()
    }

    pub fn retain<F>(&self, f: F) -> VecDeque<T>
    where
        F: for<'a> FnMut(&'a T) -> bool,
    {
        let mut q = self.aquire().clone();
        q.retain(f);
        q
    }
}
