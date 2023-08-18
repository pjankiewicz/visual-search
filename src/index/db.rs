use core::mem;
use std::collections::HashSet;
use std::sync::{Arc, RwLock};

use hnsw::{Hnsw, Searcher};
use rand_pcg::Pcg64;
use space::{Metric, Neighbor};

const MAX_REMOVED_BEFORE_REBUILD: usize = 100;

#[derive(Clone)]
pub struct Euclidean;

impl Metric<Vec<f64>> for Euclidean {
    type Unit = u64;
    fn distance(&self, a: &Vec<f64>, b: &Vec<f64>) -> u64 {
        a.iter()
            .zip(b.iter())
            .map(|(&a1, &b1)| (a1 - b1).powi(2))
            .sum::<f64>()
            .sqrt()
            .to_bits() as u64
    }
}

#[derive(Debug)]
pub struct AnnNeighbor {
    pub id: String,
    pub index: usize,
    pub distance: u32,
}

#[derive(Clone)]
pub struct VectorIndex {
    pub searcher: Arc<RwLock<Searcher<u64>>>,
    pub hnsw: Arc<RwLock<Hnsw<Euclidean, Vec<f64>, Pcg64, 12, 24>>>,
    pub vectors: Arc<RwLock<Vec<(String, Vec<f64>)>>>,
    pub removed: Arc<RwLock<HashSet<String>>>,
}

impl VectorIndex {
    pub fn new() -> Self {
        VectorIndex {
            searcher: Arc::new(RwLock::new(Searcher::default())),
            hnsw: Arc::new(RwLock::new(Hnsw::new(Euclidean))),
            vectors: Arc::new(RwLock::new(Vec::new())),
            removed: Arc::new(Default::default()),
        }
    }

    pub fn insert(&self, v: Vec<f64>, id: String) {
        let mut hnsw = self.hnsw.write().unwrap();
        let mut searcher = self.searcher.write().unwrap();
        let mut vectors = self.vectors.write().unwrap();
        vectors.push((id, v.clone()));
        hnsw.insert(v, &mut searcher);
    }

    pub fn search(&self, v: &[f64]) -> Vec<AnnNeighbor> {
        let mut neighbors = [Neighbor {
            index: !0,
            distance: !0,
        }; 8];
        let mut searcher = self.searcher.write().unwrap();
        let hnsw = self.hnsw.write().unwrap();
        let removed = self.removed.read().unwrap();
        let vectors = self.vectors.read().unwrap();
        hnsw.nearest(&v.to_vec(), 8, &mut searcher, &mut neighbors);
        println!("Neighbors {:?}", neighbors);
        neighbors
            .iter()
            .filter(|e| {
                let id = e.index;
                let id_val = vectors[id].0.clone();
                !removed.contains(&id_val)
            })
            .cloned()
            .map(|n| AnnNeighbor {
                id: vectors[n.index].0.clone(),
                index: n.index,
                distance: n.distance as u32,
            })
            .collect()
    }

    pub fn remove(&self, id: String) {
        let mut removed = self.removed.write().unwrap();
        removed.insert(id);
        if removed.len() > MAX_REMOVED_BEFORE_REBUILD {
            self.rebuild();
        }
    }

    pub fn rebuild(&self) {
        let new_index = VectorIndex::new();
        let vectors = self.vectors.read().unwrap();
        let removed = self.removed.read().unwrap();
        for (id, v) in vectors.iter() {
            if !removed.contains(id) {
                new_index.insert(v.clone(), id.clone());
            }
        }

        drop(vectors);
        drop(removed);

        // replace all data structures inplace
        let mut searcher_old = self.searcher.write().unwrap();
        let mut hnsw_old = self.hnsw.write().unwrap();
        let mut vectors_old = self.vectors.write().unwrap();
        let mut removed_old = self.removed.write().unwrap();

        let searcher_new = new_index.searcher.read().unwrap();
        mem::replace(&mut *searcher_old, (*searcher_new).clone());

        let hnsw_new = new_index.hnsw.read().unwrap();
        mem::replace(&mut *hnsw_old, (*hnsw_new).clone());

        let vectors_new = new_index.vectors.read().unwrap();
        mem::replace(&mut *vectors_old, (*vectors_new).clone());

        removed_old.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_index() {
        let index = VectorIndex::new();
        let features: [&[f64]; 9] = [
            &[0.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0, 1.0],
            &[0.0, 0.0, 1.0, 1.0],
            &[0.0, 1.0, 1.0, 0.0],
            &[1.0, 1.0, 0.0, 0.0],
            &[0.0, 0.0, 1.0, 1.0],
            &[0.0, 1.0, 1.0, 0.0],
            &[1.0, 1.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 1.0],
        ];

        let mut i = 0;
        for &feature in &features {
            let v = feature.to_vec();
            index.insert(v, i.to_string());
            i += 1;
        }

        let neighbors = index.search(&features[0].to_vec().clone());
        assert_eq!(neighbors[0].id, "0".to_string());

        index.remove("0".to_string());
        let neighbors = index.search(&features[0].to_vec().clone());
        assert_eq!(neighbors[0].id, "1".to_string());

        index.rebuild();
        let neighbors = index.search(&features[0].to_vec().clone());
        assert_eq!(neighbors[0].id, "1".to_string());
    }
}
