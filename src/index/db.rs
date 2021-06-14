use core::mem;
use std::collections::HashSet;
use std::sync::{Arc, RwLock};

use hnsw::{Hnsw, Searcher};
use rand_pcg::Pcg64;
use space::{MetricPoint, Neighbor};

const MAX_REMOVED_BEFORE_REBUILD: usize = 100;

#[derive(Clone)]
pub struct Euclidean(Vec<f32>);

impl MetricPoint for Euclidean {
    fn distance(&self, rhs: &Self) -> u32 {
        space::f32_metric(
            self.0
                .iter()
                .zip(rhs.0.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt(),
        )
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
    pub searcher: Arc<RwLock<Searcher>>,
    pub hnsw: Arc<RwLock<Hnsw<Euclidean, Pcg64, 12, 24>>>,
    pub vectors: Arc<RwLock<Vec<(String, Vec<f32>)>>>,
    pub removed: Arc<RwLock<HashSet<String>>>,
}

impl VectorIndex {
    pub fn new() -> Self {
        VectorIndex {
            searcher: Arc::new(RwLock::new(Searcher::default())),
            hnsw: Arc::new(RwLock::new(Hnsw::new())),
            vectors: Arc::new(RwLock::new(Vec::new())),
            removed: Arc::new(Default::default()),
        }
    }

    pub fn insert(&self, v: Vec<f32>, id: String) {
        let mut hnsw = self.hnsw.write().unwrap();
        let mut searcher = self.searcher.write().unwrap();
        let mut vectors = self.vectors.write().unwrap();
        vectors.push((id, v.clone()));
        hnsw.insert(Euclidean(v), &mut searcher);
    }

    pub fn search(&self, v: &Vec<f32>) -> Vec<AnnNeighbor> {
        let mut neighbors = [Neighbor::invalid(); 8];
        let mut searcher = self.searcher.write().unwrap();
        let mut hnsw = self.hnsw.write().unwrap();
        let removed = self.removed.read().unwrap();
        let vectors = self.vectors.read().unwrap();
        hnsw.nearest(&Euclidean(v.to_vec()), 24, &mut searcher, &mut neighbors);
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
                distance: n.distance,
            })
            .collect()
    }

    pub fn remove(&self, id: String) {
        let mut removed = self.removed.write().unwrap();
        removed.insert(id.clone());
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
        let features: [&[f32]; 9] = [
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

        let neighbors = index.search(features[0].to_vec().clone());
        assert_eq!(neighbors[0].id, "0".to_string());

        index.remove("0".to_string());
        let neighbors = index.search(features[0].to_vec().clone());
        assert_eq!(neighbors[0].id, "1".to_string());

        index.rebuild();
        let neighbors = index.search(features[0].to_vec().clone());
        assert_eq!(neighbors[0].id, "1".to_string());
    }
}
