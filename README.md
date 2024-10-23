# A Geometric Perspective for High-Dimensional Multiplex Graphs

High-dimensional multiplex graphs are characterized by their high number of complementary and divergent dimensions. The existence of multiple hierarchical latent relations between the graph dimensions poses significant challenges to embedding methods. In particular, the geometric distortions that might occur in the representational space have been overlooked in the literature. This work studies the problem of high-dimensional multiplex graph embedding from a geometric perspective. We find that the node representations reside on highly curved manifolds, thus rendering their exploitation more challenging for downstream tasks. Moreover, our study reveals that increasing the number of graph dimensions can cause further distortions to the highly curved manifolds. To address this problem, we propose a novel multiplex graph embedding method that harnesses hierarchical dimension embedding and Hyperbolic Graph Neural Networks. The proposed approach hierarchically extracts hyperbolic node representations that reside on Riemannian manifolds while gradually learning fewer and more expressive latent dimensions of the multiplex graph. Experimental results on real-world high-dimensional multiplex graphs show that the synergy between hierarchical and hyperbolic embeddings incurs much fewer geometric distortions and brings notable improvements over state-of-the-art approaches on downstream tasks.

## Requirements
Python 3.6 <br />
numpy <br />
scipy <br />
scikit-learn <br />
pytorch <br />
tqdm

## Run
To run experiments:

`bash run_all.sh`

## Reference
If you find this work useful in your research, please consider citing the following paper:

```
@inproceedings{abdous2024geometric,
  author = {Abdous, Kamel and Mrabah, Nairouz and Bouguessa, Mohamed},
  title = {A Geometric Perspective for High-Dimensional Multiplex Graphs},
  year = {2024},
  doi = {10.1145/3627673.3679541},
  booktitle = {Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  pages = {4–13},
  numpages = {10}
}
```
