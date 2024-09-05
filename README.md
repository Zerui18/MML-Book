# My MML Book Code

This repository contains my attempt to implement the mathematics in the book [Mathematics for Machine Learning](https://mml-book.github.io).

I divided the code according to the chapters in the book. For each chapter where concrete algorithms are discussed, I implement them in the corresponding chapter's module (e.g `chapter2.py`). For chapters with exercises, I implement the solutions (where possible) in the corresponding ipython notebook (e.g `chapter3.ipynb`).

Most of the algorithms are far from optimal, since the goal is to better understand the abstract concepts from the book by implementing them mostly as-is.

## Dependencies
### Required
 - `numpy`, which provides the basic, vectorized arithmetics of `np.ndarrray`.
 - `sympy`, which is used in finding the roots of characteristic polynomial in `chapter4.py`.

### Optional
 - `matplotlib`, which is used in plotting regression results in `chapter9.py`.

## Running the code
1. Clone the repository.
2. Install the dependencies using `pip install -r requirements.txt`.
3. Run an existing notebook or create your own.

## Lastly
Feel free to use the code in this repository for your own learning purposes. If you find any mistakes or have suggestions, please let me know by creating an issue or a pull request.