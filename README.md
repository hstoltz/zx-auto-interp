# About

This is code for finding interpretations which establish the necessity of rules in the ZX calculus, using `Mat_R`. It allows one to:

- Introduce green and red spiders `Z(n,m,k), X(n,m,k): n → m`, along with `H`, `id`, `swap`, `cup`, and `cap`. Here `n,m ∈ ℕ`, including zero.
- Specify equations between these typed maps, using `∘` and `⊗`.
- Choose a ring `R`, dimension `d`, and a rule to exclude.
- Search for a Hadamard matrix and phase values which satisfy all the other rules, and check that the excluded rule is not satisfied.

We assume `Hᵀ = H`, `H² = I`, and invertible phase values. Thus, the connectivity meta-rule, spider fusion, and color change are satisfied under this interpretation. When such an interpretation is found, the code also gives a matrix coefficient where the excluded rule fails.

# How to use

## Run

```bash
python3 zx.py bialgebra.zx --ring F17 --dim 8 --period 4 --target B
```

This searches for an interpretation in which (B) is not satisfied over `F17`, in dimension `8`, with phases modulo `4`.

You can change the search as follows:

- Edit the equations in `bialgebra.zx`, or use `phase_free.zx` for the phase-free fragment.
- `--ring` and `--dim` specify the coefficient ring and dimension.
- `--period 4` requires fourth-root phases; `--period 1` gives the phase-free case; `--period 0` allows integer-labelled phases.
- `--target B` specifies the rule to exclude.
- The resulting interpretation is saved in `certificate.json`. It can be checked separately by running `python3 zx.py bialgebra.zx --check certificate.json`.

Some other choices of rings and dimensions which also establish the necessity of (B) for the supplied rules are:

- `F9`, `d = 8`, phases modulo `4`.
- `F9`, `d = 4`, phase-free.
- `F5`, `d = 6`, phase-free.
- `Z9`, `d = 4`, phase-free.
- `F3eps3`, `d = 4`, phase-free, over `F₃[ε]/(ε³)`.

For example, in the phase-free case:

```bash
python3 zx.py phase_free.zx --ring F9 --dim 4 --period 1 --target B
```

If no suitable interpretation is found, this does not establish that the rule is derivable.
