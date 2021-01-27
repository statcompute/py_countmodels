### py_countreg

#### Introduction

The package py\_countreg is a collection of functions to estimate various regression models for count outcomes.

It is an ongoing project. More functionalities will come later.

#### Core Functions

```
Count Outcome Regressions
  |
  |-- Equi-Dispersion (Baseline)
  |     |
  |     `-- stdpoisson() : Standard Poisson
  |
  |-- Over-Dispersion
  |     |
  |     `-- negbinom2()  : Negative Binomial (NB-2)
  |
  |-- Over- and Under-Dispersions
  |     |
  |     |-- genpoisson() : Generalized Poisson
  |     |
  |     `-- compoisson() : Conway-Maxwell Poisson
  |
  |-- Zero-Inflation
  |     |
  |     |-- hdlpoisson() : Hurdle Poisson
  |     |
  |     |-- hdlnegbin2() : Hurdle Negative Binomial (NB-2)
  |     |
  |     |-- zifpoisson() : Zero-Inflated Poisson
  |     |
  |     `-- zifnegbin2() : Zero-Inflated Negative Binomial (NB-2)
  |
  `-- Zero-Truncation
        |
        |-- ztrpoisson() : Zero-Truncated Poisson
        |
        |-- ztgpoisson() : Zero-Truncated Generalized Poisson
        |
        |-- ztcpoisson() : Zero-Truncated Conway-Maxwell Poisson
        |
        `-- ztrnegbin2() : Zero-Truncated Negative Binomial (NB-2)
```

#### Reference

WenSui Liu and Jimmy Cela (2008), Count Data Models in SAS, Proceedings SAS Global Forum 2008, paper 371-2008.
