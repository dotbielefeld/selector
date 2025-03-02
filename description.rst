.. _description:

Description
===========


Selector :cite:`Selector` is an ensemble-based automated algorithm configuration method. It queries the integrated algorithm configuration (AC) methods to suggest configuations for the target algorithm. There are more suggestions made than will be evaluated. The set of suggestions is reduced by a learned model. For this, features computed from the suggested configurations are used. The selected configurations are then evaluated in tournaments. The results of the tournaments are stored and used as feedback for the suggesting AC methods.

The infrastructure regarding target algorithm evaluations and data of Selector is implemented with ray. Currently, there are three model based suggesters implemented based on:

- CPPL :cite:`CPPL`
- GGA :cite:`GGA++_paper`
- SMAC :cite:`SMAC3`

Additionally the following suggesters are implemented:

- Default parameter :cite:`GGA_paper, ParamILS`
- Random parameter :cite:`SMAC3`
- Latin Hypercube sampling :cite:`skopt`
- GGA graph crossover :cite:`aclib`

The selection is based on an iterative scoring selection mechanism :cite:`BBLION`. Find the source code at `Selector Github <https://github.com/DimitriWeiss/selector>`_.

References
==========

.. bibliography:: references.bib
   :style: plain