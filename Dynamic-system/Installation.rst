===================================================
RMFlow: Trajectory Sampling for Dynamical Systems
===============================

Installation
============

#. Install ``uv``:

   .. code:: bash

      curl -LsSf https://astral.sh/uv/install.sh | sh

#. **Suggested:** Set the package cache directory of ``uv`` to a directory in a mounted drive.
   For example,

   .. code:: bash

      echo "export UV_CACHE_DIR=/root/workspace/out/uv-cache" >> ~/.bashrc
      source ~/.bashrc

#. Install ``cuDNN``:

   .. code:: bash

      apt-get install -y cudnn9-cuda-12

#. Install Python dependencies using ``uv``:

   .. code:: bash

      uv sync

#. Test your installation by running the unit tests:

   .. code:: bash

      pytest tests

Supplementary Documentation
===========================

* `Hydra <https://hydra.cc/docs/1.3/intro/>`_: Command-line inferface configuration library for configuring the experiments in this project.
* `Optuna <https://optuna.readthedocs.io/en/v4.2.0/index.html>`_: Library for tuning the XGBoost hyperparameters.
* `Optuna Dashboard <https://optuna-dashboard.readthedocs.io/en/stable/index.html>`_: Library for visualizing the results of Optuna's optimization.
* `PyTorch <https://pytorch.org/docs/2.5/index.html>`_: Library for implementing Zgraggen algorithm.
* `PyTorch Lightning <https://lightning.ai/docs/pytorch/2.5.0/>`_: Library for handles training the Zgraggen algorithm.

Training the models
===================

To train a model, run

.. code:: bash

   python src/userfm/main.py +experiment=<experiment> dataset=<dataset> model=<model>
