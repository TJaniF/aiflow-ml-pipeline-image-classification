Overview
========

Welcome to this example repository to get started creating combined ML and data pipelines with [Apache Airflow](https://airflow.apache.org/)! :rocket:

This repository contains a fully functional data and ML orchestration pipeline that can be run locally with the [Astro CLI](https://docs.astronomer.io/astro/cli/install-cli).

This Airflow pipeline will perform model fine-tuning and testing using [S3](https://aws.amazon.com/s3/), [DuckDB](https://duckdb.org/) and [HuggingFace](https://huggingface.co/).

Use this repository to explore Airflow, experiment with your own DAGs and as a template for your own projects, as well as your own custom operators and task groups!

This project was created with :heart: by [Astronomer](https://www.astronomer.io/).

> If you are looking for an entry level written tutorial where you build your own first Airflow DAG from scratch check out: [Get started with Apache Airflow, Part 1: Write and run your first DAG](https://docs.astronomer.io/learn/get-started-with-airflow).

-------------------------------

How to use this repository
==========================

Download the [Astro CLI](https://docs.astronomer.io/astro/cli/install-cli) to run Airflow locally in Docker. `astro` is the only package you will need to install.

If you are on a Mac, install the Astro CLI is as easy as the following steps:

1. Check that you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) and [Homebrew](https://brew.sh/) installed.
2. Run `brew install astro`.

And that is it, you can now use the repository:

1. Run `git clone https://github.com/TJaniF/airflow-ml-pipeline-image-classification.git` on your computer to create a local clone of this repository.
2. Run `astro dev start` in your cloned repository.
3. After your Astro project has started. View the Airflow UI at `localhost:8080`.
4. Set your own variables pointing at your source data in `include/config_variables.py`.
5. You likely will have to make some changes to the `standard_transform_function` depending on your source data, you can find it and other functions used in `include/utils/utils.py`.