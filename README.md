# Processed IMDB WIKI Dataset

This GitHub repository contains complete preprocessed IMDB WIKI dataset in `.csv` files.

<p align="center">
  <img src="https://user-images.githubusercontent.com/34741145/51108233-75bac680-1817-11e9-8b79-6a1ee05d8aa4.png" />
</p>

## Table of contents:
- [Introduction](#introduction)
- [IMDB WIKI Dataset](#imdb-wiki-dataset)
- [The Problem](#the-problem)
- [The Solution](#the-solution)
- [File Structure](#file-structure)
- [How to Run Locally](#how-to-run-locally)
- [Acknowledgments](#acknowledgments)

## Introduction
IMDB WIKI dataset is the largest dataset of human faces with gender, name and age informations. In the dataset, images are of `.jpg` images and meta informations are in `.mat` files. 

In this project, I preprocessed all images, resized them, extract the meta informations from the `.mat` files and finally save them in `.csv` files into multiple batches.


## IMDB WIKI Dataset
IMDB WIKI dataset is the largest publically available dataset of human faces with gender, age, and name. It contains more than `500 thousand+` images with all the meta informations. 

For more information about the dataset please visit [this website](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/).

## The Problem
The dataset is great for research purposes. It contains more than `500 thousand+` images of faces. But some images do not have any faces, and some of the images are corrupted. So before using them, we need to filter those images.

Also, the metadata (description data) contains in `.mat` files. Reading `.mat` files in python is a tedious process.

## The Solution
In this project, I filter all the images, flattened them all, resized them all to `64x64`, and save them in multiple small `.csv` files. Along with that, I’ve also processed the `.mat` files and converted them in `.csv` files also.

## File Structure
This repository contains 4 files
 - `wiki_may.py`
 - `imdb_mat.py`
 - `wiki_image.py`
 - `imdb_image.py`

The first two files contain the code for processing all the `.mat` files and convert them into `.csv` files.

The last two files contain the code for processing all the images and saving them in `.csv` files.

These four files also expects 2 directories containing the main dataset in the following structure.

  - `processedData/images` - Contains the processed images as .csv files
  - `processedData/mat` - Contains the processed .mat files as .csv files
  - `unprocessedData/images` - Contains the unprocessd images
  - `unprocessedData/mat` - Contains the unprocessed meta file as .mat files

**As the size of the dataset is huge, I can not upload it here on GitHub**


## How to Run Locally
Following are the steps for running it locally
  - Download the dataset from [this](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) link and unzip it
  - Create a unprocessedData folder and save the images into an images folder
  - Locate the `.mat` files and save them under `unprocessedData/mat` folder
  - After this step, you should have the following directory structure
    - `unprocessedData/images/imdb_crop/`
    - `unprocessedData/images/wiki_crop/`
    - `unprocessedData/mat/imdb.mat`
    - `unprocessedData/mat/wiki.mat`
  - Now run the `wiki_meta.py` file
  - Now run the `wiki_image.py` file
  - Now you should have the processed data for `WIKI` dataset
  - Run the `imdb_meta.py` and `imdb_image.py` file to generate the data from `IMDB` dataset
  - Now you are ready for action with the dataset

## Acknowledgments
I really thankful to these peoples for providing this amazing dataset
  - [IMDB-WIKI – 500k+ face images with age and gender labels](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
  - [yu4u/age-gender-estimation](https://github.com/yu4u/age-gender-estimation)

