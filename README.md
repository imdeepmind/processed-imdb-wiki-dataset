# Processed IMDB WIKI Dataset

This GitHub repository contains a preprocessed IMDB WIKI dataset.

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
- [Dependencies](#dependencies)
- [Acknowledgments](#acknowledgments)

## Introduction
IMDB WIKI dataset is the largest dataset of human faces with gender, name and age information. In this project, I preprocessed the entire dataset so that it can be used easily without any problems.


## IMDB WIKI Dataset
IMDB WIKI dataset is the largest publically available dataset of human faces with gender, age, and name. It contains more than `500 thousand+` images with all the meta information. All the images are in `.jpg` format. 

For more information about the dataset please visit [this website](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/).

## The Problem
The dataset is great for research purposes. It contains more than `500 thousand+` images of faces. But the dataset is not ready for any Machine Learning algorithm. There are some problems with the dataset. 

  - All the images are of different size
  - Some of the images are completely corrupted
  - Some images don't have any faces
  - Some of the ages are invalid
  - The distribution between the gender is not equal(there are more male faces than female faces)
  - Also, the meta information is in `.mat` format. Reading `.mat` files in python is a tedious process.

## The Solution
In this project, I filter all the images, resized them all to `128x128`, remove all the images with invalid age, fix the gender distribution problem, and save them in the proper format. Along with that, I’ve also processed the `.mat` files and converted them in `.csv` files also.

## File Structure
This repository contains 3 files
 - `mat.py`
 - `gender.py`
 - `age.py`

The first `mat.py` file converts the mat files IMDB and WIKI dataset to `.csv` format and merge them into one file.  

The last two file process the images for gender and age classification.

**As the size of the dataset is huge, I can not upload it here on GitHub**


## How to Run Locally
Following are the steps for running it locally
  - Download the dataset from [this](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) link and unzip it
  - Extract the dataset and save it in the project directory
  - After that, you should have the following folders
    - `imdb_crop`
    - `wiki_crop`
  - Run the `mat.py` file
  - Run `age.py` and `gender.py` file
  - Now the dataset in preprocessed and ready for your project

## Dependencies
  - `Numpy=1.15.4`
  - `Scipy=1.2.0`
  - `pandas=0.23.4`
  - `cv2=4.0.0`

## Acknowledgments
I really thankful to these peoples for providing this amazing dataset
  - [IMDB-WIKI – 500k+ face images with age and gender labels](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
  - [yu4u/age-gender-estimation](https://github.com/yu4u/age-gender-estimation)

