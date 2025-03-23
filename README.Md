# Movie Recommendation System

This project is a machine learning-based movie recommendation system that suggests movies to users based on their preferences, viewing history, and ratings. It leverages both content-based and collaborative filtering techniques to provide personalized movie recommendations.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

Recommender systems are integral to various online platforms, helping users discover products, services, or content that align with their preferences. This Movie Recommendation System aims to enhance the user experience by suggesting movies that match individual tastes, utilizing machine learning algorithms to analyze user behavior and movie attributes.

## Features

- **Personalized Recommendations**: Provides movie suggestions tailored to individual user preferences.
- **Content-Based Filtering**: Analyzes movie features such as genre, actors, and plot descriptions to recommend similar movies.
- **Collaborative Filtering**: Utilizes user-item interactions to identify patterns and suggest movies that similar users have enjoyed.
- **Hybrid Approach**: Combines content-based and collaborative filtering methods to improve recommendation accuracy.

## Dataset

The system utilizes the [MovieLens dataset](https://grouplens.org/datasets/movielens/), a widely-used dataset in recommender system research. It contains user ratings and metadata for thousands of movies, providing a robust foundation for building and evaluating recommendation algorithms. For instance, the [MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset) consists of 20 million ratings from 138,000 users on 27,000 movies.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Harshad/Smart-Movie-Recommendation-System.git
   cd movie-recommendation-system
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

After installation, you can start the application using Gradio for the user interface:

```bash
python app.py
```



This will launch a web-based interface where users can input their preferences to receive tailored movie recommendations.


