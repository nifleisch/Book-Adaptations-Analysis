# Statistical Analysis of Book Adaptations

This was my final project for the famous [Applied Data Analysis (ADA) course](https://epfl-ada.github.io/teaching/fall2024/cs401/) that I took during my exchange semester at EPFL. Although I entered the course confident in my understanding of data analysis, I still learned several new techniques that I applied to this project, such as using propensity scores to address confounding factors. Additionally, I leveraged the incredible Wikidata database to enrich my dataset with information on book adaptations that isn’t available anywhere else. To recreate the dataset used in my analysis, refer to the `create_dataset.ipynb` notebook, which details the process of loading and merging data from multiple sources. For a comprehensive statistical analysis, please see the `data_analysis.ipynb` notebook.

## Abstract
Ever since the advent of cinema, filmmakers have drawn inspiration from books. From 1968 to 2002, [35% of all English-language films originated from books](https://www.frontier-economics.com/media/vyfd1iz3/publishings-contribution-to-the-wider-creative-industries.pdf). Particularly in high-stakes Hollywood productions, often [exceeding $100 million](https://www.statista.com/statistics/1389936/breakdown-production-budget-hollywood-movies-worldwide/#:~:text=Out%20of%20the%2089%20English,under%20ten%20million%20U.S.%20dollars), executives favor book adaptations, capitalizing on the established fanbases of beloved books. This raises the question: what are the key elements of a successful book-to-film adaptation? Are there specific genres that lend themselves more easily to cinematic adaptation? Is it more effective to adapt recent bestsellers or time-honored classics? And, in light of blockbusters like Harry Potter and Lord of the Rings, should the focus be on serial adaptations rather than standalone novels?
To address these inquiries, we will analyze data from the [CMU Movie Summary Corpus](http://www.cs.cmu.edu/~ark/personas/), [TMDB](https://www.themoviedb.org), [Wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page) and [Goodreads](https://www.kaggle.com/datasets/mdhamani/goodreads-books-100k). Our goal is to use this comprehensive analysis to develop a model that can effectively identify the most promising book for a successful film adaptation.

## Research questions
**I) How do book adaptations compare to original movies?**
* Do movies based on books perform better at the box office?
* Are movies based on books better, according to IMDB ratings?

**II) What kind of books are turned into movies?**
* When where the adapted books published?
* Do books that are adapted into movies typically have higher ratings on Goodreads?
* Do filmmakers tend to favor shorter or longer books for movie adaptations?
* Are books that are adapted into movies typically more popular and critically acclaimed?
* Are there certain book genres that are more suitable for movie adaptations?
* Are books part of a series more suitable for movie adaptations than standalone books ?
* Is it possible to predict which books will be selected for movie adaptations? If so, can we predict which books will be adapted next?

**III) What makes a commercially successful book adaptation?**
* Is there a book genre that leads to successful adaptations?
* Do book adaptations on book series perform better at the box office?
* What is the ideal time frame between a book's publication and its adaptation?
* Can we predict the expected revenue of a future book adaptation? If so, which adaptation would perform best at the box office?


## Data
**Used Datasets.**
The CMU movie summary corpus dataset contains about 80'000 movies with informations like release dates, duration, genre and actors. For the purpose of our study we enriched it using data from multiple sources:
- `IMDb Ratings`: To quantify the audience reception of movies, we extend our dataset by IMDb scores
- `TMDB`: For most entries in the CMU movie dataset, information about the revenue is missing. We used data obtained from `The Movie Database (TMDB)` to fill missing values for revenue.
- `Wikidata`: To get the information on which movies are book adaptations we queried the Wikidata Graph Database using SPARQL. In addition we also queried additional information about the books such as release dates, number of page, genre, or author. That way, the characteristics of the book can be compared and linked to the one of the movie and its final success.
- `Goodreads`: We supplement our book data with a dataset scraped from Goodreads, which includes additional information such as a user ratings or number of pages for each book.
- `Consumer Price Index`: As we deal with revenue data from multiple decades, we need to normalize them to make them compareable. Therfore we use the US-Consumer Price Index (CPI).

**Preprocessing.**
In the `create_dataset.ipynb` notebook, we load, preprocess, and then combine these datasets. One of the main challenges in curating data for our project involved merging data from various sources. To address this, we employed several strategies to effectively combine the datasets.
- `CMU-Wikidata`: In the CMU dataset, each movie is uniquely identified by its `wikipedia_id`, which differs from the IDs provided by Wikidata for the same movies. We utilized the `wikimapper` package to bridge this gap, mapping the Wikidata IDs to their corresponding Wikipedia IDs.
- `Wikidata-IMDB-TMDB`: Wikidata maintains the imdb_id for nearly every movie, a piece of information that is also present in both the IMDB and TMDB datasets.
- `Wikidata-Goodreads`: Merging the book data from Wikidata with the Goodreads dataset posed a challenge due to the sporadic availability of Goodreads IDs in Wikidata. To overcome this, we opted to join the datasets using the combination (`book_author`, `book_title`), hypothesizing that this pair would serve as a nearly unique identifier for each book. To enhance the number of successful matches, we conducted preprocessing on both the title and author name in each dataset to standardize their formats. Manual verification of the join results confirmed the validity of this approach.

Given our focus on movies spanning a 60-year range, it was necessary to adjust the revenue and budget data for inflation. To achieve this, we utilized the US Consumer Price Index (CPI) from each respective year, transforming the data so that the revenue and budget values reflect today's prices.

**Final Datasets.**
For our study, we worked exclusively with two .csv files that were generated by the create_dataset.ipynb notebook.
- `book_adaptation.csv`: Contains metadata for 82,021 movies, covering details like (inflation-adjusted) revenue, runtime, and genre. Additionally, for the 4,686 movies identified as adaptations from Wikidata, we also include corresponding metadata about the original books.
- `book.csv`: Contains metadata for 36,960 books, obtained from Wikidata and enriched with the information from the Goodreads dataset.

## Methods
**Log-Transformation.**
We noted that movie_revenue, a crucial variable in our study, follows a power law distribution. Consequently, to normalize this data and facilitate more effective analysis, we applied a logarithmic transformation to it.

**Comparing Adapted against Original Movies.**
For the introduction section of our analysis, we aimed to isolate the effect of a movie being an adaptation on two aspects: 1) its revenue, and 2) its rating. We identified several potential confounding factors, including the movie's genre, release date, country of origin, and budget. We adressed them the following way:
- `Exact Matching`: Given the higher number of original movies compared to adapted ones in our dataset, we could afford to perform exact matching on important categorical variables to ensure a fair comparison.
- `Propensity Score Estimation`: We utilized a logistic model to calculate the propensity scores, which helped in assessing the likelihood of a movie getting the "treatment" adaptation based on its characteristics.
- `Maximum Weight Matching`: We employed a maximum weight matching technique. This approach allowed us to pair adapted movies with original movies having similar propensity scores, ensuring a more accurate comparison of their revenue and ratings.
We then utilized the matched dataset, created through this matching procedure, for both plotting and statistical analyses.

`T-Test`: To assess whether the observed difference in distribution is statistically significant, we employed a standard t-test. We considered a result to be significant if the p-value was below the 5% threshold.
`Linear Regression`: To evaluate the average impact of a movie being an adaptation, we fitted a linear regression model using either log_revenue or the rating as the dependent variable. The binary variable movie_is_adapted served as the sole predictor. We then considered the slope coefficient of this model to interpret the effect of adaptation on movie revenue or rating.

Similar techniques were used to answer the question in `1) What kind of books are turned into movies?`. However, for comparing variables such as publication year or genre, we did not deem it necessary to perform matching prior to conducting the analysis.

**Predicting which books will be adapted.**
Finally, we used the metadata we have about books to predict which ones will be adapted next. For this, we leveraged the books listed in `book_adaptation.csv` as positive examples, and those from `book.csv` as negative examples. We acknowledge that `book.csv` might include books that have already been adapted, but given the low proportion of books that get adapted, we deemed this acceptable. We encountered two main challenges, which we addressed as follows:
- Dual Use of book.csv Data: We aimed to use the examples from book.csv both as negative examples for training our classification model and for inference to predict their likelihood of being adapted. To manage this, we employed a `5-fold cross-validation` approach. This involved making predictions for the data in one fold using a model trained on the other four folds, ensuring that we did not use a book to train the model which we later use to make a prediction for that book.
- Imbalance in Adapted vs. Non-Adapted Books: We recognized a significant imbalance between the number of books that were adapted and those that were not. To address this, we utilized `stratified sampling` to ensure each fold in our cross-validation contained a similar proportion of adapted and non-adapted books. Furthermore, we `oversampled` the adapted books within our training process. This was to prevent the classifier from being biased towards predicting 'no adaptation' due to the larger number of non-adapted books in the dataset.
To deal with missing values we decided to use a `Imputer` that filled them with the respective median for each columns. In attition to that we used scikit-learns `StandardScaler` to normalize the predictors. Further we considered `Logistic Regression` as `Random Forest` for classification and for each fold computed the accuracy alongside the recall and precision.

**Predicting commercial success of a movie.**
First, we analyzed how various predictors influence revenue. To quantify the linear relationship between a continuous variable and the dependent variable, we employed the `Pearson correlation coefficient`, also reporting the respective p-values.

We opted for a simple `ridge regression` model to predict revenue, selecting this predictor due to its enhanced explainability.

## Setup
Create a virtual environment:
```
python -m venv venv
```

Activate the virtual environment:
For Linux/Mac
```
source venv/bin/activate
```
For Windows
```
venv/bin/activate
```

Install required packages:
```
pip install -r requirements.txt
```
Now you should be able to run both `create_dataset.ipynb` and `data_analysis.ipynb`.
