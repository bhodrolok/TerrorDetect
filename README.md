# terrordetect
Identifying potential criminals and terrorists from social  media analysis.

#### CPSC 571 Research Project
> By Group 20

### Installation and usage
1. Download the repo from the main branch as a zip file.
2. Unzip/Extract the downloaded file to a folder.
3. Navigate to the extracted folder
4. Open a command line in this folder and run 'pip install -r requirements.txt'.

_Brief explanation of files_
- 'combiner.py': Simply combines all csv files into one singular complete dataset. 
- 'datasetcreator.py': Program designed to generate individual datasets consisting of 10 tweets per keyword (as defined in the program as a keyword of commonly used extermists)
- 'naivebayes1.py': Program for Multinomial Naive Bayes and Logistic Regression classification models. Performs tweet cleanup/preprocessing and visualizes results (confustion matrix and accuracy prediction scores).
- 'rforest.py': Same as above but uses Random Forest Classification as the model instead.
- 'svm1.py': Uses Simple Vector Classification with the same method of generating the results and visualization.
