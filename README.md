# terrordetect
Identifying potential criminals and terrorists from social  media analysis.

#### CPSC 571 Research Project

> By Group 20
-  The project report is available [here](https://github.com/Bhodrolok/TerrorDetect/blob/main/CPSC571-ProjRep-G20.pdf).

### Requirements

- Latest version of [Python](https://www.python.org/downloads/)
    - NB: Make sure that `pip` is also installed alongside the Python interpreter.
- ( :heavy_exclamation_mark: Optional :heavy_exclamation_mark: ) [Git](https://www.git-scm.com/downloads).
    - NB: Refer to the next section for details.

### Installation and usage
1. Either clone the [repository](https://github.com/bhodrolok/TerrorDetect.git) or download it from the main branch as a ZIP file.
    - NB: If going with the former method, [Git](https://www.git-scm.com/downloads) is required.
    - ![image](https://github.com/bhodrolok/TerrorDetect/assets/51386657/294342b0-590f-49d2-95df-af56e472fb7c)
2. Unzip and extract the ZIP file to a folder somewhere in your local drive.
    - If cloned, skip this step.
3. Navigate to the extracted/cloned folder.
    - Unless changed, it should be 'TerrorDetect'
5. Open a terminal in this new folder, type and run the following command:
   ```console
    $ pip install -r requirements.txt
   ``` 

### Quick rundown of files included
- [combiner.py](./combiner.py): Utility module for combining all the gathered `.csv` files into a singular dataset. 
- [datasetcreator.py](./datasetcreator.py): Module for generating _individual_ datasets consisting of 10 tweets per keyword
    - as defined in the program as a keyword of commonly used extermists
- [naivebayes](./naivebayes.py): Module which uses [Multinomial Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Multinomial_naive_Bayes) and [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression) classification models.
    - Also performs tweet cleanup and preprocessing as well as visualizes the results
        - confustion matrix and accuracy prediction scores
- [rforest.py](./rforest.py): Same as above but uses the [Random Forest](https://en.wikipedia.org/wiki/Random_forest) Classification model instead.
- [svm.py](./svm.py): Uses [Simple Vector Classification](https://en.wikipedia.org/wiki/Support_vector_machine) with the same method of generating the results and visualization.
