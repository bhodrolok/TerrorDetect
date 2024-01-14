# terrordetect
Identifying potential criminals and terrorists from social  media analysis.

#### CPSC 571 Research Project
> By Group 20
-  The project report is available [here](https://github.com/Bhodrolok/TerrorDetect/blob/main/CPSC571-ProjRep-G20.pdf).

### Installation and usage
1. Either clone the [repository](https://github.com/bhodrolok/TerrorDetect.git) or download it from the main branch as a zip file.
    - ![image](https://github.com/bhodrolok/TerrorDetect/assets/51386657/294342b0-590f-49d2-95df-af56e472fb7c)
3. Unzip and extract the downloaded file to a folder somewhere in your local drive.
4. Navigate to the extracted folder.
   - If cloned from step 1, navigate to the same folder.
6. Open a command line in this folder and run:
   ```console
    $ pip install -r requirements.txt
   ```

_Brief explanation of files_
- 'combiner.py': Simply combines all csv files into one singular complete dataset. 
- 'datasetcreator.py': Program designed to generate individual datasets consisting of 10 tweets per keyword (as defined in the program as a keyword of commonly used extermists)
- 'naivebayes1.py': Program for Multinomial Naive Bayes and Logistic Regression classification models. Performs tweet cleanup/preprocessing and visualizes results (confustion matrix and accuracy prediction scores).
- 'rforest.py': Same as above but uses Random Forest Classification as the model instead.
- 'svm1.py': Uses Simple Vector Classification with the same method of generating the results and visualization.
